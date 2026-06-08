"""
gwgpu_jax_likelihood_utils.py
==========================
High-level likelihood utilities for parameter estimation and inference.

This module provides JAX-based functions for:
  * Projecting waveforms onto detectors (h = Fp·hp + Fc·hc, with time delay).
  * Computing matched-filter inner products in the frequency domain.
  * Computing Gaussian log-likelihoods for single detectors and networks.

All functions are pure, JAX-traceable, and safe under ``jax.jit`` / ``jax.vmap``
/ ``jax.grad``.

Performance note — pre-computed inverse-noise weight
----------------------------------------------------
The inner product

    ⟨a|b⟩ = 4 · df · Σ_k m_k · Re(a*_k · b_k) / S_n(f_k)

is rewritten using a per-bin **weight** array

    w_k = 4 · df · m_k / S_n(f_k)            (zeroed where S_n is invalid)

so that the hot path collapses to

    ⟨a|b⟩ = Σ_k Re(a*_k · b_k) · w_k

``w_k`` depends only on the PSD, the band mask, and ``df`` — all of which
are fixed for a given IFO once the grid and PSD are set. Sampling loops
should therefore compute ``w`` **once** via :func:`precompute_weight` and
pass it into :func:`inner_product`, :func:`log_likelihood_ifo`,
:func:`optimal_snr_squared`, etc. Each saves an ``isfinite`` check and a
``1/x`` per FD bin per likelihood evaluation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ── Waveform projection ───────────────────────────────────────────────────────

def waveform_projection_fd(
    hp: jnp.ndarray,
    hc: jnp.ndarray,
    Fp,
    Fc,
    freqs: jnp.ndarray,
    time_delay=0.0,
) -> jnp.ndarray:
    """Project (h+, h×) onto a single detector in the frequency domain.

    h(f) = (Fp · hp(f) + Fc · hc(f)) · exp(-2πi·f·Δt)

    The phase shift is always applied; for ``time_delay == 0`` it collapses
    to ``1`` and incurs only a multiplication. Keeps the function jit-safe
    when ``time_delay`` is a traced JAX scalar.
    """
    phase = jnp.exp(-2j * jnp.pi * freqs * time_delay)
    return (Fp * hp + Fc * hc) * phase


# ── Pre-computed inverse-noise weight ────────────────────────────────────────

def precompute_weight(
    psd: jnp.ndarray,
    df,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the inner-product weight ``w = 4·df · m / S_n``.

    Bins where ``psd`` is non-finite or non-positive (DC, below f_min,
    fp32 overflow) carry zero weight, so they cannot contaminate the sum
    with NaN/inf. Out-of-band bins (``mask == False``) also carry zero
    weight.

    Parameters
    ----------
    psd : jnp.ndarray, shape (Nf,)
        One-sided PSD [Hz^{-1}].
    df : scalar
        Frequency resolution [Hz].
    mask : jnp.ndarray of bool, shape (Nf,), optional
        Analysis band. ``None`` ⇒ all bins enabled.

    Returns
    -------
    weight : jnp.ndarray, shape (Nf,), real
    """
    inv_psd = jnp.where(jnp.isfinite(psd) & (psd > 0.0), 1.0 / psd, 0.0)
    w = 4.0 * df * inv_psd
    if mask is not None:
        w = jnp.where(mask, w, 0.0)
    return w


# ── Matched-filter inner product ──────────────────────────────────────────────

def inner_product(
    a: jnp.ndarray,
    b: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ⟨a|b⟩ = Σ_k Re(a*_k · b_k) · w_k.

    The integrand is explicitly zeroed at bins where ``weight == 0`` so
    that NaN/inf values in ``a`` or ``b`` outside the analysis band
    (e.g. waveform NaN at DC, PSD overflow under f_min) cannot
    contaminate the sum via ``NaN · 0``.

    Parameters
    ----------
    a, b : jnp.ndarray, shape (Nf,)
        Complex FD arrays.
    weight : jnp.ndarray, shape (Nf,)
        Pre-computed weight from :func:`precompute_weight`.
    """
    integrand = (jnp.conj(a) * b).real
    return jnp.sum(jnp.where(weight > 0.0, integrand * weight, 0.0))


def optimal_snr_squared(
    h: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """ρ² = ⟨h|h⟩ — squared optimal SNR."""
    return inner_product(h, h, weight)


def matched_filter_snr_squared(
    d: jnp.ndarray,
    h: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """⟨d|h⟩² / ⟨h|h⟩ — squared matched-filter SNR."""
    dh = inner_product(d, h, weight)
    hh = inner_product(h, h, weight)
    return dh * dh / hh


# ── Log-likelihood (Gaussian approximation) ──────────────────────────────────

def log_likelihood_ifo(
    d_fd:   jnp.ndarray,
    h_fd:   jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Gaussian log-likelihood for a single interferometer.

    ℒ = -0.5 · ⟨d-h | d-h⟩

    Parameters
    ----------
    d_fd, h_fd : jnp.ndarray, shape (Nf,)
    weight : jnp.ndarray, shape (Nf,)
        Pre-computed inner-product weight.
    """
    residual = d_fd - h_fd
    return -0.5 * inner_product(residual, residual, weight)


def log_likelihood_network(
    d_fd_dict:   dict,
    h_fd_dict:   dict,
    weight_dict: dict,
) -> jnp.ndarray:
    """Joint log-likelihood over all detectors in a network.

    All input dicts must share the same set of detector keys.
    """
    logl_total = jnp.asarray(0.0)
    for ifo_name, d_fd in d_fd_dict.items():
        logl_total = logl_total + log_likelihood_ifo(
            d_fd, h_fd_dict[ifo_name], weight_dict[ifo_name],
        )
    return logl_total


# ── Factory: bind a likelihood to a Network's data/PSDs ──────────────────────

def make_likelihood_fn(network, mask_override: dict | None = None):
    """Build a jitted log-likelihood function bound to a Network's data.

    The returned function closes over the network's data **and the
    pre-computed inner-product weights**, so every call avoids the
    ``isfinite`` / ``1/psd`` work entirely.

    Parameters
    ----------
    network : Network
        A :class:`~gwgpu_jax.gwgpu_jax_network_ifos_utils.Network` with strain data
        and PSDs loaded for every detector.
    mask_override : dict[str, jnp.ndarray], optional
        Per-detector boolean masks. Falls back to ``ifo.grid.frequency_mask``.

    Returns
    -------
    logl_fn : callable
        ``logl_fn(h_fd_dict)`` → scalar log-likelihood. Keys of
        ``h_fd_dict`` must match the network's detector names.
    """
    mask_override = mask_override or {}
    d_fd_dict, weight_dict = {}, {}
    for ifo in network.interferometers:
        if ifo.grid is None or ifo.psd is None or ifo.strain_data_fd is None:
            raise RuntimeError(
                f"Interferometer '{ifo.name}' must have a grid, PSD, and "
                "strain data set before building a likelihood function."
            )
        d_fd_dict[ifo.name]   = ifo.strain_data_fd
        weight_dict[ifo.name] = precompute_weight(
            ifo.psd, ifo.grid.df,
            mask=mask_override.get(ifo.name, ifo.grid.frequency_mask),
        )

    @jax.jit
    def logl_fn(h_fd_dict):
        return log_likelihood_network(d_fd_dict, h_fd_dict, weight_dict)

    return logl_fn
