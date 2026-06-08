"""
gwgpu_jax_psd_utils.py
==================
Power Spectral Density (PSD) utilities for GWgpu_jax interferometers.

Analytical design-sensitivity models
--------------------------------------
  ``"aLIGO"``  — LIGO O4 design (Sathyaprakash & Schutz 2009, LRR, arXiv:0903.0338)
  ``"AdV"``    — Advanced Virgo design (adapted from aLIGO parametrisation)
  ``"KAGRA"``  — bKAGRA design (same structure, KAGRA-appropriate floor)
  ``"ET_D"``   — Einstein Telescope D (simplified 4-term model)
  ``"CE"``     — Cosmic Explorer (simplified 4-term model)
  ``"flat"``   — White (constant) PSD, useful for testing

File loading
------------
  ``psd_from_file(filepath, freqs)`` — read a two-column ASCII file
  (col 0 = frequency [Hz], col 1 = S_n(f) [Hz^{-1}]) and
  log-linearly interpolate onto ``freqs``.

Data-based estimation
----------------------
  ``psd_from_data(strain_td, sampling_rate, nperseg=None, noverlap=None,
  window="hann", target_freqs=None)`` — Welch periodogram estimate from a
  time-domain noise segment.  Pass ``target_freqs=grid.frequency_domain_array``
  to interpolate directly onto an existing :class:`TimeFrequencyGrid`.

Notes on analytical models
---------------------------
  * All analytical PSDs are rough approximations. For real parameter
    estimation load the PSD from data or from an official calibration file.
  * Values are set to ``jnp.inf`` below the lower frequency cut-off so that
    those bins are automatically down-weighted in matched-filter inner products.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp


# ── Analytical PSD models ─────────────────────────────────────────────────────

def psd_aLIGO(freqs: jnp.ndarray, f_min: float = 10.0) -> jnp.ndarray:
    """aLIGO O4 design PSD.

    Sathyaprakash & Schutz (2009), Living Rev. Rel., arXiv:0903.0338, Eq. 9.
    Floor ≈ 10^{-46} Hz^{-1} (strain ≈ 10^{-23} Hz^{-1/2}) at ~150 Hz.
    """
    f0, S0 = 215.0, 1e-46
    x = freqs / f0
    psd = S0 * ((4.49 * x) ** (-56) + 0.16 * x ** (-4.52) + 0.52 + 0.32 * x ** 2)
    return jnp.where(freqs >= f_min, psd, jnp.inf)


def psd_AdV(freqs: jnp.ndarray, f_min: float = 10.0) -> jnp.ndarray:
    """Advanced Virgo design PSD (simplified analytical approximation).

    Uses the same polynomial form as ``psd_aLIGO`` with parameters adjusted
    to match the AdV design bucket near 100 Hz.  Load from file for accurate
    analysis.
    """
    f0, S0 = 165.0, 1.2e-46
    x = freqs / f0
    psd = S0 * ((4.49 * x) ** (-56) + 0.16 * x ** (-4.52) + 0.52 + 0.32 * x ** 2)
    return jnp.where(freqs >= f_min, psd, jnp.inf)


def psd_KAGRA(freqs: jnp.ndarray, f_min: float = 10.0) -> jnp.ndarray:
    """bKAGRA design PSD (simplified analytical approximation).

    Comparable floor to aLIGO O4; see Aso et al. (2013).  Load from file for
    accurate analysis.
    """
    f0, S0 = 180.0, 3.0e-46
    x = freqs / f0
    psd = S0 * ((4.49 * x) ** (-56) + 0.16 * x ** (-4.52) + 0.52 + 0.32 * x ** 2)
    return jnp.where(freqs >= f_min, psd, jnp.inf)


def psd_ET_D(freqs: jnp.ndarray, f_min: float = 2.0) -> jnp.ndarray:
    """Einstein Telescope (ET-D) design PSD (simplified 4-term model).

    Rough approximation; floor ≈ 2e-50 Hz^{-1} (strain ≈ 5e-25 Hz^{-1/2})
    near 100 Hz, sensitive from ~2 Hz.  Load from the official ET-D curve
    for accurate analysis (Hild et al. 2011, CQG 28, 094013).
    """
    f0, S0 = 100.0, 2e-50
    x = freqs / f0
    psd = S0 * (x ** (-4) + x ** (-2) + 1.0 + x ** 2)
    return jnp.where(freqs >= f_min, psd, jnp.inf)


def psd_CE(freqs: jnp.ndarray, f_min: float = 5.0) -> jnp.ndarray:
    """Cosmic Explorer design PSD (simplified 4-term model).

    Rough approximation; floor ≈ 6e-52 Hz^{-1} (strain ≈ 2.5e-26 Hz^{-1/2})
    near 100 Hz.  Load from the official CE curve for accurate analysis.
    """
    f0, S0 = 100.0, 6e-52
    x = freqs / f0
    psd = S0 * (x ** (-4) + x ** (-2) + 1.0 + x ** 2)
    return jnp.where(freqs >= f_min, psd, jnp.inf)


def psd_flat(freqs: jnp.ndarray, S0: float = 1e-46, f_min: float = 0.0) -> jnp.ndarray:
    """Frequency-independent (white) PSD — useful for testing."""
    return jnp.where(freqs >= f_min, jnp.full_like(freqs, S0), jnp.inf)


# ── File loading ──────────────────────────────────────────────────────────────

def psd_from_file(filepath: str | Path, freqs: jnp.ndarray) -> jnp.ndarray:
    """Load a PSD from a two-column ASCII file and interpolate onto ``freqs``.

    Parameters
    ----------
    filepath : str or Path
        Path to a whitespace- or comma-delimited file with two columns:
        ``frequency [Hz]`` and ``S_n(f) [Hz^{-1}]``.  Lines starting with
        ``#`` are treated as comments.
    freqs : jnp.ndarray
        Target frequency array.  Output is set to ``jnp.inf`` for frequencies
        outside the file's range.

    Returns
    -------
    psd : jnp.ndarray, shape ``(len(freqs),)``
        Log-linearly interpolated PSD values.
    """
    data = np.loadtxt(filepath, comments="#")
    f_file = data[:, 0]
    s_file = data[:, 1]

    # Log-linear interpolation (linear in log-log space)
    log_psd = np.interp(
        np.log(np.array(freqs)),
        np.log(f_file),
        np.log(s_file),
        left=np.inf,
        right=np.inf,
    )
    return jnp.array(np.exp(log_psd))


# ── Data-based estimation (Welch) ─────────────────────────────────────────────

def psd_from_data(
    strain_td:     np.ndarray,
    sampling_rate: float,
    nperseg:       int | None = None,
    noverlap:      int | None = None,
    window:        str = "hann",
    target_freqs:  jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate the one-sided PSD from a time-domain strain via Welch's method.

    The strain is divided into *overlapping* windowed segments; periodograms are
    computed for each segment and averaged.  The result is a consistent estimate
    of the one-sided power spectral density S_n(f) [Hz^{-1}].

    Parameters
    ----------
    strain_td : array-like, shape ``(N,)``
        Time-domain noise strain.
    sampling_rate : float
        Sampling rate [Hz].
    nperseg : int, optional
        Number of samples per segment.  Determines the frequency resolution:
        ``df = sampling_rate / nperseg``.  Defaults to ``N // 8``.
        Larger values give finer frequency resolution at the cost of fewer
        averaged segments (higher variance).
    noverlap : int, optional
        Number of samples to overlap between consecutive segments.  Defaults
        to ``nperseg // 2`` (50 % overlap), which is standard for a Hann window.
        More overlap increases spectral smoothness but does not add independent
        information.
    window : str, optional
        Window function name accepted by ``scipy.signal.welch``.  Default
        ``"hann"``.  Other useful choices: ``"blackman"``, ``"flattop"``,
        ``"boxcar"`` (no windowing).
    target_freqs : array-like, optional
        If provided, the Welch estimate is **log-linearly interpolated** onto
        these frequencies (e.g. ``grid.frequency_domain_array``).  Frequencies
        outside the Welch grid — including the DC bin (f = 0) — are set to
        ``jnp.inf``.  Useful for directly obtaining a PSD on the grid of an
        :class:`~gwgpu_jax.gwgpu_jax_interferometer_utils.Interferometer`.

    Returns
    -------
    freqs : jnp.ndarray
        Frequency array [Hz].  Equal to ``target_freqs`` if supplied, otherwise
        the native Welch frequency axis (``nperseg // 2 + 1`` points from 0 to
        ``sampling_rate / 2``).
    psd : jnp.ndarray
        One-sided PSD estimate [Hz^{-1}].

    Examples
    --------
    Estimate PSD from 64 s of data and project onto a TimeFrequencyGrid:

    >>> from gwgpu_jax.gwgpu_jax_timefrequencydomain_utils import TimeFrequencyGrid
    >>> from gwgpu_jax.gwgpu_jax_psd_utils import psd_from_data
    >>> import numpy as np
    >>> grid = TimeFrequencyGrid(8.0, 2048.0)
    >>> rng  = np.random.default_rng(0)
    >>> noise = rng.standard_normal(64 * 2048) * 1e-21
    >>> freqs, psd = psd_from_data(noise, 2048.0, target_freqs=grid.frequency_domain_array)
    >>> psd.shape   # same as grid.frequency_domain_array.shape
    (8193,)
    """
    from scipy.signal import welch

    strain = np.asarray(strain_td, dtype=np.float64)
    n = len(strain)
    if nperseg is None:
        nperseg = max(n // 8, 64)
    if noverlap is None:
        noverlap = nperseg // 2

    f, pxx = welch(
        strain,
        fs=float(sampling_rate),
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        scaling="density",
    )

    if target_freqs is not None:
        target = np.asarray(target_freqs, dtype=np.float64)
        # Use only positive-frequency bins for log interpolation (exclude DC)
        mask = f > 0.0
        f_pos   = f[mask]
        pxx_pos = pxx[mask]

        log_psd_interp = np.interp(
            np.log(np.where(target > 0.0, target, np.nan)),
            np.log(f_pos),
            np.log(pxx_pos),
            left=np.inf,
            right=np.inf,
        )
        # DC bin (target == 0) and out-of-range bins → inf
        psd_out = np.where(target > 0.0, np.exp(log_psd_interp), np.inf)
        # Do NOT force float32 — real-detector PSDs are ~1e-46 Hz^-1
        # which underflows fp32's denormal floor. Let JAX honour the
        # active precision (defaults float32; float64 with jax_enable_x64).
        return jnp.asarray(target), jnp.asarray(psd_out)

    return jnp.asarray(f), jnp.asarray(pxx)


# ── Registry and dispatcher ───────────────────────────────────────────────────

_PSD_REGISTRY: dict[str, callable] = {
    "aLIGO": psd_aLIGO,
    "AdV":   psd_AdV,
    "KAGRA": psd_KAGRA,
    "ET_D":  psd_ET_D,
    "CE":    psd_CE,
    "flat":  psd_flat,
}


def get_psd(
    model:  str | jnp.ndarray,
    freqs:  jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    """Evaluate a PSD model on ``freqs``.

    Parameters
    ----------
    model : str or array
        * ``str`` name from ``_PSD_REGISTRY`` (e.g. ``"aLIGO"``).
        * ``str`` path to a two-column ASCII file.
        * JAX / NumPy array of shape ``(len(freqs),)`` — used as-is.
    freqs : jnp.ndarray
        Frequency array on which to evaluate the PSD.
    **kwargs
        Extra keyword arguments forwarded to the analytical function
        (e.g. ``f_min``).

    Returns
    -------
    psd : jnp.ndarray, shape ``(len(freqs),)``
    """
    if isinstance(model, str):
        if model in _PSD_REGISTRY:
            return _PSD_REGISTRY[model](freqs, **kwargs)
        # Treat as file path
        return psd_from_file(model, freqs)
    # Assume array-like
    arr = jnp.asarray(model)
    if arr.shape != freqs.shape:
        raise ValueError(
            f"Custom PSD array shape {arr.shape} does not match "
            f"freqs shape {freqs.shape}."
        )
    return arr
