"""
gwgpu_jax_relative_binning.py
==========================
Relative-binning (Zackay, Dai & Venumadhav 2018, arXiv:1806.08792) log-likelihood
for GWgpu_jax, with optional analytical marginalisation over the coalescence time
``tc`` and the coalescence phase ``phi_c``.

This is the GWgpu_jax-API counterpart of the SHARPy-era ``relative_binning.py``: it
keeps the same algorithm but builds everything against
:class:`gwgpu_jax.Network` / :class:`gwgpu_jax.Interferometer` and the GWgpu_jax
inner-product convention, so it interoperates with the rest of the toolkit
(``CustomWaveform``, ``GWgpu_jaxNestedSampler``, ``gwgpu_jax.compat.attach_event_to_network``,
``gwgpu_jax.make_likelihood_fn``, …) without any SHARPy dependency.

Inner-product convention
------------------------
We use the canonical noise-weighted inner product

    ⟨a|b⟩ = Σ_k Re(conj(a_k)·b_k) · w_k,        w_k = 4·df·mask_k / S_n(f_k)

(implemented in :mod:`gwgpu_jax.gwgpu_jax_likelihood_utils`).  The Gaussian
log-likelihood is then

    log L = −½ · ⟨d − h | d − h⟩
          = −½ · (dd − 2·Re⟨d|h⟩ + ⟨h|h⟩).

Per-bin RB summary quantities (one set per detector)::

    A0_j = Σ_{k∈bin_j} conj(d_k)·h0_k · w_k       (complex, NO conjugation of h0)
    B0_j = Σ_{k∈bin_j} |h0_k|² · w_k              (real, positive)
    dd   = Σ_k |d_k|² · w_k                       (real, constant)

With the ratio approximation  h(f) ≈ r_j · h0(f)  for f ∈ bin_j::

    log L_RB ≈ −½ · (dd − 2·Re[Σ_j r_j·A0_j] + Σ_j |r_j|²·B0_j)

Waveform-function contract
--------------------------
``waveform_fn`` is the same callable used by :class:`gwgpu_jax.GWgpu_jaxNestedSampler`::

    hp, hc = waveform_fn(params, freqs)        # params : dict of scalars

Sky angles ``ra, dec, psi`` and the coalescence time ``tc`` are NOT applied by
``waveform_fn`` — the projection layer here multiplies by
``(Fp·hp + Fc·hc) · exp(−2πi·f·(tc + Δt_geo→ifo))`` using
:meth:`gwgpu_jax.Interferometer.antenna_pattern` and
:meth:`gwgpu_jax.Interferometer.time_delay_from_geocenter`.

The coalescence phase ``phi_c`` must be applied **as a uniform phase rotation
on (hp, hc)** inside ``waveform_fn``, e.g.

    hp, hc = predict(...);  phase = jnp.exp(-1j * params["phi_c"])
    return hp * phase, hc * phase

This is required for the analytical φ_c marginalisation (Bessel identity) to
be correct.  See ``examples/gwgpu_jax_GW170817_mlgw_bns_jax_rb_analyt_time_phase_marg_pe.ipynb``
for a concrete mlgw_bns_jax wrapper.

Analytical marginalisations
---------------------------
- :func:`build_rb_likelihood`                — bare RB log-likelihood.
- :func:`build_rb_likelihood_time_marg`      — analytical marginalisation
  over ``tc`` via a discrete grid + ``logsumexp``.
- :func:`build_rb_likelihood_tc_phi_marg`    — joint analytical marginalisation
  over ``tc`` *and* ``phi_c`` via the Bessel-function identity
  ∫ exp(Re[z·e^{-iφ}]) dφ/(2π) = I₀(|z|),
  using the numerically stable form  log I₀(x) = log(i0e(x)) + x.
- :func:`build_full_likelihood_time_marg`,
  :func:`build_full_likelihood_tc_phi_marg`  — same marginalisations on the
  full FD grid (no RB approximation) — useful for validating accuracy.

All returned callables take a **particle dict** with the same contract as
``GWgpu_jaxNestedSampler.loglikelihood_fn`` (sampled keys ∪ fixed_params keys), and
return a scalar JAX array.  They are JIT/vmap-safe.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from gwgpu_jax.gwgpu_jax_likelihood_utils import precompute_weight, waveform_projection_fd
from gwgpu_jax.gwgpu_jax_network_ifos_utils import Network


# ============================================================================
# Data structures
# ============================================================================

class RBSummaryDet(NamedTuple):
    """Precomputed relative-binning summary data for one detector."""
    A0:      jnp.ndarray   # (n_bins,) complex   — Σ_{k∈j} conj(d)·h0 · w
    B0:      jnp.ndarray   # (n_bins,) real      — Σ_{k∈j} |h0|² · w
    dd:      jnp.ndarray   # scalar real         — Σ_k |d|² · w
    h0_bins: jnp.ndarray   # (n_bins,) complex   — reference projected waveform


class RBNetwork(NamedTuple):
    """Per-detector RB summary data + bin geometry, indexed by detector name."""
    summaries:        dict             # {ifo_name: RBSummaryDet}
    f_bins:           jnp.ndarray      # (n_bins,)  bin-centre frequencies [Hz]
    bin_edges:        np.ndarray       # (n_bins+1,) bin boundaries [Hz]
    ifo_names:        tuple            # ordered detector names
    fiducial_params:  dict             # frozen copy of θ_0 (NumPy/Python scalars)


# ============================================================================
# Bin-edge selection (pure NumPy)
# ============================================================================

def choose_bins_pn(
    f:                np.ndarray,
    n_bins:           int   = 400,
    max_bin_width_hz: float = 8.0,
) -> np.ndarray:
    """Choose frequency bin edges from a PN-proxy phase + a max-width clamp.

    Edges are placed so that

      1. The PN-proxy cumulative phase  C(f) = f_low^{-5/3} − f^{-5/3}  (∝
         leading-order chirp phase) is split into ``n_bins`` equal increments.
         This is independent of any waveform / projection — in particular it
         does not see the large ``exp(-2πi·f·timeshift)`` factor that contaminates
         the projected h0 phase near f_low for BNS signals.

      2. No bin is wider than ``max_bin_width_hz``.  Default 8 Hz keeps the
         linear-phase error from coalescence-time drift below ≈ 0.05 rad for
         |δtc| ≤ 1 ms across the high-frequency band.

    Both constraints are applied and the union of edges is kept (so the final
    bin count can exceed ``n_bins``).

    Parameters
    ----------
    f : np.ndarray, shape (N,)
        Monotonically increasing frequency array (within the analysis band).
    n_bins : int
        Target number of PN-proxy bins.
    max_bin_width_hz : float
        Maximum allowed bin width [Hz].

    Returns
    -------
    bin_edges : np.ndarray, shape (≥ n_bins+1,)
        Frequencies of bin boundaries, including ``f[0]`` and ``f[-1]``.
    """
    f = np.asarray(f, dtype=np.float64)
    if f[0] <= 0.0:
        raise ValueError("choose_bins_pn requires f[0] > 0 (in-band frequencies only).")

    cum_phase   = f[0] ** (-5.0 / 3.0) - f ** (-5.0 / 3.0)
    total_phase = cum_phase[-1]
    if total_phase < 1e-10:
        edges_phase = np.linspace(f[0], f[-1], n_bins + 1)
    else:
        target_phases = np.linspace(0.0, total_phase, n_bins + 1)
        edges_phase   = np.interp(target_phases, cum_phase, f)
        edges_phase[0]  = f[0]
        edges_phase[-1] = f[-1]

    n_freq_bins = max(n_bins, int(np.ceil((f[-1] - f[0]) / max_bin_width_hz)) + 1)
    edges_freq  = np.linspace(f[0], f[-1], n_freq_bins + 1)

    edges = np.unique(np.concatenate([edges_phase, edges_freq]))
    edges = edges[(edges >= f[0]) & (edges <= f[-1])]
    edges[0], edges[-1] = f[0], f[-1]
    return edges


def bin_centres(bin_edges: np.ndarray) -> np.ndarray:
    """Return the centre frequency of each bin."""
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


# ============================================================================
# Projection helpers (single-frequency-array, JAX-traceable)
# ============================================================================

def _project_at_freqs(
    waveform_fn: Callable,
    params:      dict,
    freqs:       jnp.ndarray,
    ifo,
    gmst:        float = 0.0,
) -> jnp.ndarray:
    """Return the projected detector strain ``h(f)`` at the given frequencies.

    Matches the projection used inside :class:`gwgpu_jax.GWgpu_jaxNestedSampler`::

        h_proj = (Fp·hp + Fc·hc) · exp(-2πi·f·(tc + Δt_geo→ifo))

    Sky angles ``ra, dec, psi`` and time-of-coalescence ``tc`` are read from
    ``params``.  ``waveform_fn(params, freqs)`` returns ``(hp, hc)``.
    """
    hp, hc = waveform_fn(params, freqs)
    Fp, Fc = ifo.antenna_pattern(params["ra"], params["dec"], params["psi"], gmst)
    dt_ifo = ifo.time_delay_from_geocenter(params["ra"], params["dec"], gmst)
    return waveform_projection_fd(hp, hc, Fp, Fc, freqs, params["tc"] + dt_ifo)


# ============================================================================
# Precompute RB summary (runs once on full grid before the sampler starts)
# ============================================================================

def precompute_rb_network(
    network:           Network,
    waveform_fn:       Callable,
    fiducial_params:   dict,
    n_bins:            int   = 400,
    max_bin_width_hz:  float = 8.0,
    gmst:              float = 0.0,
    verbose:           bool  = True,
) -> RBNetwork:
    """Precompute the per-detector RB summary data and bin geometry.

    Parameters
    ----------
    network : gwgpu_jax.Network
        Populated with strain data and PSDs on each interferometer
        (see :func:`gwgpu_jax.compat.attach_event_to_network`).
    waveform_fn : callable
        ``waveform_fn(params_dict, freqs) -> (hp, hc)`` — same contract as
        :class:`gwgpu_jax.GWgpu_jaxNestedSampler`.  Must apply ``phi_c`` as a uniform
        phase rotation on (hp, hc) and not include any sky/time projection.
    fiducial_params : dict[str, float]
        Reference parameters θ_0 used to compute h0.  Should be close to the
        true signal; an approximate MAP is fine.  Must include everything
        ``waveform_fn`` and the projection layer consume:
        ``ra, dec, psi, tc, phi_c`` plus the waveform's intrinsic parameters.
    n_bins : int
        Target PN-proxy bin count.
    max_bin_width_hz : float
        Maximum bin width [Hz].
    gmst : float
        Greenwich Mean Sidereal Time [rad] used in the projection layer.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    rb_network : RBNetwork
    """
    ifos = list(network.interferometers)
    if not ifos:
        raise ValueError("network has no interferometers.")
    grid = ifos[0].grid
    if grid is None:
        raise RuntimeError("Network IFOs are missing a TimeFrequencyGrid.")

    freqs_full = grid.frequency_domain_array
    df         = grid.df
    fmask      = grid.frequency_mask

    # Restrict bin selection to the in-band frequencies (mask=True).
    in_band     = np.asarray(fmask)
    f_full_np   = np.asarray(freqs_full)
    f_band_np   = f_full_np[in_band]
    if f_band_np.size == 0:
        raise RuntimeError("Empty analysis band — check grid.f_min / f_max.")

    # ── Pure-NumPy fiducial-params dict (cast traced/JAX scalars to Python) ──
    fid_np = {k: float(v) for k, v in fiducial_params.items()}
    fid_jax = {k: jnp.asarray(v) for k, v in fid_np.items()}

    if verbose:
        print(f"[RB] in-band points: {f_band_np.size}  (df={df:.4f} Hz)")
        print(f"[RB] choosing ~{n_bins} PN-proxy bins (max width {max_bin_width_hz:g} Hz) …")

    # ── Choose bins from the in-band grid ────────────────────────────────────
    bin_edges = choose_bins_pn(f_band_np, n_bins=n_bins,
                               max_bin_width_hz=max_bin_width_hz)
    f_bins_np = bin_centres(bin_edges)
    f_bins    = jnp.asarray(f_bins_np, dtype=jnp.float64)

    # ── Assign each in-band frequency to a bin ───────────────────────────────
    bin_idx_band = np.searchsorted(bin_edges, f_band_np, side="right") - 1
    bin_idx_band = np.clip(bin_idx_band, 0, len(f_bins_np) - 1)

    # ── Precompute h0 once per IFO on the full FD grid, then at bin centres ──
    summaries: dict = {}
    for ifo in ifos:
        if ifo.strain_data_fd is None or ifo.psd is None:
            raise RuntimeError(
                f"Interferometer '{ifo.name}' is missing strain_data_fd or PSD."
            )
        # Per-IFO weight w_k = 4·df·mask/PSD (the GWgpu_jax inner-product weight).
        w_full = precompute_weight(ifo.psd, df, mask=fmask)
        w_band = np.asarray(w_full)[in_band]
        d_band = np.asarray(ifo.strain_data_fd)[in_band]

        # Full-grid h0 and bin-centre h0 — both as NumPy after evaluation so
        # the summary precompute is plain NumPy (avoids JAX tracing in a loop).
        h0_full = np.asarray(
            _project_at_freqs(waveform_fn, fid_jax, freqs_full, ifo, gmst=gmst)
        )[in_band]
        h0_bins = np.asarray(
            _project_at_freqs(waveform_fn, fid_jax, f_bins, ifo, gmst=gmst)
        )

        # Per-bin A0, B0
        n_bins_actual = len(f_bins_np)
        A0 = np.zeros(n_bins_actual, dtype=np.complex128)
        B0 = np.zeros(n_bins_actual, dtype=np.float64)
        # Bin-wise reductions via np.add.at (handles arbitrary index map)
        np.add.at(A0, bin_idx_band, np.conj(d_band) * h0_full * w_band)
        np.add.at(B0, bin_idx_band, (np.abs(h0_full) ** 2) * w_band)

        dd = float(np.sum((np.abs(d_band) ** 2) * w_band))

        if verbose:
            print(
                f"  {ifo.name}: dd={dd:.3e}  "
                f"|A0|max={float(np.max(np.abs(A0))):.3e}  "
                f"ΣB0={float(np.sum(B0)):.3e}"
            )

        summaries[ifo.name] = RBSummaryDet(
            A0      = jnp.asarray(A0, dtype=jnp.complex128),
            B0      = jnp.asarray(B0, dtype=jnp.float64),
            dd      = jnp.asarray(dd, dtype=jnp.float64),
            h0_bins = jnp.asarray(h0_bins, dtype=jnp.complex128),
        )

    if verbose:
        print(f"[RB] ready: {f_band_np.size} pts → {len(f_bins_np)} bins "
              f"({f_band_np.size // max(1, len(f_bins_np))}× reduction)")

    return RBNetwork(
        summaries       = summaries,
        f_bins          = f_bins,
        bin_edges       = np.asarray(bin_edges, dtype=np.float64),
        ifo_names       = tuple(ifo.name for ifo in ifos),
        fiducial_params = fid_np,
    )


# ============================================================================
# Bare RB log-likelihood
# ============================================================================

def build_rb_likelihood(
    network:           Network,
    waveform_fn:       Callable,
    fiducial_params:   dict,
    n_bins:            int   = 400,
    max_bin_width_hz:  float = 8.0,
    gmst:              float = 0.0,
    fixed_params:      dict | None = None,
    verbose:           bool  = True,
) -> tuple[Callable, RBNetwork]:
    """Build a particle-dict → scalar relative-binning log-likelihood.

    Parameters
    ----------
    network, waveform_fn, fiducial_params, n_bins, max_bin_width_hz, gmst, verbose
        See :func:`precompute_rb_network`.
    fixed_params : dict, optional
        Parameters that the sampler does not move; merged into the particle
        dict before each call (same contract as ``GWgpu_jaxNestedSampler``).

    Returns
    -------
    log_L_rb : callable
        ``log_L_rb(particle_dict) -> scalar``.  JIT-safe.
    rb_network : RBNetwork
    """
    rb = precompute_rb_network(
        network, waveform_fn, fiducial_params,
        n_bins=n_bins, max_bin_width_hz=max_bin_width_hz,
        gmst=gmst, verbose=verbose,
    )

    fixed   = dict(fixed_params or {})
    ifos    = list(network.interferometers)
    f_bins  = rb.f_bins
    sums    = rb.summaries
    _gmst   = float(gmst)

    def log_L_rb(particle: dict) -> jnp.ndarray:
        params = {**fixed, **particle}
        logl   = jnp.asarray(0.0)
        for ifo in ifos:
            h_bins = _project_at_freqs(waveform_fn, params, f_bins, ifo, gmst=_gmst)
            s      = sums[ifo.name]
            r      = jnp.where(jnp.abs(s.h0_bins) > 0.0,
                               h_bins / s.h0_bins,
                               jnp.zeros_like(h_bins))
            cross  = jnp.real(jnp.sum(r * s.A0))
            self_t = jnp.sum((jnp.abs(r) ** 2) * s.B0)
            logl   = logl + (-0.5) * (s.dd - 2.0 * cross + self_t)
        return logl

    return log_L_rb, rb


# ============================================================================
# Analytical tc marginalisation (RB)
# ============================================================================

def build_rb_likelihood_time_marg(
    network:           Network,
    waveform_fn:       Callable,
    fiducial_params:   dict,
    tc_grid:           np.ndarray,
    n_bins:            int   = 400,
    max_bin_width_hz:  float = 8.0,
    gmst:              float = 0.0,
    fixed_params:      dict | None = None,
    verbose:           bool  = True,
) -> tuple[Callable, RBNetwork]:
    """Build an RB log-likelihood analytically marginalised over ``tc``.

    The particle dict passed to the returned callable **must not contain
    ``tc``**; the fiducial ``tc_0 = fiducial_params['tc']`` is used as the
    reference, and the marginalisation runs over the user-supplied ``tc_grid``::

        log L(tc_k) = −½·(dd − 2·Re[(phase_matrix @ (r·A0))_k] + ⟨|r|², B0⟩)
        log L_marg = logsumexp_k log L(tc_k) − log N_tc

    The cross-term shift uses the identity that the projected strain at a
    different tc differs from the reference by exactly ``exp(−2πi·f·Δtc)``,
    so the per-tc cross term reduces to a single ``(N_tc × n_bins) @ (n_bins,)``
    matmul.

    Parameters
    ----------
    network, waveform_fn, fiducial_params, n_bins, max_bin_width_hz, gmst,
    fixed_params, verbose
        See :func:`build_rb_likelihood`.
    tc_grid : np.ndarray, shape (N_tc,)
        Absolute ``tc`` values to marginalise over (same units/zero point as
        the ``tc`` in ``fiducial_params``).

    Returns
    -------
    log_L_marg : callable
        ``log_L_marg(particle_dict_without_tc) -> scalar``.
    rb_network : RBNetwork
    """
    rb = precompute_rb_network(
        network, waveform_fn, fiducial_params,
        n_bins=n_bins, max_bin_width_hz=max_bin_width_hz,
        gmst=gmst, verbose=verbose,
    )

    fixed   = dict(fixed_params or {})
    ifos    = list(network.interferometers)
    f_bins  = rb.f_bins
    sums    = rb.summaries
    _gmst   = float(gmst)

    tc_0    = float(fiducial_params["tc"])
    tc_arr  = jnp.asarray(tc_grid, dtype=jnp.float64)
    N_tc    = int(tc_arr.shape[0])
    tc_delta = tc_arr - tc_0                                          # (N_tc,)
    phase_mat = jnp.exp(-1j * 2.0 * jnp.pi
                        * tc_delta[:, None] * f_bins[None, :])         # (N_tc, n_bins)
    log_N_tc = jnp.log(jnp.asarray(N_tc, dtype=jnp.float64))

    def log_L_marg(particle: dict) -> jnp.ndarray:
        params = {**fixed, **particle, "tc": tc_0}
        log_L_tc = jnp.zeros(N_tc, dtype=jnp.float64)
        for ifo in ifos:
            h_bins = _project_at_freqs(waveform_fn, params, f_bins, ifo, gmst=_gmst)
            s      = sums[ifo.name]
            r      = jnp.where(jnp.abs(s.h0_bins) > 0.0,
                               h_bins / s.h0_bins,
                               jnp.zeros_like(h_bins))
            self_t = jnp.sum((jnp.abs(r) ** 2) * s.B0)
            cross_tc = jnp.real(phase_mat @ (r * s.A0))               # (N_tc,)
            log_L_tc = log_L_tc + (-0.5) * (s.dd - 2.0 * cross_tc + self_t)
        return jax.scipy.special.logsumexp(log_L_tc) - log_N_tc

    return log_L_marg, rb


# ============================================================================
# Joint tc + phi_c marginalisation (RB)
# ============================================================================

def build_rb_likelihood_tc_phi_marg(
    network:           Network,
    waveform_fn:       Callable,
    fiducial_params:   dict,
    tc_grid:           np.ndarray,
    n_bins:            int   = 400,
    max_bin_width_hz:  float = 8.0,
    gmst:              float = 0.0,
    fixed_params:      dict | None = None,
    verbose:           bool  = True,
) -> tuple[Callable, RBNetwork]:
    """Build an RB log-likelihood jointly marginalised over ``tc`` and ``phi_c``.

    The returned callable expects a particle dict **without ``tc`` or
    ``phi_c``**.  Internally we evaluate the template at ``tc=tc_0`` and
    ``phi_c=0``, then

      * sum over a discrete ``tc`` grid (same as the tc-only variant);
      * marginalise over ``phi_c`` analytically using the Bessel identity

            ∫₀^{2π} exp(Re[z · e^{-iφ}]) dφ / (2π) = I₀(|z|),

        evaluated in the numerically stable form
        log I₀(x) = log(i0e(x)) + x.

    Requires ``fiducial_params['phi_c'] == 0`` so the reference ``h0`` (and the
    summary ``A0``) are phi_c-free.

    Parameters
    ----------
    network, waveform_fn, fiducial_params, n_bins, max_bin_width_hz, gmst,
    fixed_params, verbose, tc_grid
        See :func:`build_rb_likelihood_time_marg`.

    Returns
    -------
    log_L_marg : callable
        ``log_L_marg(particle_dict_without_tc_or_phi_c) -> scalar``.
    rb_network : RBNetwork
    """
    if "phi_c" not in fiducial_params:
        raise KeyError("fiducial_params must include 'phi_c' (set it to 0.0).")
    if abs(float(fiducial_params["phi_c"])) > 1e-12:
        raise ValueError(
            "build_rb_likelihood_tc_phi_marg requires fiducial_params['phi_c'] == 0; "
            "the reference h0 (and A0) must be phi_c-free for the Bessel identity "
            f"to apply.  Got phi_c = {fiducial_params['phi_c']!r}."
        )

    rb = precompute_rb_network(
        network, waveform_fn, fiducial_params,
        n_bins=n_bins, max_bin_width_hz=max_bin_width_hz,
        gmst=gmst, verbose=verbose,
    )

    fixed   = dict(fixed_params or {})
    ifos    = list(network.interferometers)
    f_bins  = rb.f_bins
    sums    = rb.summaries
    _gmst   = float(gmst)

    tc_0      = float(fiducial_params["tc"])
    tc_arr    = jnp.asarray(tc_grid, dtype=jnp.float64)
    N_tc      = int(tc_arr.shape[0])
    tc_delta  = tc_arr - tc_0
    phase_mat = jnp.exp(-1j * 2.0 * jnp.pi
                        * tc_delta[:, None] * f_bins[None, :])      # (N_tc, n_bins)
    log_N_tc  = jnp.log(jnp.asarray(N_tc, dtype=jnp.float64))

    def log_L_marg(particle: dict) -> jnp.ndarray:
        params = {**fixed, **particle, "tc": tc_0, "phi_c": 0.0}
        # tc-independent constant: −½·(dd + hh)
        const = jnp.asarray(0.0, dtype=jnp.float64)
        # Accumulate W(tc_k) = Σ_det Σ_j r_j·A0_j · exp(−2πi·f_j·Δtc_k)
        W_tc  = jnp.zeros(N_tc, dtype=jnp.complex128)

        for ifo in ifos:
            h_bins = _project_at_freqs(waveform_fn, params, f_bins, ifo, gmst=_gmst)
            s      = sums[ifo.name]
            r      = jnp.where(jnp.abs(s.h0_bins) > 0.0,
                               h_bins / s.h0_bins,
                               jnp.zeros_like(h_bins))
            self_t = jnp.sum((jnp.abs(r) ** 2) * s.B0)                # tc-independent
            const  = const + (-0.5) * (s.dd + self_t)
            W_tc   = W_tc + phase_mat @ (r * s.A0)                    # (N_tc,)

        x = jnp.abs(W_tc)
        log_bessel = jnp.log(jax.scipy.special.i0e(x)) + x            # log I₀(|W|)
        log_L_phi_tc = const + log_bessel                             # (N_tc,)
        return jax.scipy.special.logsumexp(log_L_phi_tc) - log_N_tc

    return log_L_marg, rb


# ============================================================================
# Full-grid validation variants (no RB approximation)
# ============================================================================

def _band_data(network: Network) -> tuple[list, jnp.ndarray, jnp.ndarray, list, list]:
    """Pull in-band (freqs, weights, data) per IFO from the network.

    Returns
    -------
    ifos       : list[Interferometer]
    freqs_band : (N_band,) frequencies in band
    fmask      : (N_full,) bool mask
    d_band_list, w_band_list : lists of (N_band,) JAX arrays
    """
    ifos   = list(network.interferometers)
    grid   = ifos[0].grid
    fmask  = grid.frequency_mask
    freqs  = grid.frequency_domain_array
    freqs_band = freqs[fmask]
    d_band_list, w_band_list = [], []
    for ifo in ifos:
        w_full = precompute_weight(ifo.psd, grid.df, mask=fmask)
        d_band_list.append(ifo.strain_data_fd[fmask])
        w_band_list.append(w_full[fmask])
    return ifos, freqs_band, fmask, d_band_list, w_band_list


def build_full_likelihood_time_marg(
    network:         Network,
    waveform_fn:     Callable,
    fiducial_params: dict,
    tc_grid:         np.ndarray,
    gmst:            float = 0.0,
    fixed_params:    dict | None = None,
) -> Callable:
    """Full-grid time-marginalised log-likelihood (no RB approximation).

    Uses the same particle-dict-without-``tc`` contract as
    :func:`build_rb_likelihood_time_marg`.  Heavier per-call than RB but with
    no approximation — useful for validating the RB output at the fiducial
    point and a handful of perturbed points.
    """
    ifos, freqs_band, fmask, d_list, w_list = _band_data(network)
    fixed   = dict(fixed_params or {})
    tc_0    = float(fiducial_params["tc"])
    tc_arr  = jnp.asarray(tc_grid, dtype=jnp.float64)
    N_tc    = int(tc_arr.shape[0])
    tc_delta = tc_arr - tc_0
    phase_mat_full = jnp.exp(
        -1j * 2.0 * jnp.pi * tc_delta[:, None] * freqs_band[None, :]
    )                                                                # (N_tc, N_band)
    log_N_tc = jnp.log(jnp.asarray(N_tc, dtype=jnp.float64))
    _gmst    = float(gmst)

    def log_L_full_tc_marg(particle: dict) -> jnp.ndarray:
        params  = {**fixed, **particle, "tc": tc_0}
        log_L_tc = jnp.zeros(N_tc, dtype=jnp.float64)
        for ifo, d_b, w_b in zip(ifos, d_list, w_list):
            h_b   = _project_at_freqs(waveform_fn, params, freqs_band, ifo, gmst=_gmst)
            dd_i  = jnp.sum((jnp.abs(d_b) ** 2) * w_b)
            hh_i  = jnp.sum((jnp.abs(h_b) ** 2) * w_b)
            # h_b is projected at tc_0 (its phase already carries exp(-2πi f tc_0));
            # phase_mat_full applies the *relative* shift exp(-2πi f Δtc), Δtc=tc_k-tc_0,
            # so the net template phase is exp(-2πi f tc_k) — same convention as the RB
            # path (A0 keeps the tc_0 phase). Do NOT strip tc_0 here.
            integrand    = jnp.conj(d_b) * h_b * w_b
            cross_tc     = jnp.real(phase_mat_full @ integrand)      # (N_tc,)
            log_L_tc     = log_L_tc + (-0.5) * (dd_i - 2.0 * cross_tc + hh_i)
        return jax.scipy.special.logsumexp(log_L_tc) - log_N_tc

    return log_L_full_tc_marg


def build_full_likelihood_tc_phi_marg(
    network:         Network,
    waveform_fn:     Callable,
    fiducial_params: dict,
    tc_grid:         np.ndarray,
    gmst:            float = 0.0,
    fixed_params:    dict | None = None,
) -> Callable:
    """Full-grid joint tc+φ_c marginalised log-likelihood (no RB approximation).

    Same particle-dict-without-``tc``-or-``phi_c`` contract as
    :func:`build_rb_likelihood_tc_phi_marg`.  Provides a slow but exact
    reference for validating the RB variant.
    """
    if "phi_c" not in fiducial_params:
        raise KeyError("fiducial_params must include 'phi_c' (set it to 0.0).")
    if abs(float(fiducial_params["phi_c"])) > 1e-12:
        raise ValueError(
            "build_full_likelihood_tc_phi_marg requires fiducial_params['phi_c'] == 0."
        )

    ifos, freqs_band, fmask, d_list, w_list = _band_data(network)
    fixed   = dict(fixed_params or {})
    tc_0    = float(fiducial_params["tc"])
    tc_arr  = jnp.asarray(tc_grid, dtype=jnp.float64)
    N_tc    = int(tc_arr.shape[0])
    tc_delta = tc_arr - tc_0
    phase_mat_full = jnp.exp(
        -1j * 2.0 * jnp.pi * tc_delta[:, None] * freqs_band[None, :]
    )
    log_N_tc = jnp.log(jnp.asarray(N_tc, dtype=jnp.float64))
    _gmst    = float(gmst)

    def log_L_full_tp_marg(particle: dict) -> jnp.ndarray:
        params = {**fixed, **particle, "tc": tc_0, "phi_c": 0.0}
        const  = jnp.asarray(0.0, dtype=jnp.float64)
        W_tc   = jnp.zeros(N_tc, dtype=jnp.complex128)
        for ifo, d_b, w_b in zip(ifos, d_list, w_list):
            h_b   = _project_at_freqs(waveform_fn, params, freqs_band, ifo, gmst=_gmst)
            dd_i  = jnp.sum((jnp.abs(d_b) ** 2) * w_b)
            hh_i  = jnp.sum((jnp.abs(h_b) ** 2) * w_b)
            const = const + (-0.5) * (dd_i + hh_i)
            # h_b already carries the tc_0 phase; phase_mat_full applies the relative
            # shift exp(-2πi f Δtc) only (Δtc=tc_k-tc_0), matching the RB convention.
            # Stripping tc_0 here and re-adding only Δtc drops tc_0 from the template.
            integrand = jnp.conj(d_b) * h_b * w_b
            W_tc      = W_tc + (phase_mat_full @ integrand)
        x = jnp.abs(W_tc)
        log_bessel  = jnp.log(jax.scipy.special.i0e(x)) + x
        log_L_phi_tc = const + log_bessel
        return jax.scipy.special.logsumexp(log_L_phi_tc) - log_N_tc

    return log_L_full_tp_marg


# ============================================================================
# Convenience: SNR at fiducial template
# ============================================================================

def compute_matched_filter_snr(
    network:         Network,
    waveform_fn:     Callable,
    fiducial_params: dict,
    gmst:            float = 0.0,
) -> dict:
    """Optimal matched-filter SNR for the fiducial template (debug helper).

    The *optimal* (phase-maximised) SNR is

        ρ_opt² = ⟨h|h⟩ = Σ_k |h_k|² · w_k

    where ``w_k = 4·df·mask_k/PSD_k``.  For GW170817 with H1+L1+V1 and a
    decent fiducial template, the network value should be ≈ 32.
    """
    ifos, freqs_band, fmask, d_list, w_list = _band_data(network)
    fid_jax = {k: jnp.asarray(v) for k, v in fiducial_params.items()}
    snr_opt_det:    list = []
    snr_signed_det: list = []
    for ifo, d_b, w_b in zip(ifos, d_list, w_list):
        h_b = _project_at_freqs(waveform_fn, fid_jax, freqs_band, ifo, gmst=gmst)
        hh  = float(jnp.sum((jnp.abs(h_b) ** 2) * w_b))
        dh  = float(jnp.real(jnp.sum(jnp.conj(d_b) * h_b * w_b)))
        rho = float(np.sqrt(max(0.0, hh)))
        snr_opt_det.append(rho)
        snr_signed_det.append(dh / rho if rho > 0 else 0.0)
    snr_opt_network = float(np.sqrt(np.sum(np.asarray(snr_opt_det) ** 2)))
    return {
        "per_ifo":           dict(zip([i.name for i in ifos], snr_opt_det)),
        "per_ifo_signed":    dict(zip([i.name for i in ifos], snr_signed_det)),
        "network":           snr_opt_network,
    }
