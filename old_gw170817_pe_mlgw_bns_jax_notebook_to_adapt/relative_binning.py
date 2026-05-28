"""Relative binning likelihood for SHARPy GW parameter estimation.

Implements the method of Zackay, Dai & Venumadhav (2018) [arXiv:1806.08792]
to reduce the cost of each likelihood evaluation from O(N_freq) to O(N_bins),
where N_bins << N_freq.  This allows SHARPy's SMC sampler to run far more
particles simultaneously without exhausting GPU/CPU memory.

Memory benefit
--------------
Typical GW170817 analysis (T = 128 s, f ∈ [23, 2000] Hz):
    N_freq ≈ 3 000 points (after resampling) or ≈ 8 192 (full grid)
    N_bins ≈ 100 – 500 points (configurable)
    → 6 – 80× fewer complex numbers to store per particle per detector

Convention (matches SHARPy ``GW_likelihood.py`` exactly)
---------------------------------------------------------
    log L = −TwoDeltaTOverN · Σ_k |d_k − h_k|² / σ²_k

where σ²_k = PSD(f_k) · dt² and TwoDeltaTOverN = 2·dt/N.

Relative-binning summary quantities per detector, per frequency bin j
----------------------------------------------------------------------
    A0_j = Σ_{k∈bin_j} conj(d_k) · h0_k / σ²_k        (complex, NOT conjugated on h0)
    B0_j = Σ_{k∈bin_j} |h0_k|² / σ²_k                  (real, positive)
    dd   = Σ_k |d_k|² / σ²_k                            (real, constant)

Approximate likelihood
----------------------
    log L_RB ≈ −TwoDeltaTOverN · (dd − 2·Re[Σ_j r_j · A0_j] + Σ_j |r_j|² · B0_j)

where  r_j = H_proj(f_j ; θ) / H_proj(f_j ; θ_0)  is the ratio of the
projected detector strain at bin-centre frequency f_j, evaluated for
candidate parameters θ relative to the fiducial parameters θ_0.

The key approximation  h(f) ≈ r_j · h0(f)  for f ∈ bin_j  is accurate
when the bins are chosen fine enough that the waveform ratio varies by
≲ 0.1 rad in phase and ≲ 1 % in amplitude within each bin.

Note on waveform convention
---------------------------
The template function should return (hp, hc) WITHOUT applying an
additional conjugation.  The mlgw_bns_jax predict function returns
    hp(f) = pre_plus · A(f) · exp(+i φ(f))
in the standard positive-frequency Fourier convention.  A coalescence-
phase rotation exp(−i φ_c) is applied inside the template wrapper
(see ``_template_wrapper`` below) as an explicit overall phase, NOT as a
conjugation of the waveform.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ── SHARPy imports (needed for antenna patterns and time delay) ────────
from sharpy.GW_likelihood import antenna_pattern_functions
from sharpy.utils import TimeDelayFromEarthCenter, McQ2Masses


# ======================================================================
# Data structures
# ======================================================================

class RBSummaryDet(NamedTuple):
    """Precomputed relative-binning summary data for one detector."""
    A0: jnp.ndarray    # shape (n_bins,) complex — Σ conj(d)·h0/σ²  per bin
    B0: jnp.ndarray    # shape (n_bins,) real   — Σ |h0|²/σ²       per bin
    dd: float          # scalar real             — Σ |d|²/σ²
    TwoDeltaTOverN: float
    # Reference projected waveform at bin centres (for ratio computation)
    h0_bins: jnp.ndarray   # shape (n_bins,) complex


class RBNetwork(NamedTuple):
    """Relative-binning data for the full detector network."""
    summaries: list          # list of RBSummaryDet, one per detector
    f_bins: jnp.ndarray      # shape (n_bins,) — bin centre frequencies
    # Detector geometry (needed at evaluation time)
    latitudes: jnp.ndarray   # shape (n_det,)
    longitudes: jnp.ndarray
    elevations: jnp.ndarray
    gammas: jnp.ndarray
    zetas: jnp.ndarray
    trigtimes: jnp.ndarray
    T_durations: jnp.ndarray


# ======================================================================
# Bin-edge selection
# ======================================================================

def choose_bins_pn(
    f: np.ndarray,
    h0: np.ndarray,
    n_bins: int = 400,
    max_bin_width_hz: float = 8.0,
) -> np.ndarray:
    """Choose frequency bin edges based on accumulated PN chirp phase.

    Bin boundaries are placed so that:
    1. The accumulated PN chirp phase within each bin is approximately equal.
       Uses the analytical proxy ``f_low^{-5/3} - f^{-5/3}`` (proportional
       to the leading-order PN phase) rather than the absolute phase of the
       projected waveform h0.  This is critical for BNS signals: the
       projected waveform includes a large time-shift term
       ``exp(-i 2π f · timeshift)`` whose phase derivative ≈ 2π · 127 s can
       dominate h0's phase near ``f_low`` and create a spurious "slow-phase"
       saddle point that places too few bins at 26–35 Hz, causing
       ``|ΔlogL| ≈ 3–9`` errors for ±1σ chirp-mass perturbations.
       The PN proxy is free from this contamination and gives per-bin
       phase variation < 0.03 rad for δmc = 0.001 M☉ on GW170817-like data.
    2. No bin is wider than ``max_bin_width_hz`` Hz (coalescence-time accuracy —
       prevents large linear-phase errors at high frequencies where the PN
       phase varies slowly).

    Parameters
    ----------
    f : np.ndarray, shape (N,)
        Frequency array in Hz.
    h0 : np.ndarray, shape (N,) complex
        Projected reference waveform at all frequencies.  Retained in the
        signature for API compatibility; no longer used for bin selection.
    n_bins : int
        Target number of bins.
    max_bin_width_hz : float
        Maximum allowed bin width in Hz.  Default 8 Hz → ≤ 0.05 rad phase
        error per bin for δt_c ≤ 1 ms.

    Returns
    -------
    bin_edges : np.ndarray, shape (≥ 2,)
        Frequencies of bin boundaries, including f[0] and f[-1].
        Actual bin count ≥ n_bins due to the max-width constraint.
    """
    # PN-proxy cumulative phase: C(f) = f_low^{-5/3} - f^{-5/3}
    # Proportional to the accumulated PN chirp phase from f_low to f.
    # Placing equal C increments per bin is equivalent to equal-rate-of-
    # phase-change binning and gives bin widths ∝ f^{8/3} — narrow at low
    # frequencies where the chirp-mass sensitivity is highest.
    cum_phase = f[0] ** (-5.0 / 3.0) - f ** (-5.0 / 3.0)  # monotone increasing
    total_phase = cum_phase[-1]
    if total_phase < 1e-10:
        edges_phase = np.linspace(f[0], f[-1], n_bins + 1)
    else:
        target_phases = np.linspace(0.0, total_phase, n_bins + 1)
        edges_phase = np.interp(target_phases, cum_phase, f)
        edges_phase[0] = f[0]
        edges_phase[-1] = f[-1]

    # Frequency-based edges (maximum bin width constraint for tc accuracy)
    n_freq_bins = max(n_bins, int(np.ceil((f[-1] - f[0]) / max_bin_width_hz)) + 1)
    edges_freq = np.linspace(f[0], f[-1], n_freq_bins + 1)

    # Merge both sets of edges and keep unique values
    edges_merged = np.unique(np.concatenate([edges_phase, edges_freq]))
    edges_merged = edges_merged[
        (edges_merged >= f[0]) & (edges_merged <= f[-1])
    ]
    edges_merged[0] = f[0]
    edges_merged[-1] = f[-1]
    return edges_merged


def bin_centres(bin_edges: np.ndarray) -> np.ndarray:
    """Return the centre frequency of each bin."""
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


# ======================================================================
# Projected waveform evaluation
# ======================================================================

def project_waveform_at_freqs(
    params: jnp.ndarray,
    f: jnp.ndarray,
    template_fn: Callable,
    lat: float,
    lon: float,
    elev: float,
    gamma: float,
    zeta: float,
    trigtime: float,
    T_duration: float,
) -> jnp.ndarray:
    """Evaluate the projected detector strain at arbitrary frequencies ``f``.

    This mirrors SHARPy's ``project_waveform`` but accepts a custom frequency
    array and an arbitrary template function, making it usable at both the
    full grid and the sparse bin centres.

    Parameters
    ----------
    params : array, shape (13,)
        SHARPy parameter vector:
        [ra, dec, logdist, incl, phic, pol, mc, q, tc,
         chi1, chi2, lambda1, lambda2]
    f : array, shape (K,)
        Frequencies at which to evaluate the strain, in Hz.
    template_fn : callable
        (params, f) → (hp, hc) — waveform template.
    lat, lon, elev, gamma, zeta : float
        Detector geometry.
    trigtime : float
        GPS trigger time of the detector segment.
    T_duration : float
        Segment duration in seconds.

    Returns
    -------
    h : array, shape (K,) complex
        Projected strain at frequencies ``f``.
    """
    hp, hc = template_fn(params, f)

    fplus, fcross = antenna_pattern_functions(
        params, lat, lon, gamma, zeta, trigtime
    )

    ra = params[0]
    dec = params[1]
    tc = trigtime + params[8]
    timedelay = TimeDelayFromEarthCenter(lat, lon, elev, ra, dec, tc)
    timeshift = timedelay + params[8] + (T_duration - 1.0)
    shift = 2.0 * jnp.pi * f * timeshift

    h = (fplus * hp + fcross * hc) * (jnp.cos(shift) - 1j * jnp.sin(shift))
    return h


# ======================================================================
# Summary-data precomputation  (NumPy — runs once before the sampler)
# ======================================================================

def precompute_rb_summary(
    f_full: np.ndarray,
    data: np.ndarray,
    h0_full: np.ndarray,
    sigmasq: np.ndarray,
    TwoDeltaTOverN: float,
    bin_edges: np.ndarray,
    f_bins: np.ndarray,
    h0_bins: np.ndarray,
) -> RBSummaryDet:
    """Compute per-bin summary quantities for one detector.

    Parameters
    ----------
    f_full : array, shape (N,)
    data : array, shape (N,) complex — d(f), the observed frequency-domain data
    h0_full : array, shape (N,) complex — projected reference waveform H0(f)
    sigmasq : array, shape (N,) — PSD · dt²
    TwoDeltaTOverN : float
    bin_edges : array, shape (n_bins+1,) — bin boundaries
    f_bins : array, shape (n_bins,) — bin centres
    h0_bins : array, shape (n_bins,) complex — H0 at bin centres

    Returns
    -------
    RBSummaryDet
    """
    n_bins = len(f_bins)
    A0 = np.zeros(n_bins, dtype=complex)
    B0 = np.zeros(n_bins, dtype=float)
    dd = float(np.sum(np.abs(data) ** 2 / sigmasq))

    # Assign each frequency point to a bin
    bin_indices = np.searchsorted(bin_edges, f_full, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for j in range(n_bins):
        mask = bin_indices == j
        if not np.any(mask):
            continue
        d_j = data[mask]
        h0_j = h0_full[mask]
        s_j = sigmasq[mask]
        # A0_j = Σ conj(d) · h0 / σ²  — NO conjugation on h0
        A0[j] = np.sum(np.conj(d_j) * h0_j / s_j)
        B0[j] = float(np.sum(np.abs(h0_j) ** 2 / s_j))

    return RBSummaryDet(
        A0=jnp.array(A0, dtype=jnp.complex128),
        B0=jnp.array(B0, dtype=jnp.float64),
        dd=float(dd),
        TwoDeltaTOverN=float(TwoDeltaTOverN),
        h0_bins=jnp.array(h0_bins, dtype=jnp.complex128),
    )


# ======================================================================
# JAX-jittable single-detector RB likelihood
# ======================================================================

def _single_det_rb_logL(
    r: jnp.ndarray,
    summary: RBSummaryDet,
) -> jnp.ndarray:
    """Relative-binning log-likelihood for one detector.

    Parameters
    ----------
    r : array, shape (n_bins,) complex
        Waveform ratio H(f_j; θ) / H0(f_j; θ_0) at bin centres.
    summary : RBSummaryDet
        Precomputed per-bin summary data.

    Returns
    -------
    log_L : scalar
    """
    # cross term:  Re[ Σ_j r_j · A0_j ]
    cross = jnp.real(jnp.sum(r * summary.A0))
    # self term:   Σ_j |r_j|² · B0_j
    self_term = jnp.sum(jnp.abs(r) ** 2 * summary.B0)
    # log L = -TwoDeltaTOverN * (dd - 2·cross + self_term)
    return -summary.TwoDeltaTOverN * (summary.dd - 2.0 * cross + self_term)


# ======================================================================
# Main builder: returns a fast log-likelihood function
# ======================================================================

def build_rb_likelihood(
    batched_detector,
    fiducial_params: np.ndarray,
    template_fn: Callable,
    n_bins: int = 400,
) -> tuple[Callable, RBNetwork]:
    """Precompute relative-binning summary data and return a fast likelihood.

    Parameters
    ----------
    batched_detector : sharpy Detector (stacked)
        Output of ``GWNetwork.batched_detector``.
    fiducial_params : array, shape (13,)
        Reference parameter vector θ_0 (best-guess or approximate MAP).
        Should be close to the true signal parameters for accuracy.
    template_fn : callable
        (params, freq_array) → (hp, hc).
        Must NOT apply any additional conjugation to the polarisations.
    n_bins : int
        Number of frequency bins.  More bins → better accuracy but higher
        cost per evaluation.  400 bins is a good default for BNS PE.

    Returns
    -------
    log_likelihood_rb : callable
        (params_13) → scalar.  Drop-in replacement for SHARPy's
        ``log_likelihood_det``.
    rb_network : RBNetwork
        Struct storing all precomputed data (useful for inspection/testing).
    """
    n_det = len(batched_detector.latitude)

    # ── Extract geometry arrays ──────────────────────────────────────
    lats = np.array(batched_detector.latitude)
    lons = np.array(batched_detector.longitude)
    elevs = np.array(batched_detector.elevation)
    gammas = np.array(batched_detector.gamma)
    zetas = np.array(batched_detector.zeta)
    trigtimes = np.array(batched_detector.trigtime)
    T_durs = np.array(batched_detector.T)

    # ── Use first detector's frequency array for bin selection ───────
    f_full_np = np.array(batched_detector.Frequency[0])

    # ── Compute reference projected waveform for each detector ───────
    print("[RB] Computing reference waveform on full grid …")
    h0_full_list = []
    for i in range(n_det):
        h0_i = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                jnp.array(f_full_np, dtype=jnp.float64),
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )
        h0_full_list.append(h0_i)

    # Average-amplitude reference for bin selection
    h0_avg = np.mean(np.abs(np.stack(h0_full_list)), axis=0)
    h0_phase_ref = h0_full_list[0]  # use first detector's phase for bins

    # ── Choose bin edges ─────────────────────────────────────────────
    print(f"[RB] Choosing {n_bins} frequency bins …")
    # Use the network-averaged amplitude weighted by the phase of det 0
    h0_for_bins = h0_avg * np.exp(1j * np.angle(h0_phase_ref))
    # Mask out zero-amplitude points (outside signal band)
    amp_mask = h0_avg > 0.0
    if np.sum(amp_mask) < n_bins:
        # Too few non-zero points — fall back to uniform bins
        bin_edges = np.linspace(f_full_np[0], f_full_np[-1], n_bins + 1)
    else:
        bin_edges = choose_bins_pn(f_full_np, h0_for_bins, n_bins=n_bins)

    f_bins_np = bin_centres(bin_edges)
    f_bins_jax = jnp.array(f_bins_np, dtype=jnp.float64)

    # ── Compute reference waveform at bin centres for each detector ──
    h0_bins_list = []
    for i in range(n_det):
        h0_b = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                f_bins_jax,
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )
        h0_bins_list.append(h0_b)

    # ── Precompute summary data per detector ─────────────────────────
    print("[RB] Precomputing per-detector summary data …")
    summaries = []
    for i in range(n_det):
        data_i = np.array(batched_detector.FrequencySeries[i])
        sigmasq_i = np.array(batched_detector.sigmasq[i])
        TwoDTN_i = float(batched_detector.TwoDeltaTOverN[i])
        summary_i = precompute_rb_summary(
            f_full_np,
            data_i,
            h0_full_list[i],
            sigmasq_i,
            TwoDTN_i,
            bin_edges,
            f_bins_np,
            h0_bins_list[i],
        )
        summaries.append(summary_i)
        print(f"  det {i}: dd={summary_i.dd:.3e}, "
              f"|A0| max={float(jnp.max(jnp.abs(summary_i.A0))):.3e}, "
              f"B0 sum={float(jnp.sum(summary_i.B0)):.3e}")

    n_bins_actual = len(f_bins_np)
    print(f"[RB] Ready.  Full grid: {len(f_full_np)} pts → "
          f"RB bins: {n_bins_actual} pts "
          f"({len(f_full_np)//n_bins_actual}× reduction)")

    rb_network = RBNetwork(
        summaries=summaries,
        f_bins=f_bins_jax,
        latitudes=jnp.array(lats),
        longitudes=jnp.array(lons),
        elevations=jnp.array(elevs),
        gammas=jnp.array(gammas),
        zetas=jnp.array(zetas),
        trigtimes=jnp.array(trigtimes),
        T_durations=jnp.array(T_durs),
    )

    # ── Build JAX-jittable likelihood ────────────────────────────────
    # Freeze all arrays into the closure
    f_bins_frozen = f_bins_jax
    summaries_frozen = summaries
    lats_frozen = rb_network.latitudes
    lons_frozen = rb_network.longitudes
    elevs_frozen = rb_network.elevations
    gammas_frozen = rb_network.gammas
    zetas_frozen = rb_network.zetas
    trigs_frozen = rb_network.trigtimes
    Ts_frozen = rb_network.T_durations

    # Pre-extract Python scalars for the closure (needed for JAX JIT)
    lats_py = [float(rb_network.latitudes[i]) for i in range(n_det)]
    lons_py = [float(rb_network.longitudes[i]) for i in range(n_det)]
    elevs_py = [float(rb_network.elevations[i]) for i in range(n_det)]
    gammas_py = [float(rb_network.gammas[i]) for i in range(n_det)]
    zetas_py = [float(rb_network.zetas[i]) for i in range(n_det)]
    trigs_py = [float(rb_network.trigtimes[i]) for i in range(n_det)]
    Ts_py = [float(rb_network.T_durations[i]) for i in range(n_det)]

    def log_likelihood_rb(params: jnp.ndarray) -> jnp.ndarray:
        """Relative-binning log-likelihood (fast).

        Parameters
        ----------
        params : array, shape (13,)
            SHARPy parameter vector.

        Returns
        -------
        log_L : scalar
        """
        log_L = jnp.float64(0.0)
        for i in range(n_det):
            # Evaluate projected waveform at bin centres only
            # All geometry values are Python scalars (JAX-JIT compatible)
            h_bins_i = project_waveform_at_freqs(
                params,
                f_bins_frozen,
                template_fn,
                lats_py[i],
                lons_py[i],
                elevs_py[i],
                gammas_py[i],
                zetas_py[i],
                trigs_py[i],
                Ts_py[i],
            )
            # Waveform ratio (no conjugation on h)
            h0_b = summaries_frozen[i].h0_bins
            # Avoid division by zero using a small guard
            r_i = jnp.where(
                jnp.abs(h0_b) > 0.0,
                h_bins_i / h0_b,
                jnp.zeros_like(h_bins_i),
            )
            log_L = log_L + _single_det_rb_logL(r_i, summaries_frozen[i])
        return log_L

    return log_likelihood_rb, rb_network


# ======================================================================
# Time-marginalized RB likelihood
# ======================================================================

def build_rb_likelihood_time_marg(
    batched_detector,
    fiducial_params: np.ndarray,
    template_fn: Callable,
    tc_grid: np.ndarray,
    n_bins: int = 400,
) -> tuple[Callable, RBNetwork]:
    """Build a time-marginalized relative-binning log-likelihood.

    Analytically marginalises over the coalescence time ``tc`` (``params[8]``)
    by summing over a discrete grid of tc values, using the approximation

        h_proj(f; tc) ≈ h_proj(f; tc₀) · exp(−i 2π f (tc − tc₀))

    which is accurate when the angular frequency changes by ≪1 rad across
    the tc integration range (typically ±150 ms).

    The returned function takes a 12-element parameter vector **without tc**
    (all other SHARPy parameters in order, with index-8 removed):
        [ra, dec, logdist, incl, phic, pol, mc, q,
         chi1, chi2, lambda1, lambda2]

    For each tc_k in ``tc_grid``:
        log L(tc_k) = −Σ_det TwoDTN_det · (dd_det − 2·Re[r⁰·A0 · exp(−i2πf Δtc_k)] + |r⁰|²·B0_det)
    Then: log L_marg = logsumexp_k(log L(tc_k)) − log(N_tc)

    Parameters
    ----------
    batched_detector : sharpy Detector (stacked)
    fiducial_params : array, shape (13,)
        SHARPy reference parameter vector including tc at index 8.
    template_fn : callable
        (params_13, freq) → (hp, hc).
    tc_grid : array, shape (N_tc,)
        Discrete tc values (params[8], i.e. seconds relative to trigtime) to
        integrate over.  3000 points over ±150 ms is typical.
    n_bins : int
        Number of RB frequency bins.

    Returns
    -------
    log_likelihood_rb_time_marg : callable
        (params_no_tc_12,) → scalar (JAX-jittable).
    rb_network : RBNetwork
    """
    n_det = len(batched_detector.latitude)

    # ── Extract geometry ─────────────────────────────────────────────
    lats = np.array(batched_detector.latitude)
    lons = np.array(batched_detector.longitude)
    elevs = np.array(batched_detector.elevation)
    gammas = np.array(batched_detector.gamma)
    zetas = np.array(batched_detector.zeta)
    trigtimes = np.array(batched_detector.trigtime)
    T_durs = np.array(batched_detector.T)

    # ── Full-grid frequency array ────────────────────────────────────
    f_full_np = np.array(batched_detector.Frequency[0])

    # ── Reference projected waveform (at fiducial params) ───────────
    print("[RB-TM] Computing reference waveform on full grid …")
    h0_full_list = []
    for i in range(n_det):
        h0_i = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                jnp.array(f_full_np, dtype=jnp.float64),
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )
        h0_full_list.append(h0_i)

    h0_avg = np.mean(np.abs(np.stack(h0_full_list)), axis=0)
    h0_phase_ref = h0_full_list[0]

    # ── Choose bin edges ─────────────────────────────────────────────
    print(f"[RB-TM] Choosing {n_bins} frequency bins …")
    h0_for_bins = h0_avg * np.exp(1j * np.angle(h0_phase_ref))
    if np.sum(h0_avg > 0.0) < n_bins:
        bin_edges = np.linspace(f_full_np[0], f_full_np[-1], n_bins + 1)
    else:
        bin_edges = choose_bins_pn(f_full_np, h0_for_bins, n_bins=n_bins)

    f_bins_np = bin_centres(bin_edges)
    f_bins_jax = jnp.array(f_bins_np, dtype=jnp.float64)

    # ── Reference waveform at bin centres ────────────────────────────
    h0_bins_list = []
    for i in range(n_det):
        h0_b = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                f_bins_jax,
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )
        h0_bins_list.append(h0_b)

    # ── Per-detector summary data ────────────────────────────────────
    print("[RB-TM] Precomputing per-detector summary data …")
    summaries = []
    for i in range(n_det):
        data_i = np.array(batched_detector.FrequencySeries[i])
        sigmasq_i = np.array(batched_detector.sigmasq[i])
        TwoDTN_i = float(batched_detector.TwoDeltaTOverN[i])
        summary_i = precompute_rb_summary(
            f_full_np, data_i, h0_full_list[i], sigmasq_i, TwoDTN_i,
            bin_edges, f_bins_np, h0_bins_list[i],
        )
        summaries.append(summary_i)

    rb_network = RBNetwork(
        summaries=summaries,
        f_bins=f_bins_jax,
        latitudes=jnp.array(lats),
        longitudes=jnp.array(lons),
        elevations=jnp.array(elevs),
        gammas=jnp.array(gammas),
        zetas=jnp.array(zetas),
        trigtimes=jnp.array(trigtimes),
        T_durations=jnp.array(T_durs),
    )

    # ── Precompute tc phase matrix ────────────────────────────────────
    # tc_delta[k] = tc_grid[k] - tc_0
    tc_0 = float(fiducial_params[8])
    tc_delta = jnp.array(tc_grid, dtype=jnp.float64) - tc_0   # (N_tc,)
    N_tc = len(tc_grid)

    # phase_matrix[k, j] = exp(-i·2π·f_j·Δtc_k)   shape (N_tc, n_bins)
    phase_matrix = jnp.exp(
        -1j * 2.0 * jnp.pi * tc_delta[:, None] * f_bins_jax[None, :]
    )  # (N_tc, n_bins) complex128

    # ── Freeze scalars for JIT closure ───────────────────────────────
    lats_py    = [float(lats[i])     for i in range(n_det)]
    lons_py    = [float(lons[i])     for i in range(n_det)]
    elevs_py   = [float(elevs[i])    for i in range(n_det)]
    gammas_py  = [float(gammas[i])   for i in range(n_det)]
    zetas_py   = [float(zetas[i])    for i in range(n_det)]
    trigs_py   = [float(trigtimes[i]) for i in range(n_det)]
    Ts_py      = [float(T_durs[i])   for i in range(n_det)]
    tc_0_py    = float(tc_0)

    n_bins_actual = len(f_bins_np)
    print(
        f"[RB-TM] Ready.  Full grid: {len(f_full_np)} pts → "
        f"RB bins: {n_bins_actual} pts, tc grid: {N_tc} pts "
        f"({len(f_full_np) // max(n_bins_actual, 1)}× freq reduction)"
    )

    def log_likelihood_rb_time_marg(params_no_tc: jnp.ndarray) -> jnp.ndarray:
        """Time-marginalized RB log-likelihood.

        Parameters
        ----------
        params_no_tc : array, shape (12,)
            SHARPy parameters with tc removed:
            [ra, dec, logdist, incl, phic, pol, mc, q,
             chi1, chi2, lambda1, lambda2]

        Returns
        -------
        log_L_marg : scalar
        """
        # Re-insert fiducial tc at index 8
        params_13 = jnp.concatenate([
            params_no_tc[:8],
            jnp.array([tc_0_py], dtype=jnp.float64),
            params_no_tc[8:],
        ])

        # Accumulate log_L(tc_k) for all detectors
        log_L_tc = jnp.zeros(N_tc, dtype=jnp.float64)

        for i in range(n_det):
            # Projected waveform at bin centres (at fiducial tc)
            h_bins_i = project_waveform_at_freqs(
                params_13, f_bins_jax,
                template_fn,
                lats_py[i], lons_py[i], elevs_py[i],
                gammas_py[i], zetas_py[i],
                trigs_py[i], Ts_py[i],
            )
            h0_b = summaries[i].h0_bins
            # Waveform ratio at tc_0 (tc variation handled via phase_matrix)
            r_i = jnp.where(
                jnp.abs(h0_b) > 0.0,
                h_bins_i / h0_b,
                jnp.zeros_like(h_bins_i),
            )
            # self_term: independent of tc
            self_term_i = jnp.sum(jnp.abs(r_i) ** 2 * summaries[i].B0)
            # cross(tc_k) = Re[ phase_matrix @ (r_j · A0_j) ]
            A0_weight = r_i * summaries[i].A0           # (n_bins,) complex
            cross_tc = jnp.real(phase_matrix @ A0_weight)  # (N_tc,)
            TwoDTN_i = summaries[i].TwoDeltaTOverN
            log_L_tc = log_L_tc + (
                -TwoDTN_i * (summaries[i].dd - 2.0 * cross_tc + self_term_i)
            )

        return jax.scipy.special.logsumexp(log_L_tc) - jnp.log(
            jnp.array(N_tc, dtype=jnp.float64)
        )

    return log_likelihood_rb_time_marg, rb_network


def build_rb_likelihood_tc_phi_marg(
    batched_detector,
    fiducial_params: np.ndarray,
    template_fn: Callable,
    tc_grid: np.ndarray,
    n_bins: int = 400,
) -> tuple[Callable, RBNetwork]:
    """Build a time- and phase-marginalized relative-binning log-likelihood.

    Analytically marginalises over both the coalescence time ``tc`` (``params[8]``)
    and the coalescence phase ``φc`` (``params[4]``).

    The phase marginalisation uses the Bessel-function identity
        ∫₀^{2π} exp(x·cos θ) dθ / (2π) = I₀(x)
    so that for each tc value:
        log L_φ(tc) = const_det + log I₀(2 · |W(tc)|)
    where
        const_det = −Σ_det TwoDTN_det · (dd_det + hh_det)
        W(tc)     = Σ_det TwoDTN_det · Σ_j  r_j^(φ=0)(tc) · A0_j^(nophase)

    The log I₀ is computed in a numerically stable way as
        log I₀(x) = log(i0e(x)) + x
    where ``i0e(x) = I₀(x) · exp(−x)`` (``jax.scipy.special.i0e``).

    The returned function takes an 11-element parameter vector with **both
    tc and φc removed**:
        [ra, dec, logdist, incl, pol, mc, q, chi1, chi2, lambda1, lambda2]

    Parameters
    ----------
    batched_detector : sharpy Detector (stacked)
    fiducial_params : array, shape (13,)
    template_fn : callable
    tc_grid : array, shape (N_tc,)
    n_bins : int

    Returns
    -------
    log_likelihood_rb_tc_phi_marg : callable
        (params_no_tc_no_phi_11,) → scalar (JAX-jittable).
    rb_network : RBNetwork
    """
    n_det = len(batched_detector.latitude)

    # ── Extract geometry ─────────────────────────────────────────────
    lats     = np.array(batched_detector.latitude)
    lons     = np.array(batched_detector.longitude)
    elevs    = np.array(batched_detector.elevation)
    gammas   = np.array(batched_detector.gamma)
    zetas    = np.array(batched_detector.zeta)
    trigtimes = np.array(batched_detector.trigtime)
    T_durs   = np.array(batched_detector.T)

    f_full_np = np.array(batched_detector.Frequency[0])

    # ── Reference waveform at fiducial params ────────────────────────
    print("[RB-TP] Computing reference waveform on full grid …")
    h0_full_list = []
    for i in range(n_det):
        h0_i = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                jnp.array(f_full_np, dtype=jnp.float64),
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )
        h0_full_list.append(h0_i)

    h0_avg = np.mean(np.abs(np.stack(h0_full_list)), axis=0)
    h0_phase_ref = h0_full_list[0]

    print(f"[RB-TP] Choosing {n_bins} frequency bins …")
    h0_for_bins = h0_avg * np.exp(1j * np.angle(h0_phase_ref))
    if np.sum(h0_avg > 0.0) < n_bins:
        bin_edges = np.linspace(f_full_np[0], f_full_np[-1], n_bins + 1)
    else:
        bin_edges = choose_bins_pn(f_full_np, h0_for_bins, n_bins=n_bins)

    f_bins_np = bin_centres(bin_edges)
    f_bins_jax = jnp.array(f_bins_np, dtype=jnp.float64)

    h0_bins_list = []
    for i in range(n_det):
        h0_b = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                f_bins_jax,
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )
        h0_bins_list.append(h0_b)

    print("[RB-TP] Precomputing per-detector summary data …")
    summaries = []
    for i in range(n_det):
        data_i   = np.array(batched_detector.FrequencySeries[i])
        sigmasq_i = np.array(batched_detector.sigmasq[i])
        TwoDTN_i = float(batched_detector.TwoDeltaTOverN[i])
        summary_i = precompute_rb_summary(
            f_full_np, data_i, h0_full_list[i], sigmasq_i, TwoDTN_i,
            bin_edges, f_bins_np, h0_bins_list[i],
        )
        summaries.append(summary_i)

    rb_network = RBNetwork(
        summaries=summaries,
        f_bins=f_bins_jax,
        latitudes=jnp.array(lats),
        longitudes=jnp.array(lons),
        elevations=jnp.array(elevs),
        gammas=jnp.array(gammas),
        zetas=jnp.array(zetas),
        trigtimes=jnp.array(trigtimes),
        T_durations=jnp.array(T_durs),
    )

    # ── Fiducial phase and tc ────────────────────────────────────────
    phic_0 = float(fiducial_params[4])
    tc_0   = float(fiducial_params[8])
    N_tc   = len(tc_grid)

    tc_delta = jnp.array(tc_grid, dtype=jnp.float64) - tc_0  # (N_tc,)

    # phase_matrix[k, j] = exp(-i·2π·f_j·Δtc_k)
    phase_matrix = jnp.exp(
        -1j * 2.0 * jnp.pi * tc_delta[:, None] * f_bins_jax[None, :]
    )  # (N_tc, n_bins)

    # Phase-strip factor for A0: removes fiducial phic from A0
    # A0_j ∝ exp(-i·phic_0)  →  A0_nophic_j = A0_j · exp(+i·phic_0)
    strip_phic = np.exp(1j * phic_0)
    # Precomputed phase-stripped A0 for each detector
    A0_nophic_list = [
        jnp.array(np.array(summaries[i].A0) * strip_phic, dtype=jnp.complex128)
        for i in range(n_det)
    ]

    # Phase-stripped h0_bins_noPhic = h0_bins · exp(+i·phic_0)
    h0_bins_nophic_list = [
        jnp.array(np.array(h0_bins_list[i]) * strip_phic, dtype=jnp.complex128)
        for i in range(n_det)
    ]

    # Freeze scalars
    lats_py   = [float(lats[i])      for i in range(n_det)]
    lons_py   = [float(lons[i])      for i in range(n_det)]
    elevs_py  = [float(elevs[i])     for i in range(n_det)]
    gammas_py = [float(gammas[i])    for i in range(n_det)]
    zetas_py  = [float(zetas[i])     for i in range(n_det)]
    trigs_py  = [float(trigtimes[i]) for i in range(n_det)]
    Ts_py     = [float(T_durs[i])    for i in range(n_det)]
    tc_0_py   = float(tc_0)
    phic_0_py = float(phic_0)

    n_bins_actual = len(f_bins_np)
    print(
        f"[RB-TP] Ready.  Full grid: {len(f_full_np)} pts → "
        f"RB bins: {n_bins_actual} pts, tc grid: {N_tc} pts"
    )

    def log_likelihood_rb_tc_phi_marg(params_no_tc_no_phi: jnp.ndarray) -> jnp.ndarray:
        """Time- and phase-marginalized RB log-likelihood.

        Parameters
        ----------
        params_no_tc_no_phi : array, shape (11,)
            [ra, dec, logdist, incl, pol, mc, q, chi1, chi2, lambda1, lambda2]
            (tc at index 8 and phic at index 4 both removed; pol is now at index 4)

        Returns
        -------
        log_L_marg : scalar
        """
        # Re-insert phic=0 at index 4, then tc=tc_0 at index 8
        # params_no_tc_no_phi: [0:ra, 1:dec, 2:logdist, 3:incl, 4:pol, 5:mc, 6:q,
        #                        7:chi1, 8:chi2, 9:lambda1, 10:lambda2]
        params_13 = jnp.concatenate([
            params_no_tc_no_phi[:4],                          # ra, dec, logdist, incl
            jnp.array([0.0], dtype=jnp.float64),              # phic = 0
            params_no_tc_no_phi[4:7],                         # pol, mc, q
            jnp.array([tc_0_py], dtype=jnp.float64),          # tc = tc_0
            params_no_tc_no_phi[7:],                          # chi1, chi2, lambda1, lambda2
        ])

        # Accumulate: constant term, weighted A0 sum for |W(tc)|
        const = jnp.float64(0.0)
        A0_total = jnp.zeros(N_tc, dtype=jnp.complex128)   # Σ_det TwoDTN·Z_det(tc)

        for i in range(n_det):
            # Evaluate template at phic=0, tc=tc_0
            h_bins_i = project_waveform_at_freqs(
                params_13, f_bins_jax,
                template_fn,
                lats_py[i], lons_py[i], elevs_py[i],
                gammas_py[i], zetas_py[i],
                trigs_py[i], Ts_py[i],
            )
            h0_b_nophic = h0_bins_nophic_list[i]  # h0_bins · exp(+i·phic_0)
            # Ratio using phase-stripped reference (so r⁰ is phic-independent)
            r_nophic = jnp.where(
                jnp.abs(h0_b_nophic) > 0.0,
                h_bins_i / h0_b_nophic,
                jnp.zeros_like(h_bins_i),
            )
            # hh (self-overlap, phic-independent)
            hh_i = jnp.sum(jnp.abs(r_nophic) ** 2 * summaries[i].B0)
            TwoDTN_i = summaries[i].TwoDeltaTOverN

            # const contribution: -TwoDTN*(dd + hh)
            const = const + (-TwoDTN_i * (summaries[i].dd + hh_i))

            # W(tc_k) += TwoDTN_i · Σ_j r_j^(nophic) · A0_j^(nophic) · exp(-i2πfΔtc)
            A0_weight = r_nophic * A0_nophic_list[i]         # (n_bins,) complex
            A0_total = A0_total + TwoDTN_i * (phase_matrix @ A0_weight)  # (N_tc,)

        # log I₀(2|W|) using numerically stable: log(i0e(x)) + x
        x = 2.0 * jnp.abs(A0_total)   # (N_tc,)
        log_bessel = jnp.log(jax.scipy.special.i0e(x)) + x  # (N_tc,)

        # log L_phi(tc_k) = const + log I₀(2|W|)
        log_L_phi_tc = const + log_bessel   # (N_tc,)

        # Marginalise over tc
        return jax.scipy.special.logsumexp(log_L_phi_tc) - jnp.log(
            jnp.array(N_tc, dtype=jnp.float64)
        )

    return log_likelihood_rb_tc_phi_marg, rb_network


# ======================================================================
# Full-likelihood time marginalization (no RB approximation)
# ======================================================================

def build_full_likelihood_time_marg(
    batched_detector,
    fiducial_params: np.ndarray,
    template_fn: Callable,
    tc_grid: np.ndarray,
) -> Callable:
    """Build a time-marginalized full (non-RB) log-likelihood.

    Uses the full frequency grid.  Cheaper per detector than the raw
    SHARPy likelihood (no waveform evaluation per tc), but more expensive
    than the RB variant.  Useful for validation.

    The marginalisation proceeds by precomputing
        integrand_det(f) = conj(d(f)) · h_noTc(f; θ) / σ²(f)
    and then computing the cross-correlation at each tc_k via
        cross_det(tc_k) = Re[ Σ_f integrand_det(f) · exp(−i2πf·tc_k) ]
    using a single matrix multiply over all tc values simultaneously.

    Parameters
    ----------
    batched_detector : sharpy Detector
    fiducial_params  : array, shape (13,)
    template_fn      : callable
    tc_grid          : array, shape (N_tc,)  — absolute tc values (params[8])

    Returns
    -------
    log_likelihood_full_time_marg : callable
        (params_no_tc_12,) → scalar.  Inserts tc=tc_0 when calling the
        waveform, then applies phase shifts for the tc grid.
    """
    n_det = len(batched_detector.latitude)
    lats     = np.array(batched_detector.latitude)
    lons     = np.array(batched_detector.longitude)
    elevs    = np.array(batched_detector.elevation)
    gammas   = np.array(batched_detector.gamma)
    zetas    = np.array(batched_detector.zeta)
    trigtimes = np.array(batched_detector.trigtime)
    T_durs   = np.array(batched_detector.T)

    f_full_np = np.array(batched_detector.Frequency[0])
    f_full_jax = jnp.array(f_full_np, dtype=jnp.float64)

    data_list    = [jnp.array(batched_detector.FrequencySeries[i], dtype=jnp.complex128)
                    for i in range(n_det)]
    sigmasq_list = [jnp.array(batched_detector.sigmasq[i], dtype=jnp.float64)
                    for i in range(n_det)]
    TwoDTN_list  = [float(batched_detector.TwoDeltaTOverN[i]) for i in range(n_det)]

    tc_0   = float(fiducial_params[8])
    N_tc   = len(tc_grid)
    tc_arr = jnp.array(tc_grid, dtype=jnp.float64)

    # phase_matrix[k, f] = exp(-i·2π·f_f·tc_k)   shape (N_tc, N_freq)
    phase_matrix_full = jnp.exp(
        -1j * 2.0 * jnp.pi * tc_arr[:, None] * f_full_jax[None, :]
    )

    lats_py   = [float(lats[i])      for i in range(n_det)]
    lons_py   = [float(lons[i])      for i in range(n_det)]
    elevs_py  = [float(elevs[i])     for i in range(n_det)]
    gammas_py = [float(gammas[i])    for i in range(n_det)]
    zetas_py  = [float(zetas[i])     for i in range(n_det)]
    trigs_py  = [float(trigtimes[i]) for i in range(n_det)]
    Ts_py     = [float(T_durs[i])    for i in range(n_det)]
    tc_0_py   = float(tc_0)

    def log_likelihood_full_time_marg(params_no_tc: jnp.ndarray) -> jnp.ndarray:
        """Full time-marginalized log-likelihood (no RB approximation).

        Parameters
        ----------
        params_no_tc : array, shape (12,)
            [ra, dec, logdist, incl, phic, pol, mc, q,
             chi1, chi2, lambda1, lambda2]
        """
        params_13 = jnp.concatenate([
            params_no_tc[:8],
            jnp.array([tc_0_py], dtype=jnp.float64),
            params_no_tc[8:],
        ])

        log_L_tc = jnp.zeros(N_tc, dtype=jnp.float64)

        for i in range(n_det):
            # Full-grid projected waveform at tc_0 (tc variation via phase)
            h_full_i = project_waveform_at_freqs(
                params_13, f_full_jax, template_fn,
                lats_py[i], lons_py[i], elevs_py[i],
                gammas_py[i], zetas_py[i],
                trigs_py[i], Ts_py[i],
            )
            dd_i = jnp.sum(jnp.abs(data_list[i]) ** 2 / sigmasq_list[i])
            hh_i = jnp.sum(jnp.abs(h_full_i) ** 2 / sigmasq_list[i])
            # Strip the tc_0 phase so the cross-term can be computed at any tc via
            # cross(tc_k) = Re[Σ_f conj(d_f)·h_noTc_f/σ²_f · exp(-i2πf·tc_k)]
            # where h_noTc = h(tc_0) · exp(+i2πf·tc_0) removes the tc_0 contribution.
            h_noTc = h_full_i * jnp.exp(1j * 2.0 * jnp.pi * f_full_jax * tc_0_py)
            integrand_noTc = jnp.conj(data_list[i]) * h_noTc / sigmasq_list[i]
            cross_tc = jnp.real(phase_matrix_full @ integrand_noTc)  # (N_tc,)
            TwoDTN_i = TwoDTN_list[i]
            log_L_tc = log_L_tc + (
                -TwoDTN_i * (dd_i - 2.0 * cross_tc + hh_i)
            )

        return jax.scipy.special.logsumexp(log_L_tc) - jnp.log(
            jnp.array(N_tc, dtype=jnp.float64)
        )

    return log_likelihood_full_time_marg


def build_full_likelihood_tc_phi_marg(
    batched_detector,
    fiducial_params: np.ndarray,
    template_fn: Callable,
    tc_grid: np.ndarray,
) -> Callable:
    """Build a time- and phase-marginalized full (non-RB) log-likelihood.

    Uses the full frequency grid with Bessel-function phase marginalisation.

    Parameters
    ----------
    batched_detector : sharpy Detector
    fiducial_params  : array, shape (13,)
    template_fn      : callable
    tc_grid          : array, shape (N_tc,)

    Returns
    -------
    log_likelihood_full_tc_phi_marg : callable
        (params_no_tc_no_phi_11,) → scalar.
    """
    n_det = len(batched_detector.latitude)
    lats     = np.array(batched_detector.latitude)
    lons     = np.array(batched_detector.longitude)
    elevs    = np.array(batched_detector.elevation)
    gammas   = np.array(batched_detector.gamma)
    zetas    = np.array(batched_detector.zeta)
    trigtimes = np.array(batched_detector.trigtime)
    T_durs   = np.array(batched_detector.T)

    f_full_np  = np.array(batched_detector.Frequency[0])
    f_full_jax = jnp.array(f_full_np, dtype=jnp.float64)

    data_list    = [jnp.array(batched_detector.FrequencySeries[i], dtype=jnp.complex128)
                    for i in range(n_det)]
    sigmasq_list = [jnp.array(batched_detector.sigmasq[i], dtype=jnp.float64)
                    for i in range(n_det)]
    TwoDTN_list  = [float(batched_detector.TwoDeltaTOverN[i]) for i in range(n_det)]

    tc_0   = float(fiducial_params[8])
    N_tc   = len(tc_grid)
    tc_arr = jnp.array(tc_grid, dtype=jnp.float64)

    phase_matrix_full = jnp.exp(
        -1j * 2.0 * jnp.pi * tc_arr[:, None] * f_full_jax[None, :]
    )  # (N_tc, N_freq)

    lats_py   = [float(lats[i])      for i in range(n_det)]
    lons_py   = [float(lons[i])      for i in range(n_det)]
    elevs_py  = [float(elevs[i])     for i in range(n_det)]
    gammas_py = [float(gammas[i])    for i in range(n_det)]
    zetas_py  = [float(zetas[i])     for i in range(n_det)]
    trigs_py  = [float(trigtimes[i]) for i in range(n_det)]
    Ts_py     = [float(T_durs[i])    for i in range(n_det)]
    tc_0_py   = float(tc_0)

    def log_likelihood_full_tc_phi_marg(
        params_no_tc_no_phi: jnp.ndarray,
    ) -> jnp.ndarray:
        """Full time+phase-marginalized log-likelihood.

        Parameters
        ----------
        params_no_tc_no_phi : array, shape (11,)
            [ra, dec, logdist, incl, pol, mc, q, chi1, chi2, lambda1, lambda2]
        """
        params_13 = jnp.concatenate([
            params_no_tc_no_phi[:4],                          # ra, dec, logdist, incl
            jnp.array([0.0], dtype=jnp.float64),              # phic = 0
            params_no_tc_no_phi[4:7],                         # pol, mc, q
            jnp.array([tc_0_py], dtype=jnp.float64),          # tc = tc_0
            params_no_tc_no_phi[7:],                          # chi1, chi2, λ1, λ2
        ])

        const     = jnp.float64(0.0)
        W_tc      = jnp.zeros(N_tc, dtype=jnp.complex128)

        for i in range(n_det):
            h_full_i = project_waveform_at_freqs(
                params_13, f_full_jax, template_fn,
                lats_py[i], lons_py[i], elevs_py[i],
                gammas_py[i], zetas_py[i],
                trigs_py[i], Ts_py[i],
            )
            # Strip the tc_0 phase: h_noTc = h(tc_0) * exp(+i2πf·tc_0)
            h_noTc = h_full_i * jnp.exp(1j * 2.0 * jnp.pi * f_full_jax * tc_0_py)
            dd_i   = jnp.sum(jnp.abs(data_list[i]) ** 2 / sigmasq_list[i])
            hh_i   = jnp.sum(jnp.abs(h_noTc) ** 2 / sigmasq_list[i])
            TwoDTN_i = TwoDTN_list[i]
            const  = const + (-TwoDTN_i * (dd_i + hh_i))
            # Accumulate W(tc_k) = Σ_det TwoDTN_det · Σ_f conj(d)·h_noTc/σ² · exp(-i2πf·tc)
            integrand_noTc = jnp.conj(data_list[i]) * h_noTc / sigmasq_list[i]
            W_tc = W_tc + TwoDTN_i * (phase_matrix_full @ integrand_noTc)  # (N_tc,)

        x = 2.0 * jnp.abs(W_tc)
        log_bessel = jnp.log(jax.scipy.special.i0e(x)) + x
        log_L_phi_tc = const + log_bessel  # (N_tc,)

        return jax.scipy.special.logsumexp(log_L_phi_tc) - jnp.log(
            jnp.array(N_tc, dtype=jnp.float64)
        )

    return log_likelihood_full_tc_phi_marg


# ======================================================================
# Signal-capture check (matched filter SNR)
# ======================================================================

def compute_matched_filter_snr(
    batched_detector,
    fiducial_params: np.ndarray,
    template_fn: Callable,
) -> dict:
    """Compute the optimal matched-filter network SNR for a given template.

    The *optimal* (phase-maximised) SNR is:
        SNR_opt² = 4 · df · Σ_f |h(f)|² / S_n(f)

    where the sum is over positive frequencies and df is the frequency
    resolution.  This equals the SNR that would be achieved with perfect
    phase alignment.  For GW170817 with the H1+L1+V1 network and a good
    template, the expected network SNR is ~32.

    Note: A negative *signed* SNR  (Σ conj(d)·h / σ²) at a given parameter
    point does not imply the signal is absent — it merely means the template
    phase is not aligned with the data at those specific parameters (e.g.
    wrong φ_c or ψ).  The optimal SNR above is phase-independent.

    Parameters
    ----------
    batched_detector : sharpy Detector
    fiducial_params : array, shape (13,)
    template_fn : callable

    Returns
    -------
    dict with keys:
        'snr_opt_det'     - list of per-detector optimal SNR
        'snr_opt_network' - network SNR (quadrature sum)
        'snr_signed_det'  - list of signed matched-filter SNR (phase dependent)
    """
    n_det = len(batched_detector.latitude)
    lats = np.array(batched_detector.latitude)
    lons = np.array(batched_detector.longitude)
    elevs = np.array(batched_detector.elevation)
    gammas = np.array(batched_detector.gamma)
    zetas = np.array(batched_detector.zeta)
    trigtimes = np.array(batched_detector.trigtime)
    T_durs = np.array(batched_detector.T)

    snr_opt_det = []
    snr_signed_det = []
    for i in range(n_det):
        f_i = np.array(batched_detector.Frequency[i])
        data_i = np.array(batched_detector.FrequencySeries[i])
        sigmasq_i = np.array(batched_detector.sigmasq[i])
        TwoDTN_i = float(batched_detector.TwoDeltaTOverN[i])

        h_i = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                jnp.array(f_i, dtype=jnp.float64),
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )

        # Optimal (phase-maximised) SNR:
        #   SNR_opt² = 2 · TwoDTN · Σ |h|² / σ²  = (h|h) in SHARPy convention
        hh = float(np.real(np.sum(np.abs(h_i) ** 2 / sigmasq_i)))
        snr_opt_i = float(np.sqrt(max(0.0, 2.0 * TwoDTN_i * hh)))
        snr_opt_det.append(snr_opt_i)

        # Signed cross-correlation (phase dependent — can be negative)
        dh = float(np.real(np.sum(np.conj(data_i) * h_i / sigmasq_i)))
        snr_signed_i = (2.0 * TwoDTN_i * dh) / snr_opt_i if snr_opt_i > 0 else 0.0
        snr_signed_det.append(snr_signed_i)

    snr_opt_network = float(np.sqrt(np.sum(np.array(snr_opt_det) ** 2)))
    return {
        "snr_opt_det": snr_opt_det,
        "snr_opt_network": snr_opt_network,
        "snr_signed_det": snr_signed_det,
    }
