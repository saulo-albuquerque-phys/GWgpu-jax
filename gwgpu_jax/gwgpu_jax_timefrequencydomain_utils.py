"""
gwgpu_jax_timefrequencydomain_utils.py
===================================
Utility class that builds the time and frequency grids used throughout GWgpu_jax.

Four main arrays are produced from three scalars (``duration``, ``sampling_rate``,
``initial_time``) and two optional frequency limits (``f_min``, ``f_max``):

  ``time_domain_array``
      Standard time samples in physical seconds.
      Shape ``(N,)`` where ``N = duration × sampling_rate``.
      Runs from ``initial_time`` to ``initial_time + duration - dt``.

  ``frequency_domain_array``
      One-sided (positive) frequency bins from an FFT of ``time_domain_array``.
      Shape ``(N//2 + 1,)``, running from 0 Hz to the Nyquist frequency
      ``sampling_rate / 2``.  Frequency resolution is ``df = 1 / duration``.
      Always the full grid — never truncated — so FFT/IFFT operations work
      without modification.

  ``frequency_mask``
      Boolean array of shape ``(N//2 + 1,)`` that is ``True`` for bins
      satisfying ``f_min ≤ f ≤ f_max``.  Use it to restrict matched-filter
      inner products, SNR integrals, or PSD estimations to the analysis band::

          inner = 4 * df * jnp.sum((h_tilde * s_tilde.conj()).real[grid.frequency_mask]
                                   / psd[grid.frequency_mask])

  ``time_domain_array_mlgw``
      Time grid in the convention required by ``mlgw_bbh_jax``:
      t = 0 is placed at the merger (where mlgw_bbh puts the waveform peak).
      The grid runs from ``mlgw_final_time − duration`` to
      ``mlgw_final_time − dt`` in steps of ``dt`` — i.e. it is
      ``time_domain_array`` shifted so the merger sits ``mlgw_final_time``
      seconds before the end of the segment.

      **Why it must cross zero.** mlgw_bbh generates the waveform on its own
      native reduced-time grid (peak at 0, ringdown out to ~``times[-1]·M``)
      and interpolates onto this array with the amplitude held at *zero*
      outside the native range (``jnp.interp(..., left=0., right=0.)``).
      If the grid stops at ``−dt`` (the old ``mlgw_final_time = 0`` choice)
      the merger peak and the *entire* post-merger ringdown fall on the
      ``right=0`` side and are silently dropped, leaving a ringdown-less,
      discontinuous template whose rFFT is badly distorted. This is invisible
      to inject↔recover (the distorted template matches itself) but wrecks
      real-data PE — the coalescence time and masses rail at the prior edges.
      A small positive ``mlgw_final_time`` (default 0.01 s, after the
      upstream ``GW_FD_generator``) makes the grid straddle zero so the peak
      and ringdown are captured. Pair it with an asymmetric Tukey taper on
      the time-domain (h₊, h×) before the rFFT to suppress the inspiral
      turn-on / ringdown-edge leakage.

Usage
-----
>>> from gwgpu_jax.gwgpu_jax_timefrequencydomain_utils import TimeFrequencyGrid
>>> grid = TimeFrequencyGrid(duration=8.0, sampling_rate=2048.0, initial_time=0.0,
...                          f_min=20.0, f_max=1024.0)
>>> grid.time_domain_array        # shape (16384,)
>>> grid.frequency_domain_array   # shape (8193,)  — full FFT grid
>>> grid.frequency_mask           # shape (8193,)  — True for 20–1024 Hz
>>> grid.time_domain_array_mlgw   # shape (16384,), in [0.01−8, 0.01−dt)
"""

from __future__ import annotations

import jax.numpy as jnp


class TimeFrequencyGrid:
    """Pre-computed time and frequency grids for a GW data segment.

    Parameters
    ----------
    duration : float
        Segment duration [s].  Must be positive.
    sampling_rate : float
        Sampling rate [Hz].  Must be positive.  ``N = duration × sampling_rate``
        must be an integer (within floating-point tolerance).
    initial_time : float
        GPS or relative start time of the segment [s].  Defaults to 0.
    f_min : float, optional
        Lower frequency boundary of the analysis band [Hz].  Defaults to 0
        (no lower cut-off).  Used to build ``frequency_mask``.
    f_max : float or None, optional
        Upper frequency boundary of the analysis band [Hz].  Defaults to
        ``None``, which means the Nyquist frequency ``sampling_rate / 2``.
        Used to build ``frequency_mask``.
    mlgw_final_time : float, optional
        Seconds of post-merger time to include at the end of
        ``time_domain_array_mlgw`` [s].  Must satisfy ``0 <= mlgw_final_time
        < duration``.  Defaults to 0.01 s (matching the upstream
        ``GW_FD_generator``), enough to capture the peak and ringdown of a
        stellar-mass BBH; raise it for very high total masses (longer
        ringdown).  Has no effect on any array other than
        ``time_domain_array_mlgw``.

    Attributes
    ----------
    duration : float
    sampling_rate : float
    initial_time : float
    f_min : float
    f_max : float
    dt : float
        Time step ``1 / sampling_rate`` [s].
    df : float
        Frequency resolution ``1 / duration`` [Hz].
    n_samples : int
        Number of time samples ``N = duration × sampling_rate``.
    time_domain_array : jax.Array, shape (N,)
        Physical time samples [s], starting at ``initial_time``.
    frequency_domain_array : jax.Array, shape (N//2 + 1,)
        Full one-sided FFT frequency bins [Hz].  Never truncated.
    frequency_mask : jax.Array of bool, shape (N//2 + 1,)
        True for bins satisfying ``f_min ≤ f ≤ f_max``.
    time_domain_array_mlgw : jax.Array, shape (N,)
        Time samples relative to merger (t=0 at the peak) [s], running
        ``[mlgw_final_time − duration, mlgw_final_time − dt]`` so the peak
        and ringdown are sampled. Used as the ``t_grid`` argument for
        ``MLGWBBHGenerator``.
    mlgw_final_time : float
        Post-merger tail length baked into ``time_domain_array_mlgw`` [s].
    """

    def __init__(
        self,
        duration:      float,
        sampling_rate: float,
        initial_time:  float = 0.0,
        f_min:         float = 0.0,
        f_max:         float | None = None,
        mlgw_final_time: float = 0.01,
    ) -> None:
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
        if f_min < 0:
            raise ValueError(f"f_min must be non-negative, got {f_min}")
        if not (0.0 <= mlgw_final_time < duration):
            raise ValueError(
                f"mlgw_final_time must lie in [0, duration={duration}), "
                f"got {mlgw_final_time}"
            )

        n_samples = int(round(duration * sampling_rate))
        if abs(n_samples - duration * sampling_rate) > 1e-6:
            raise ValueError(
                f"duration × sampling_rate = {duration * sampling_rate} is not an integer. "
                "Choose values whose product is exactly an integer."
            )

        self.duration      = float(duration)
        self.sampling_rate = float(sampling_rate)
        self.initial_time  = float(initial_time)
        self.n_samples     = n_samples
        self.dt            = 1.0 / self.sampling_rate
        self.df            = 1.0 / self.duration
        self.f_min         = float(f_min)
        self.f_max         = float(f_max) if f_max is not None else self.sampling_rate / 2.0
        self.mlgw_final_time = float(mlgw_final_time)

        # ── time_domain_array: [t0, t0+dt, …, t0+(N-1)·dt] ──────────────────
        self.time_domain_array: jnp.ndarray = (
            self.initial_time + jnp.arange(n_samples) * self.dt
        )

        # ── frequency_domain_array: one-sided FFT bins [0, df, …, fs/2] ─────
        self.frequency_domain_array: jnp.ndarray = jnp.fft.rfftfreq(
            n_samples, d=self.dt
        )

        # ── frequency_mask: True for bins within [f_min, f_max] ──────────────
        self.frequency_mask: jnp.ndarray = (
            (self.frequency_domain_array >= self.f_min)
            & (self.frequency_domain_array <= self.f_max)
        )

        # ── time_domain_array_mlgw: merger at t=0, ``mlgw_final_time`` of
        #    post-merger time kept at the tail so the peak + ringdown are
        #    actually sampled (see the class docstring for why this matters).
        # t_mlgw[i] = i·dt − duration + mlgw_final_time, i.e. the grid runs
        #             [mlgw_final_time − duration, mlgw_final_time − dt].
        # Computed relative to the merger (not initial_time) to avoid float
        # cancellation when initial_time is a large GPS timestamp.
        self.time_domain_array_mlgw: jnp.ndarray = (
            jnp.arange(n_samples) * self.dt - self.duration + self.mlgw_final_time
        )

    def __repr__(self) -> str:
        n_band = int(self.frequency_mask.sum())
        return (
            f"TimeFrequencyGrid("
            f"duration={self.duration}s, "
            f"sampling_rate={self.sampling_rate}Hz, "
            f"initial_time={self.initial_time}s, "
            f"N={self.n_samples}, "
            f"df={self.df:.6f}Hz, "
            f"f_min={self.f_min:.1f}Hz, "
            f"f_max={self.f_max:.1f}Hz, "
            f"n_band_bins={n_band})"
        )
