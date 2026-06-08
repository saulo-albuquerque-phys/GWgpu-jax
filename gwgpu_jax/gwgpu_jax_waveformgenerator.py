"""
gwgpu_jax_waveformgenerator.py
==========================
Unified waveform generator that wraps all three GWgpu_jax backends under a single
``get_hphc`` / ``get_hphc_batch`` interface.

Supported waveform choices
--------------------------
  ``"ripplegw"``      — frequency-domain BBH via IMRPhenom approximants
  ``"mlgw_bbh_jax"``  — time-domain BBH via MLGW-JAX (SEOBNRv5HM, model_4)
  ``"mlgw_bns_jax"``  — frequency-domain BNS via mlgw_bns JAX

Common interface
----------------
  ``get_hphc(grid, m1, m2, chi_1=0, chi_2=0, distance=100, inclination=0, **kwargs)``

      grid       : a ``TimeFrequencyGrid`` instance **or** a raw 1-D JAX array.
                   When a ``TimeFrequencyGrid`` is supplied the correct array
                   is selected automatically:
                     • FD backends ("ripplegw", "mlgw_bns_jax") →
                       ``grid.frequency_domain_array``
                     • TD backend  ("mlgw_bbh_jax") →
                       ``grid.time_domain_array_mlgw``
                   A raw array is passed through unchanged (legacy behaviour).
      m1, m2     : component masses [M_sun], m1 >= m2.
      chi_1      : dimensionless aligned spin of the primary object.
      chi_2      : dimensionless aligned spin of the secondary object.
      distance   : luminosity distance [Mpc].
      inclination: inclination angle [rad].

  ``get_hphc_batch(grid, theta_batch, distance=100, inclination=0, **kwargs)``

      grid        : ``TimeFrequencyGrid`` or raw 1-D array (same rules as above).
      theta_batch : JAX array of shape ``(N, 4)`` with columns
                    ``[m1, m2, chi_1, chi_2]`` for all backends, **or**
                    shape ``(N, 6)`` with columns
                    ``[m1, m2, lambda_1, lambda_2, chi_1, chi_2]``
                    (accepted only by the ``"mlgw_bns_jax"`` backend).
      distance    : scalar luminosity distance [Mpc] applied to all events.
      inclination : scalar inclination [rad] applied to all events.

Backend-specific keyword arguments
-----------------------------------
  ripplegw:
      approximant : str   — "IMRPhenomD" (default), "IMRPhenomXAS",
                            "IMRPhenomD_NRTidalv2", "IMRPhenomPv2".
      f_ref       : float — reference frequency [Hz] (default 20.0).
      tc          : float — time of coalescence (default 0.0).
      phi_c       : float — phase of coalescence (default 0.0).

  mlgw_bns_jax:
      lambda_1    : float — tidal deformability of the primary (default 0.0).
      lambda_2    : float — tidal deformability of the secondary (default 0.0).

  mlgw_bbh_jax:
      phi_0       : float — reference orbital phase [rad] (default 0.0).
      modes       : None | tuple — mode selection (default None = all modes).
      jit         : bool  — use JIT compilation (default True).

Usage
-----
>>> from gwgpu_jax.gwgpu_jax_waveformgenerator import GWgpu_jaxWaveformGenerator
>>> import jax.numpy as jnp
>>>
>>> from gwgpu_jax.gwgpu_jax_timefrequencydomain_utils import TimeFrequencyGrid
>>> tfg = TimeFrequencyGrid(duration=8.0, sampling_rate=2048.0, initial_time=0.0)
>>>
>>> # --- ripplegw (FD BBH) — TimeFrequencyGrid supplies frequency_domain_array ---
>>> gen = GWgpu_jaxWaveformGenerator("ripplegw")
>>> hp, hc = gen.get_hphc(tfg, m1=30., m2=10., chi_1=0.3, chi_2=0.,
...                        distance=400., inclination=0.4)
>>>
>>> # --- mlgw_bbh_jax (TD BBH) — TimeFrequencyGrid supplies time_domain_array_mlgw ---
>>> gen = GWgpu_jaxWaveformGenerator("mlgw_bbh_jax")
>>> hp, hc = gen.get_hphc(tfg, m1=30., m2=10., chi_1=0.5, chi_2=-0.3,
...                        distance=400., inclination=0.4)
>>>
>>> # --- mlgw_bns_jax (FD BNS) — TimeFrequencyGrid supplies frequency_domain_array ---
>>> gen = GWgpu_jaxWaveformGenerator("mlgw_bns_jax")
>>> hp, hc = gen.get_hphc(tfg, m1=1.4, m2=1.2,
...                        distance=100., inclination=0.,
...                        lambda_1=300., lambda_2=300.)
>>>
>>> # Raw JAX arrays are still accepted (legacy) ---
>>> import jax.numpy as jnp
>>> freqs = jnp.linspace(20., 1024., 2048)
>>> hp, hc = gen.get_hphc(freqs, m1=1.4, m2=1.2, distance=100., inclination=0.)
"""

from __future__ import annotations

from typing import Literal, Union

import jax
import jax.numpy as jnp

from gwgpu_jax.gwgpu_jax_timefrequencydomain_utils import TimeFrequencyGrid

# ── ripplegw waveforms ───────────────────────────────────────────────────────
# Note: the (m1, m2) → (Mc, eta) helper used to live at ``ripplegw.ms_to_Mc_eta``
# but its import path moved between 0.0.9 and 0.0.10, so we inline the math
# below rather than depend on a specific upstream layout.
from ripplegw.waveforms import (
    IMRPhenomD,
    IMRPhenomXAS,
    IMRPhenomD_NRTidalv2,
    IMRPhenomPv2,
)

# ── GWgpu_jax sub-generators (lazy imports handled inside __init__) ───────────────
from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator
from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator


_RIPPLE_APPROXIMANTS: dict[str, callable] = {
    "IMRPhenomD":         IMRPhenomD.gen_IMRPhenomD_hphc,
    "IMRPhenomXAS":       IMRPhenomXAS.gen_IMRPhenomXAS_hphc,
    "IMRPhenomD_NRTidalv2": IMRPhenomD_NRTidalv2.gen_IMRPhenomD_NRTidalv2_hphc,
    "IMRPhenomPv2":       IMRPhenomPv2.gen_IMRPhenomPv2,
}

WaveformChoice = Literal["ripplegw", "mlgw_bbh_jax", "mlgw_bns_jax"]


class GWgpu_jaxWaveformGenerator:
    """Unified front-end for the three GWgpu_jax waveform backends.

    Parameters
    ----------
    waveform : str
        One of ``"ripplegw"``, ``"mlgw_bbh_jax"``, ``"mlgw_bns_jax"``.
    approximant : str
        For ``"ripplegw"`` only: IMRPhenom approximant name.
        Valid choices: ``"IMRPhenomD"`` (default), ``"IMRPhenomXAS"``,
        ``"IMRPhenomD_NRTidalv2"``, ``"IMRPhenomPv2"``.
    """

    def __init__(
        self,
        waveform: WaveformChoice,
        approximant: str = "IMRPhenomD",
    ) -> None:
        if waveform not in {"ripplegw", "mlgw_bbh_jax", "mlgw_bns_jax"}:
            raise ValueError(
                f"Unknown waveform '{waveform}'. "
                "Choose from 'ripplegw', 'mlgw_bbh_jax', 'mlgw_bns_jax'."
            )
        self.waveform = waveform

        if waveform == "ripplegw":
            if approximant not in _RIPPLE_APPROXIMANTS:
                raise ValueError(
                    f"Unknown approximant '{approximant}'. "
                    f"Choose from {list(_RIPPLE_APPROXIMANTS)}."
                )
            self.approximant = approximant
            self._ripple_fn = _RIPPLE_APPROXIMANTS[approximant]

        elif waveform == "mlgw_bbh_jax":
            self._bbh_gen = MLGWBBHGenerator()

        elif waveform == "mlgw_bns_jax":
            self._bns_gen = MLGWBNSGenerator()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve_grid(self, grid: Union[TimeFrequencyGrid, jnp.ndarray]) -> jnp.ndarray:
        """Return the correct 1-D JAX array for the active backend.

        If *grid* is a :class:`TimeFrequencyGrid`:
          - FD backends ("ripplegw", "mlgw_bns_jax") → ``frequency_domain_array``
          - TD backend  ("mlgw_bbh_jax")             → ``time_domain_array_mlgw``

        If *grid* is already a JAX / NumPy array it is returned unchanged.
        """
        if isinstance(grid, TimeFrequencyGrid):
            if self.waveform == "mlgw_bbh_jax":
                return grid.time_domain_array_mlgw
            return grid.frequency_domain_array
        return grid

    @staticmethod
    def _to_Mc_eta(m1, m2):
        """Convert component masses to (Mchirp, eta) — JAX-traceable."""
        m1, m2 = jnp.asarray(m1), jnp.asarray(m2)
        Mc  = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        eta = m1 * m2 / (m1 + m2) ** 2
        return Mc, eta

    # ── single-event interface ────────────────────────────────────────────────

    def get_hphc(
        self,
        grid:        Union[TimeFrequencyGrid, jnp.ndarray],
        m1:          float | jnp.ndarray,
        m2:          float | jnp.ndarray,
        chi_1:       float | jnp.ndarray = 0.0,
        chi_2:       float | jnp.ndarray = 0.0,
        distance:    float | jnp.ndarray = 100.0,
        inclination: float | jnp.ndarray = 0.0,
        # ripplegw extra
        f_ref: float = 20.0,
        tc:    float = 0.0,
        phi_c: float = 0.0,
        # mlgw_bns_jax extra
        lambda_1: float | jnp.ndarray = 0.0,
        lambda_2: float | jnp.ndarray = 0.0,
        # mlgw_bbh_jax extra
        phi_0: float | jnp.ndarray = 0.0,
        modes=None,
        jit: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(h+, h×)`` for a single event.

        Parameters
        ----------
        grid        : ``TimeFrequencyGrid`` (recommended) or raw 1-D JAX array.
                      FD backends receive ``frequency_domain_array``;
                      mlgw_bbh_jax receives ``time_domain_array_mlgw``.
        m1, m2      : component masses [M_sun], m1 >= m2.
        chi_1       : aligned spin of the primary (dimensionless).
        chi_2       : aligned spin of the secondary (dimensionless).
        distance    : luminosity distance [Mpc].
        inclination : inclination angle [rad].
        f_ref       : (ripplegw) reference frequency [Hz].
        tc          : (ripplegw) time of coalescence.
        phi_c       : (ripplegw) phase of coalescence.
        lambda_1    : (mlgw_bns_jax) primary tidal deformability.
        lambda_2    : (mlgw_bns_jax) secondary tidal deformability.
        phi_0       : (mlgw_bbh_jax) reference orbital phase [rad].
        modes       : (mlgw_bbh_jax) mode selection — None = all modes.
        jit         : use JIT compilation.

        Returns
        -------
        hp, hc : JAX arrays.  Complex for FD backends, real for mlgw_bbh_jax.
        """
        grid = self._resolve_grid(grid)

        if self.waveform == "ripplegw":
            Mc, eta = self._to_Mc_eta(m1, m2)
            theta = jnp.array([
                float(Mc), float(eta),
                float(chi_1), float(chi_2),
                float(distance), float(tc), float(phi_c),
                float(inclination),
            ])
            fn = jax.jit(self._ripple_fn) if jit else self._ripple_fn
            return fn(grid, theta, f_ref)

        if self.waveform == "mlgw_bbh_jax":
            return self._bbh_gen.get_hphc(
                grid, m1=m1, m2=m2, s1z=chi_1, s2z=chi_2,
                d_l=distance, inclination=inclination, phi_0=phi_0,
                modes=modes, jit=jit,
            )

        # mlgw_bns_jax
        q = float(m1) / float(m2)
        total_mass = float(m1) + float(m2)
        return self._bns_gen.get_hphc(
            grid,
            q=q, lambda_1=lambda_1, lambda_2=lambda_2,
            chi_1=chi_1, chi_2=chi_2,
            total_mass=total_mass, distance=distance, inclination=inclination,
            jit=jit,
        )

    # ── batch interface ───────────────────────────────────────────────────────

    def get_hphc_batch(
        self,
        grid:         Union[TimeFrequencyGrid, jnp.ndarray],
        theta_batch:  jnp.ndarray,
        distance:     float | jnp.ndarray = 100.0,
        inclination:  float | jnp.ndarray = 0.0,
        # ripplegw extra
        f_ref: float = 20.0,
        tc:    float = 0.0,
        phi_c: float = 0.0,
        # mlgw_bbh_jax extra
        phi_0: float | jnp.ndarray = 0.0,
        modes=None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate waveforms for a batch of intrinsic parameters.

        Parameters
        ----------
        grid         : ``TimeFrequencyGrid`` (recommended) or raw 1-D JAX array.
        theta_batch  : JAX array of shape ``(N, 4)`` with columns
                       ``[m1, m2, chi_1, chi_2]`` for all backends,
                       **or** shape ``(N, 6)`` with columns
                       ``[m1, m2, lambda_1, lambda_2, chi_1, chi_2]``
                       (``"mlgw_bns_jax"`` only).
        distance     : scalar luminosity distance [Mpc] for all events.
        inclination  : scalar inclination [rad] for all events.
        f_ref        : (ripplegw) reference frequency [Hz].
        tc           : (ripplegw) time of coalescence.
        phi_c        : (ripplegw) phase of coalescence.
        phi_0        : (mlgw_bbh_jax) reference orbital phase [rad].
        modes        : (mlgw_bbh_jax) mode selection.

        Returns
        -------
        hp_batch, hc_batch : JAX arrays of shape ``(N, len(grid))``.
        """
        grid = self._resolve_grid(grid)

        if self.waveform == "ripplegw":
            # Build (N, 8) theta matrix: [Mc, eta, chi1, chi2, D, tc, phic, iota]
            m1 = theta_batch[:, 0]
            m2 = theta_batch[:, 1]
            chi_1 = theta_batch[:, 2]
            chi_2 = theta_batch[:, 3]
            Mc  = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
            eta = m1 * m2 / (m1 + m2) ** 2
            N = theta_batch.shape[0]
            ones = jnp.ones(N)
            theta_full = jnp.stack([
                Mc, eta,
                chi_1, chi_2,
                ones * float(distance), ones * float(tc), ones * float(phi_c),
                ones * float(inclination),
            ], axis=1)  # (N, 8)
            fn = jax.jit(jax.vmap(self._ripple_fn, in_axes=(None, 0, None)))
            return fn(grid, theta_full, f_ref)

        if self.waveform == "mlgw_bbh_jax":
            # theta_batch: (N, 4) = [m1, m2, chi_1, chi_2]
            return self._bbh_gen.get_hphc_batch(
                grid, theta_batch,
                d_l=distance, inclination=inclination, phi_0=phi_0, modes=modes,
            )

        # mlgw_bns_jax — theta_batch: (N,4) or (N,6)
        ncols = theta_batch.shape[1]
        if ncols == 6:
            # [m1, m2, lambda_1, lambda_2, chi_1, chi_2]
            m1   = theta_batch[:, 0]
            m2   = theta_batch[:, 1]
            lam1 = theta_batch[:, 2]
            lam2 = theta_batch[:, 3]
            chi1 = theta_batch[:, 4]
            chi2 = theta_batch[:, 5]
        elif ncols == 4:
            # [m1, m2, chi_1, chi_2] — tidal params default to 0
            m1   = theta_batch[:, 0]
            m2   = theta_batch[:, 1]
            chi1 = theta_batch[:, 2]
            chi2 = theta_batch[:, 3]
            lam1 = jnp.zeros_like(m1)
            lam2 = jnp.zeros_like(m1)
        else:
            raise ValueError(
                f"mlgw_bns_jax theta_batch must have 4 or 6 columns, got {ncols}."
            )

        q          = m1 / m2
        total_mass = m1 + m2
        # Build (N, 5) intrinsic params: [q, lambda_1, lambda_2, chi_1, chi_2]
        params_bns = jnp.stack([q, lam1, lam2, chi1, chi2], axis=1).astype(jnp.float64)
        total_mass = total_mass.astype(jnp.float64)
        # vmap over both params and total_mass (which varies per event);
        # grid, distance, and inclination are broadcast scalars.
        batched = jax.jit(
            jax.vmap(self._bns_gen.predict, in_axes=(0, None, 0, None, None))
        )
        return batched(
            params_bns,
            grid,
            total_mass,
            jnp.array(distance,    dtype=jnp.float64),
            jnp.array(inclination, dtype=jnp.float64),
        )
