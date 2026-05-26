"""
mlgw_bns_jax_waveform_generator.py
====================================
Wrapper around the cloned mlgw_bns_jax (jax-core branch) that exposes a
clean, JIT/vmap-compatible interface for generating BNS frequency-domain
waveforms without requiring the package to be installed.

The sub-repo lives at:
  gwjax/mlgw_jax/mlgw_bns_jax/

The model weights are stored in:
  gwjax/mlgw_jax/mlgw_bns_jax/mlgw_bns_jax_model.h5

Parameter conventions
---------------------
  q          : mass ratio m1/m2 >= 1                 (dimensionless)
  lambda_1   : tidal deformability of heavier star   (dimensionless, [0, 5000])
  lambda_2   : tidal deformability of lighter star   (dimensionless, [0, 5000])
  chi_1      : aligned spin of heavier star          (dimensionless, |χ| ≤ 0.3)
  chi_2      : aligned spin of lighter star          (dimensionless, |χ| ≤ 0.3)
  total_mass : total binary mass                     [M_sun]
  distance   : luminosity distance                   [Mpc]
  inclination: inclination angle                     [rad]  (0 = face-on)

Usage
-----
>>> from gwjax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator
>>> gen = MLGWBNSGenerator()
>>> import jax.numpy as jnp
>>> freqs = jnp.linspace(20., 1024., 2048)
>>> hp, hc = gen.get_hphc(freqs, q=1.2, lambda_1=300., lambda_2=300.,
...                        chi_1=0.05, chi_2=0.05,
...                        total_mass=2.8, distance=100., inclination=0.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# ── x64 is required by jax_import_n_predict ──────────────────────────────────
jax.config.update("jax_enable_x64", True)

# ── Inject the cloned sub-repo into sys.path (no install needed) ─────────────
_REPO_DIR  = Path(__file__).parent / "mlgw_bns_jax"
_MODEL_H5  = _REPO_DIR / "mlgw_bns_jax_model.h5"

if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from jax_import_n_predict import load_predict  # noqa: E402  (path injection above)


class MLGWBNSGenerator:
    """Frequency-domain BNS waveform generator backed by mlgw_bns JAX.

    The model is loaded once on construction and JIT-compiled on the first
    call.  Subsequent calls (same frequency-array shape) are fast.

    The underlying ``predict`` function is also exposed directly so it can
    be wrapped with ``jax.jit`` or ``jax.vmap`` by the caller.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path is not None else _MODEL_H5
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Make sure you have cloned the jax-core branch of mlgw_bns_jax."
            )
        self._predict = load_predict(str(path))
        self._predict_jit = jax.jit(self._predict)

    # ── Core interface ────────────────────────────────────────────────────────

    @property
    def predict(self):
        """Raw JAX predict function — suitable for jit/vmap wrapping."""
        return self._predict

    @property
    def predict_jit(self):
        """JIT-compiled version of predict."""
        return self._predict_jit

    def get_hphc(
        self,
        freqs:      jnp.ndarray,
        q:          float | jnp.ndarray,
        lambda_1:   float | jnp.ndarray,
        lambda_2:   float | jnp.ndarray,
        chi_1:      float | jnp.ndarray = 0.0,
        chi_2:      float | jnp.ndarray = 0.0,
        total_mass: float | jnp.ndarray = 2.8,
        distance:   float | jnp.ndarray = 100.0,
        inclination: float | jnp.ndarray = 0.0,
        jit: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (h+, hx) complex frequency-domain polarizations.

        Parameters
        ----------
        freqs       : 1-D frequency array [Hz], between ~10 and ~2048 Hz.
        q           : mass ratio m1/m2 ≥ 1.
        lambda_1    : tidal deformability of the heavier star.
        lambda_2    : tidal deformability of the lighter star.
        chi_1       : aligned spin of the heavier star (|χ| ≤ 0.3).
        chi_2       : aligned spin of the lighter star (|χ| ≤ 0.3).
        total_mass  : total binary mass [M_sun].
        distance    : luminosity distance [Mpc].
        inclination : inclination angle [rad].
        jit         : use the JIT-compiled predict (default True).

        Returns
        -------
        hp, hc : complex JAX arrays of shape (len(freqs),).
        """
        params = jnp.array([
            float(q),
            float(lambda_1),
            float(lambda_2),
            float(chi_1),
            float(chi_2),
        ], dtype=jnp.float64)

        fn = self._predict_jit if jit else self._predict
        return fn(
            params,
            freqs,
            total_mass   = jnp.array(total_mass,   dtype=jnp.float64),
            distance_mpc = jnp.array(distance,      dtype=jnp.float64),
            inclination  = jnp.array(inclination,   dtype=jnp.float64),
        )

    def get_hphc_batch(
        self,
        freqs:       jnp.ndarray,
        params_batch: jnp.ndarray,
        total_mass:  float | jnp.ndarray = 2.8,
        distance:    float | jnp.ndarray = 100.0,
        inclination: float | jnp.ndarray = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate waveforms for a batch of intrinsic parameters via vmap.

        Parameters
        ----------
        freqs         : 1-D frequency array [Hz].
        params_batch  : array of shape (N, 5) with columns
                        [q, lambda_1, lambda_2, chi_1, chi_2].
        total_mass    : scalar total mass [M_sun] (same for all in batch).
        distance      : scalar luminosity distance [Mpc].
        inclination   : scalar inclination [rad].

        Returns
        -------
        hp_batch, hc_batch : arrays of shape (N, len(freqs)).
        """
        batched = jax.jit(
            jax.vmap(self._predict, in_axes=(0, None, None, None, None))
        )
        return batched(
            params_batch.astype(jnp.float64),
            freqs,
            jnp.array(total_mass,   dtype=jnp.float64),
            jnp.array(distance,      dtype=jnp.float64),
            jnp.array(inclination,   dtype=jnp.float64),
        )

