"""
mlgw_bbh_jax_waveform_generator.py
====================================
Wrapper around the cloned mlgw_bbh_jax (new_training branch) that exposes a
clean, JIT/vmap/grad-compatible interface for generating BBH time-domain
waveforms without requiring the package to be installed.

The sub-repo lives at:
    gwgpu_jax/mlgw_jax/mlgw_bbh_jax/

Active model: **model_4** (SEOBNRv5HM, Phys. Rev. D 108, 124035 (2023))
    Modes:  22  21  32  33  43  44  55
    Range:  mass ratio q = m1/m2 ∈ [1, 10]
            aligned spins χ₁, χ₂ ∈ [−0.9, 0.9]

Parameter conventions
---------------------
    m1          : mass of the heavier BH             [M_sun],  m1 >= m2
    m2          : mass of the lighter BH             [M_sun]
    s1z         : aligned (z) spin of BH 1           (dimensionless, |s| ≤ 0.9)
    s2z         : aligned (z) spin of BH 2           (dimensionless, |s| ≤ 0.9)
    d_l         : luminosity distance                [Mpc]       default 1 Mpc
    inclination : inclination angle                  [rad]       default 0 (face-on)
    phi_0       : reference orbital phase            [rad]       default 0

Time grid
---------
    The time grid is in physical seconds.  By convention, t = 0 marks the
    peak (merger) and the signal starts at negative times.
    Typical grid: ``jnp.linspace(-8., 0.005, 8192)``

Note (recovery)
---------------
This file was reconstructed on 2026-05-27 after an accidental working-tree
reset wiped the original. Logic was rebuilt from the cached .pyc bytecode
(class/method signatures, attribute names, docstrings) plus the original
file header. The user should verify behaviour against the upstream model
before relying on the JIT/grad paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# Suppress TF/XLA startup noise (must be set before TF import)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Inject the cloned sub-repo into sys.path (no install needed) ─────────────
_REPO_DIR = Path(__file__).resolve().parent / "mlgw_bbh_jax"

if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from mlgw.GW_generator import GW_generator  # noqa: E402  (path injection above)


# ── Time-domain → frequency-domain helpers ───────────────────────────────────
#
# mlgw_bbh is a time-domain surrogate; turning it into the complex FD
# (h₊, h×) that the GWgpu_jax likelihood expects needs two things the bare
# ``rfft`` does not provide, both lifted from the upstream
# ``mlgw.GW_FD_generator``:
#
#   1. an evaluation grid that *crosses zero* so the peak + ringdown are
#      sampled (see ``TimeFrequencyGrid.time_domain_array_mlgw``); and
#   2. an **asymmetric Tukey taper** applied in the time domain before the
#      rFFT, to kill the spectral leakage from the abrupt inspiral turn-on
#      (long ``alpha_left``) and the truncated ringdown edge (tiny
#      ``alpha_right``).
#
# Skipping the taper is what leaves the FD template distorted enough to
# rail the coalescence time and masses on real-data PE.

def tukey_asymmetric(n: int, alpha_left: float = 0.1, alpha_right: float = 0.001):
    """Asymmetric Tukey (tapered-cosine) window, JAX-traceable.

    ``alpha_left`` / ``alpha_right`` are the fractions of the window cosine-
    tapered at the start / end. Matches ``GW_FD_generator.tukey_asymmetric``.

    Parameters
    ----------
    n : int
        Window length (number of time samples).
    alpha_left, alpha_right : float
        Taper fractions in ``[0, 1]``. Defaults 0.1 / 0.001 follow upstream.

    Returns
    -------
    w : jnp.ndarray, shape (n,)
    """
    x = jnp.linspace(0.0, 1.0, n)
    w = jnp.ones_like(x)
    left  = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * (x / alpha_left - 0.5)))
    right = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * ((x - 1.0) / alpha_right + 0.5)))
    w = jnp.where(x < alpha_left  / 2.0, left,  w)
    w = jnp.where(x > 1.0 - alpha_right / 2.0, right, w)
    return w


def hphc_td_to_fd(
    hp_td, hc_td, dt,
    alpha_left:  float = 0.1,
    alpha_right: float = 0.001,
):
    """Window (h₊, h×) in the time domain and rFFT to the GWgpu_jax FD convention.

    Applies an :func:`tukey_asymmetric` taper then ``ĥ(f) = rfft(h)·dt``
    (the convention used everywhere else in GWgpu_jax, e.g.
    ``Interferometer.load_data``). Equivalent to
    ``GW_FD_generator.nfft_jax`` (which divides by the sampling rate).

    Parameters
    ----------
    hp_td, hc_td : jnp.ndarray, shape (N,)
        Real time-domain polarisations from ``MLGWBBHGenerator``, evaluated
        on a grid that crosses zero (``grid.time_domain_array_mlgw``).
    dt : float
        Time step ``1 / sampling_rate`` [s].
    alpha_left, alpha_right : float
        Tukey taper fractions; forwarded to :func:`tukey_asymmetric`.

    Returns
    -------
    hp_fd, hc_fd : complex jnp.ndarray, shape (N//2 + 1,)
    """
    w = tukey_asymmetric(hp_td.shape[-1], alpha_left, alpha_right)
    hp_fd = jnp.fft.rfft(hp_td * w) * dt
    hc_fd = jnp.fft.rfft(hc_td * w) * dt
    return hp_fd, hc_fd


class MLGWBBHGenerator:
    """Time-domain BBH waveform generator backed by MLGW-JAX (model_4, SEOBNRv5HM).

    The model (all 7 modes) is loaded once on construction.  JIT-compiled
    callables for the two most common mode selections (all modes and the
    leading 2-2 mode) are cached as ``_jit_all`` and ``_jit_22`` so that
    repeated calls with the same time grid do not trigger recompilation.

    Parameters
    ----------
    model : str or int, optional
        Identifier passed through to :class:`mlgw.GW_generator.GW_generator`.
        Defaults to ``"model_4"`` (SEOBNRv5HM).

    Attributes
    ----------
    _gen : mlgw.GW_generator.GW_generator
        The underlying JAX-export model.  ``self._gen.get_WF(theta, t_grid, modes=...)``
        is the primitive forward call used by everything below.
    _jit_all : Callable
        ``jax.jit(lambda theta, t: self._gen.get_WF(theta, t, modes=None))``.
    _jit_22  : Callable
        ``jax.jit(lambda theta, t: self._gen.get_WF(theta, t, modes=(2, 2)))``.
    """

    def __init__(self, model="model_4") -> None:
        self._gen     = GW_generator(model)
        self._jit_all = jax.jit(lambda theta, t: self._gen.get_WF(theta, t, modes=None))
        self._jit_22  = jax.jit(lambda theta, t: self._gen.get_WF(theta, t, modes=(2, 2)))

    # ── lightweight introspection ────────────────────────────────────────────

    @property
    def generate(self):
        """Raw :func:`GW_generator.get_WF` — suitable for ``jax.jit`` / ``jax.grad`` / ``jax.vmap``.

        Signature::

            hp, hc = generate(theta, t_grid, modes=None)

        ``theta`` may be 1-D ``(D,)`` (single event) or 2-D ``(N, D)``
        (batch), with D ∈ {3, 4, 5, 6, 7}.  For the most common case use
        D = 4: ``[m1, m2, s1z, s2z]``.

        Example — gradient of h+ w.r.t. component masses::

            dhp_dm = jax.jacobian(
                lambda th: gen.generate(th, t_grid, modes=(2,2))[0]
            )(jnp.array([30., 10., 0.5, -0.3]))
        """
        return self._gen.get_WF

    @property
    def available_modes(self):
        """Return the list of (l, m) mode tuples available in the loaded model."""
        return self._gen.list_modes()

    # ── single-event interface ───────────────────────────────────────────────

    def get_hphc(
        self,
        t_grid,
        m1,
        m2,
        s1z         = 0.0,
        s2z         = 0.0,
        d_l         = 1.0,
        inclination = 0.0,
        phi_0       = 0.0,
        modes       = None,
        jit:  bool  = True,
    ):
        """Return (h+, h×) time-domain polarizations for a single BBH event.

        Parameters
        ----------
        t_grid      : 1-D time array [s].  Zero marks the merger; a typical
                      range is ``jnp.linspace(-8., 0.005, 8192)``.
        m1          : mass of the heavier BH [M_sun].
        m2          : mass of the lighter BH [M_sun].
        s1z         : dimensionless aligned spin of BH 1 (|s| ≤ 0.9).
        s2z         : dimensionless aligned spin of BH 2 (|s| ≤ 0.9).
        d_l         : luminosity distance [Mpc].
        inclination : inclination angle [rad].
        phi_0       : reference orbital phase [rad].
        modes       : ``None`` (all available modes), a single tuple such as
                      ``(2, 2)``, or a list of tuples.  Using ``None`` or
                      ``(2, 2)`` enables the pre-cached JIT path.
        jit         : use JIT compilation (default ``True``).  The first call
                      per unique ``t_grid`` length triggers compilation;
                      subsequent calls are fast.

        Returns
        -------
        hp, hc : real JAX arrays of shape ``(len(t_grid),)``.
        """
        theta = jnp.asarray([
            float(m1), float(m2),
            float(s1z), float(s2z),
            float(d_l), float(inclination), float(phi_0),
        ])

        if jit and modes is None:
            return self._jit_all(theta, t_grid)
        if jit and modes == (2, 2):
            return self._jit_22(theta, t_grid)
        if jit:
            # Non-cached mode selection — compile on demand.
            return jax.jit(lambda th, tg: self._gen.get_WF(th, tg, modes=modes))(theta, t_grid)
        return self._gen.get_WF(theta, t_grid, modes=modes)

    # ── batched interface ────────────────────────────────────────────────────

    def get_hphc_batch(
        self,
        t_grid,
        theta_batch,
        d_l         = 1.0,
        inclination = 0.0,
        phi_0       = 0.0,
        modes       = None,
        jit:  bool  = True,
    ):
        """Generate waveforms for a batch of intrinsic parameters.

        Parameters
        ----------
        t_grid       : 1-D time array [s].
        theta_batch  : array of shape ``(N, 4)`` with columns
                       ``[m1, m2, s1z, s2z]`` (M_sun / dimensionless).
        d_l          : luminosity distance [Mpc] — scalar or shape ``(N,)``.
        inclination  : inclination angle [rad] — scalar or shape ``(N,)``.
        phi_0        : reference orbital phase [rad] — scalar or shape ``(N,)``.
        modes        : ``None`` (all modes), ``(2, 2)``, or list of tuples.
        jit          : use JIT compilation (default ``True``).

        Returns
        -------
        hp_batch, hc_batch : real JAX arrays of shape ``(N, len(t_grid))``.
        """
        theta_batch = jnp.asarray(theta_batch)
        N = theta_batch.shape[0]
        ones = jnp.ones((N, 1))

        def col(x):
            arr = jnp.asarray(x)
            if arr.ndim == 0:
                return ones * float(arr)
            return arr.reshape(N, 1)

        theta_full = jnp.concatenate([
            theta_batch,         # (N, 4) = [m1, m2, s1z, s2z]
            col(d_l),            # (N, 1)
            col(inclination),    # (N, 1)
            col(phi_0),          # (N, 1)
        ], axis=1)               # (N, 7)

        if jit and modes is None:
            return self._jit_all(theta_full, t_grid)
        if jit and modes == (2, 2):
            return self._jit_22(theta_full, t_grid)
        if jit:
            return jax.jit(lambda th, tg: self._gen.get_WF(th, tg, modes=modes))(theta_full, t_grid)
        return self._gen.get_WF(theta_full, t_grid, modes=modes)
