"""
gwgpu_jax_jim_interface.py
==========================
Adapters that expose the two machine-learning surrogates shipped with
GWgpu_jax,

  * ``mlgw_bbh_jax``  — time-domain BBH surrogate (SEOBNRv5HM, model_4), and
  * ``mlgw_bns_jax``  — frequency-domain BNS surrogate (TEOBResumS),

as waveform models of the `Jim <https://github.com/GW-JAX-Team/Jim>`_
parameter-estimation pipeline (``jimgw >= 0.6``).

Jim's waveform protocol
-----------------------
Since ``jimgw`` 0.6 a waveform is a :class:`ripplegw.interfaces.Waveform`
(``ripplegw >= 0.5``): an object configured once and then called as
``waveform(frequency, params)``, returning ``{"p": h_plus, "c": h_cross}``,
that declares the parameters it reads in ``parameter_names``.  Both adapters
below are :class:`~ripplegw.interfaces.FrequencyDomainWaveform` subclasses, so
they can be handed to ``TransientLikelihoodFD``, ``HeterodynedTransientLikelihoodFD``
and ``Detector.inject_signal`` exactly like ``RippleIMRPhenomD``::

    from gwgpu_jax.gwgpu_jax_jim_interface import MLGWBBHJimWaveform
    waveform   = MLGWBBHJimWaveform(duration=4.0, sampling_frequency=2048.0)
    likelihood = TransientLikelihoodFD(ifos, waveform=waveform, f_min=20., f_max=896.)

They are also registered with ripplegw, so ``ripplegw.waveform("MLGW_BBH",
duration=4.0, sampling_frequency=2048.0)`` and ``ripplegw.waveform("MLGW_BNS")``
work once this module has been imported.

Conventions (identical to the ripplegw models Jim ships with)
-------------------------------------------------------------
  * Parameters use Jim's names: ``M_c`` [M_sun], ``eta``, ``s1_z``, ``s2_z``,
    ``d_L`` [Mpc], ``phase_c`` [rad], ``iota`` [rad]; the BNS model adds
    ``lambda_1``, ``lambda_2``.  Body 1 is the heavier one.
  * The returned polarisations follow the LAL Fourier convention
    ``h̃(f) = ∫ h(t) exp(-2πi f t) dt`` with the **merger at t = 0**.  Jim's
    detectors apply the ``t_c`` (and trigger-time) shift themselves, so
    ``t_c`` is *not* a waveform parameter.
  * Both models scale as ``1 / d_L``, which is declared through
    :class:`~ripplegw.interfaces.DistanceScaledWaveform` so Jim can reuse
    waveform caches across distance-only updates.

Environment
-----------
Jim 0.6 needs Python >= 3.12, ``jax >= 0.8.2`` and ``ripplegw >= 0.5``.  The
adapters only need the surrogates themselves, so nothing else from GWgpu_jax
(its samplers, its pinned blackjax fork) has to be functional in the Jim
environment.  ``mlgw_bbh_jax`` additionally needs ``tensorflow``, ``tf2jax``
and ``joblib`` (the ``mlgw`` extra) to load its weights; ``mlgw_bns_jax`` only
needs ``h5py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

import jax
import jax.numpy as jnp

try:
    from ripplegw.interfaces import DistanceScaledWaveform, FrequencyDomainWaveform
    from ripplegw.registry import register
except ImportError as exc:  # ripplegw < 0.5 (the version GWgpu_jax itself pins)
    raise ImportError(
        "gwgpu_jax.gwgpu_jax_jim_interface needs the waveform protocol of "
        "ripplegw >= 0.5 (installed together with jimgw >= 0.6, Python >= 3.12). "
        "Install Jim first:  pip install 'jimgw>=0.6'"
    ) from exc


__all__ = ["MLGWBBHJimWaveform", "MLGWBNSJimWaveform"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _component_masses(params: Mapping[str, Any]):
    """``(M_c, eta) -> (m1, m2)`` with ``m1 >= m2`` — JAX-traceable."""
    M_c, eta = params["M_c"], params["eta"]
    M = M_c / eta ** 0.6
    # eta can exceed 1/4 by a rounding error after Jim's q -> eta transform.
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    return 0.5 * M * (1.0 + delta), 0.5 * M * (1.0 - delta)


def _is_concrete(x) -> bool:
    return not isinstance(x, jax.core.Tracer)


def _tukey_asymmetric(n: int, alpha_left: float, alpha_right: float):
    """Asymmetric Tukey window — same as ``mlgw_bbh_jax_waveform_generator.
    tukey_asymmetric``, repeated here so it can be built without importing
    TensorFlow (that module loads the mlgw_bbh weights on import)."""
    x = jnp.linspace(0.0, 1.0, n)
    w = jnp.ones_like(x)
    if alpha_left > 0:
        left = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * (x / alpha_left - 0.5)))
        w = jnp.where(x < alpha_left / 2.0, left, w)
    if alpha_right > 0:
        right = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * ((x - 1.0) / alpha_right + 0.5)))
        w = jnp.where(x > 1.0 - alpha_right / 2.0, right, w)
    return w


def _bundled_bbh_model(model):
    """Map ``"model_N"`` to the bundled model index ``N``.

    ``mlgw.GW_generator`` resolves an int against its own ``TD_models`` folder
    but treats a string as a path relative to the working directory, so a bare
    ``"model_4"`` only loads when run from inside ``TD_models``.
    """
    import os

    if isinstance(model, str) and not os.path.isdir(model):
        name = model.removeprefix("model_")
        if name.isdigit():
            return int(name)
    return model


def _normalise_modes(modes):
    """Return ``modes`` as ``None`` (all), ``(2, 2)`` or a list of ``(l, m)``."""
    if modes is None:
        return None
    if isinstance(modes, tuple) and len(modes) == 2 and all(
        isinstance(x, (int, np.integer)) for x in modes
    ):
        modes = [modes]
    modes = [(int(l), int(m)) for l, m in modes]
    return (2, 2) if modes == [(2, 2)] else modes


# ── mlgw_bbh_jax ─────────────────────────────────────────────────────────────

@register("MLGW_BBH", is_tidal=False, is_precessing=False, source_type="cbc")
class MLGWBBHJimWaveform(FrequencyDomainWaveform, DistanceScaledWaveform):
    """``mlgw_bbh_jax`` (SEOBNRv5HM surrogate, aligned spins) as a Jim waveform.

    ``mlgw_bbh`` is a *time-domain* model, while Jim's likelihoods are
    frequency-domain.  The adapter therefore owns a fixed time grid of
    ``duration * sampling_frequency`` samples, evaluates the surrogate on it,
    applies the asymmetric Tukey taper used throughout GWgpu_jax and Fourier
    transforms the result (see :func:`~gwgpu_jax.mlgw_jax.
    mlgw_bbh_jax_waveform_generator.hphc_td_to_fd`).  The grid is
    ``[final_time - duration, final_time)`` with the peak of the (2, 2) mode
    at t = 0, i.e. the convention of ``TimeFrequencyGrid.time_domain_array_mlgw``;
    the transform is referred to t = 0, so the returned spectrum has its
    merger at t = 0 as Jim expects.

    Frequencies requested by Jim are served in one of two ways:

    ``"fft"``
        one rFFT, then a lookup.  Exact, and the fast path, but only defined
        on the grid ``k / duration``.  This is what ``TransientLikelihoodFD``
        and ``inject_signal`` ask for when ``duration`` equals (or is an
        integer multiple of) the data duration.
    ``"dtft"``
        the discrete-time Fourier transform ``dt Σ_n h(t_n) exp(-2πi f t_n)``
        evaluated directly at the requested frequencies.  It coincides with
        the rFFT on the grid and is defined in between, which is what the
        relative-binning likelihood needs for its bin edges.  Costs an
        ``n_freq × n_samples`` complex matrix, so keep ``n_freq`` modest
        (heterodyne bin edges: a few hundred to a few thousand).
    ``"auto"`` (default)
        ``"fft"`` when the frequencies are known at trace time and all lie on
        the grid, otherwise ``"dtft"``.

    Parameters
    ----------
    duration : float
        Length of the internal time grid [s].  Use the duration of the data
        segment (4 s for GW150914-like signals; see the GWgpu_jax notebooks).
        The surrogate is evaluated over the whole grid, so the template
        starts ``duration - final_time`` before the merger and is tapered in
        over the first ``alpha_left / 2`` of it.
    sampling_frequency : float
        Sampling rate of the internal time grid [Hz]; must be at least twice
        the highest frequency Jim asks for.
    modes : None | (l, m) | list of (l, m)
        Modes summed into the polarisations.  ``None`` (default) uses every
        mode of model_4: 22, 21, 32, 33, 43, 44, 55.  ``(2, 2)`` gives the
        dominant mode only.
    final_time : float
        Post-merger length of the grid [s].  The default 0.12 s is the value
        of the GWgpu_jax GW150914 notebooks and covers the ringdown of model_4
        up to a total mass of ~230 M_sun; a tail as short as 0.01 s truncates
        it (mismatch ~6e-3 against SEOBNRv5HM).
    alpha_left, alpha_right : float
        Tukey taper fractions at the start / end of the grid (defaults
        0.2 / 0.001, again those of the GW150914 notebooks).
    frequency_mode : {"auto", "fft", "dtft"}
        See above.
    model : int or str
        Bundled model index (default 4, i.e. ``TD_models/model_4``; the name
        ``"model_4"`` is accepted too) or the path of a model folder.
    f_ref : float
        Unused by the model (aligned spins need no reference frequency); kept
        because Jim's helper transforms read ``waveform.f_ref``.

    Notes
    -----
    * mlgw defines ``h = h_+ + i h_x`` while LAL, ripplegw and Jim's antenna
      patterns use ``h = h_+ - i h_x``; the adapter returns ``-h_x`` so that
      ``iota`` and ``psi`` mean the same as for every other Jim waveform.
    * ``phase_c`` is forwarded as mlgw's reference orbital phase ``phi_0``.
      With higher modes it is not a pure overall phase, so use Jim's
      ``phase_marginalization`` only with ``modes=(2, 2)`` (Jim warns
      otherwise, from ``waveform_metadata["highest_mode"]``).
    * Training range of model_4: mass ratio ``m1/m2`` in [1, 10], i.e.
      ``q = m2/m1 >= 0.1`` in Jim's convention, and ``|s_z| <= 0.9``.  Keep
      the priors inside it — the network is not clipped outside.
    """

    def __init__(
        self,
        duration: float,
        sampling_frequency: float,
        modes=None,
        *,
        final_time: float = 0.12,
        alpha_left: float = 0.2,
        alpha_right: float = 0.001,
        frequency_mode: str = "auto",
        model: int | str = 4,
        f_ref: float = 20.0,
        generator=None,
    ) -> None:
        if frequency_mode not in ("auto", "fft", "dtft"):
            raise ValueError(
                f"frequency_mode must be 'auto', 'fft' or 'dtft', got {frequency_mode!r}."
            )
        if not 0.0 <= final_time < duration:
            raise ValueError(
                f"final_time must lie in [0, duration={duration}), got {final_time}."
            )
        n_samples = int(round(duration * sampling_frequency))
        if n_samples < 2 or abs(n_samples - duration * sampling_frequency) > 1e-6:
            raise ValueError(
                "duration * sampling_frequency must be an integer number of samples."
            )

        if generator is None:
            # Imported here: pulls in TensorFlow / tf2jax to load the weights.
            from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import (
                MLGWBBHGenerator,
            )
            generator = MLGWBBHGenerator(_bundled_bbh_model(model))

        self._generator = generator
        self.duration = float(duration)
        self.sampling_frequency = float(sampling_frequency)
        self.final_time = float(final_time)
        self.alpha_left = float(alpha_left)
        self.alpha_right = float(alpha_right)
        self.frequency_mode = frequency_mode
        self.model = model
        self.f_ref = float(f_ref)
        self.modes = _normalise_modes(modes)

        self.n_samples = n_samples
        self.dt = 1.0 / self.sampling_frequency
        #: time grid [s], merger (peak of the 22 mode) at t = 0
        self.times = jnp.arange(n_samples) * self.dt - self.duration + self.final_time
        self._window = _tukey_asymmetric(n_samples, self.alpha_left, self.alpha_right)

        used = self.modes if self.modes is not None else generator.available_modes
        used = [used] if used == (2, 2) else used
        # Instance-level copy: the highest m depends on the modes selected.
        self.waveform_metadata = {
            **type(self).waveform_metadata,
            "highest_mode": max(int(m) for _, m in used),
        }

    # ── ripplegw / Jim protocol ──────────────────────────────────────────────

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("M_c", "eta", "s1_z", "s2_z", "d_L", "phase_c", "iota")

    def time_domain(self, params: Mapping[str, Any]):
        """Untapered ``(h_plus, h_cross)`` on :attr:`times`, LAL sign convention."""
        m1, m2 = _component_masses(params)
        theta = jnp.stack([
            jnp.asarray(x, dtype=self.times.dtype) for x in (
                m1, m2, params["s1_z"], params["s2_z"],
                params["d_L"], params["iota"], params["phase_c"],
            )
        ])
        hp, hc = self._generator.generate(theta, self.times, modes=self.modes)
        # mlgw builds h = h_+ + i h_x, LAL (and Jim's antenna patterns) use
        # h = h_+ - i h_x: flip the cross polarisation, as the GWgpu_jax
        # GW150914 notebooks do.
        return hp, -hc

    def __call__(self, frequency, /, params: Mapping[str, Any]):
        """Evaluate the plus (``"p"``) and cross (``"c"``) polarisations.

        ``frequency`` [Hz] is any 1-D array; see the class docstring for how
        on-grid and off-grid frequencies are handled.
        """
        hp_td, hc_td = self.time_domain(params)
        h_td = jnp.stack([hp_td, hc_td]) * self._window          # (2, N)

        if self._use_fft(frequency):
            index = jnp.asarray(
                np.rint(np.asarray(frequency) * self.duration).astype(np.int64)
            )
            f_grid = index / self.duration
            # rfft takes the first sample as t = 0; the grid starts at
            # final_time - duration, and exp(-2πi f (final_time - duration))
            # = exp(-2πi f final_time) on the grid f = k / duration.
            shift = jnp.exp(-2j * jnp.pi * f_grid * self.final_time)
            h_fd = jnp.fft.rfft(h_td, axis=-1)[:, index] * (self.dt * shift)
        else:
            frequency = jnp.asarray(frequency)
            kernel = jnp.exp(-2j * jnp.pi * frequency[:, None] * self.times[None, :])
            h_fd = (h_td.astype(kernel.dtype) @ kernel.T) * self.dt  # (2, n_freq)
        return {"p": h_fd[0], "c": h_fd[1]}

    # ── internals ────────────────────────────────────────────────────────────

    def _use_fft(self, frequency) -> bool:
        """Decide between the rFFT lookup and the direct DTFT (at trace time)."""
        if self.frequency_mode == "dtft":
            return False
        if not _is_concrete(frequency):
            if self.frequency_mode == "fft":
                raise ValueError(
                    "frequency_mode='fft' needs frequencies known at trace time; "
                    "close over the frequency array instead of passing it as a "
                    "jit/vmap argument, or use frequency_mode='dtft'."
                )
            return False
        freq = np.asarray(frequency, dtype=np.float64)
        nyquist = 0.5 * self.sampling_frequency
        if freq.size and (freq.min() < 0.0 or freq.max() > nyquist * (1 + 1e-12)):
            raise ValueError(
                f"Requested frequencies span [{freq.min()}, {freq.max()}] Hz, outside "
                f"[0, {nyquist}] Hz covered by sampling_frequency={self.sampling_frequency}."
            )
        k = freq * self.duration
        on_grid = bool(np.all(np.abs(k - np.rint(k)) <= 1e-6))
        if self.frequency_mode == "fft" and not on_grid:
            raise ValueError(
                "frequency_mode='fft' needs frequencies on the grid k / duration "
                f"(duration={self.duration} s). Use 'auto' or 'dtft' for other grids."
            )
        return on_grid

    def __repr__(self) -> str:
        modes = "all" if self.modes is None else self.modes
        return (
            f"MLGWBBHJimWaveform(model={self.model!r}, modes={modes}, "
            f"duration={self.duration}, sampling_frequency={self.sampling_frequency})"
        )


# ── mlgw_bns_jax ─────────────────────────────────────────────────────────────

@register(
    "MLGW_BNS", is_tidal=True, is_precessing=False, highest_mode=2,
    source_type="cbc",
)
class MLGWBNSJimWaveform(FrequencyDomainWaveform, DistanceScaledWaveform):
    """``mlgw_bns_jax`` (TEOBResumS surrogate, (2, 2) mode, tides) as a Jim waveform.

    ``mlgw_bns`` is natively frequency-domain and can be evaluated at arbitrary
    frequencies, so this is a thin re-parameterisation: it works unchanged with
    ``TransientLikelihoodFD``, with the relative-binning
    ``HeterodynedTransientLikelihoodFD`` (the usual choice for long BNS
    signals) and with ``inject_signal``.

    Parameters
    ----------
    model_path : str or Path, optional
        Exported model (``.h5``).  Defaults to the one bundled with GWgpu_jax.
    f_ref : float
        Unused by the model; kept because Jim's helper transforms read
        ``waveform.f_ref``.

    Notes
    -----
    * Parameters: ``M_c``, ``eta``, ``s1_z``, ``s2_z``, ``lambda_1``,
      ``lambda_2``, ``d_L``, ``phase_c``, ``iota`` — the same set and meaning
      as ``RippleIMRPhenomD_NRTidalv2`` with individual tidal deformabilities.
    * ``phase_c`` enters as the overall factor ``exp(2i phase_c)`` (the
      convention of ripplegw's (2, 2)-mode models), so Jim's
      ``phase_marginalization`` is exact.
    * The surrogate already places the merger at t = 0 to within a few ms
      (checked against ``IMRPhenomD_NRTidalv2``: the two differ by a ~3 ms
      time offset for a GW170817-like source), so no time shift is applied
      and a ``t_c`` prior centred on the trigger time is appropriate.
    * Validity (see ``mlgw_bns_jax_waveform_generator``): ``lambda`` in
      [0, 5000], ``|s_z|`` up to ~0.3, and frequencies inside the surrogate's
      band, 3.6 – 2926 Hz for a total mass of 2.8 M_sun and scaling as
      ``2.8 / M``.  Outside the band the output is a finite but meaningless
      extrapolation, so keep the likelihood's ``f_min`` / ``f_max`` inside it.
    """

    def __init__(
        self,
        model_path=None,
        *,
        f_ref: float = 20.0,
        generator=None,
    ) -> None:
        if generator is None:
            from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import (
                MLGWBNSGenerator,
            )
            generator = MLGWBNSGenerator(model_path)
        self._generator = generator
        self.f_ref = float(f_ref)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c", "eta", "s1_z", "s2_z", "lambda_1", "lambda_2",
            "d_L", "phase_c", "iota",
        )

    def __call__(self, frequency, /, params: Mapping[str, Any]):
        """Evaluate the plus (``"p"``) and cross (``"c"``) polarisations."""
        frequency = jnp.asarray(frequency, dtype=jnp.float64)
        m1, m2 = _component_masses(params)
        intrinsic = jnp.stack([
            jnp.asarray(x, dtype=jnp.float64) for x in (
                m1 / m2, params["lambda_1"], params["lambda_2"],
                params["s1_z"], params["s2_z"],
            )
        ])
        hp, hc = self._generator.predict(
            intrinsic,
            frequency,
            jnp.asarray(m1 + m2, dtype=jnp.float64),
            jnp.asarray(params["d_L"], dtype=jnp.float64),
            jnp.asarray(params["iota"], dtype=jnp.float64),
        )
        phase = 2.0 * jnp.asarray(params["phase_c"], dtype=jnp.float64)
        rotation = jnp.cos(phase) + 1j * jnp.sin(phase)
        return {"p": hp * rotation, "c": hc * rotation}

    def __repr__(self) -> str:
        return "MLGWBNSJimWaveform()"
