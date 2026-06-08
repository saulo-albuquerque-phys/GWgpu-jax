"""
gwgpu_jax_custom_waveform.py
=========================
Bilby-style user-supplied waveform wrapper.

If you already have a JAX function that returns ``(h_plus, h_cross)`` for
a binary in your favourite parametrisation, :class:`CustomWaveform` is
the smallest possible adapter that plugs it into the GWgpu_jax workflow
(``Network.project_waveform``, ``Interferometer.inject_signal``,
``make_likelihood_fn``, ``GWgpu_jaxNestedSampler``).

Conventions
-----------
Your function must have the signature

    h_plus, h_cross = waveform_function(grid_array, **params)

where

  * ``grid_array`` is a 1-D ``jax.Array`` — the FD frequency axis for
    ``domain="fd"`` or the TD time axis for ``domain="td"``.
  * ``**params`` are the physical parameters; their names are inferred
    from the function signature or set explicitly via ``parameter_names``.

It must be **pure** JAX (no Python-level branching on traced values) so
``jit`` / ``vmap`` / ``grad`` work.

Quickstart
----------
>>> import jax.numpy as jnp, gwgpu_jax
>>>
>>> def my_waveform(freqs, mass_1, mass_2, distance, inclination):
...     # toy: TaylorF2-ish amplitude scaling, no real physics
...     Mc = (mass_1 * mass_2)**0.6 / (mass_1 + mass_2)**0.2
...     amp = (Mc / 30.0)**(5/6) * (100.0 / distance) * jnp.cos(inclination)
...     hp = amp * freqs**(-7/6) * jnp.exp(-1j * jnp.pi * freqs)
...     hc = 1j * amp * freqs**(-7/6) * jnp.exp(-1j * jnp.pi * freqs)
...     return hp, hc
>>>
>>> wf = gwgpu_jax.CustomWaveform(my_waveform, domain="fd",
...                           default_values={"distance": 100., "inclination": 0.})
>>> # Use with the sampler — CustomWaveform is itself a (params, freqs) callable
>>> sampler = gwgpu_jax.GWgpu_jaxNestedSampler(
...     network=net, waveform_fn=wf,
...     param_bounds={"mass_1": (5., 50.), "mass_2": (5., 50.),
...                   "ra": (0., 6.28), "dec": (-1.5, 1.5),
...                   "psi": (0., 3.14), "tc": (-0.05, 0.05)},
... )

Parameter conversion
--------------------
If you want to sample in a different space from your function's
signature (e.g. sample ``chirp_mass``/``mass_ratio`` but the function
takes ``mass_1``/``mass_2``), pass a pure JAX
``parameter_conversion(params_dict) -> params_dict`` callable. It runs
inside the jit/vmap region, so it must be JAX-traceable too.
"""

from __future__ import annotations

import inspect
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from gwgpu_jax.gwgpu_jax_timefrequencydomain_utils import TimeFrequencyGrid


class CustomWaveform:
    """Wrap a user-supplied JAX waveform function for use throughout GWgpu_jax.

    Parameters
    ----------
    waveform_function : Callable
        ``hp, hc = waveform_function(grid_array, **params)``. Must be
        JAX-traceable (jit / vmap / grad-safe).
    domain : ``"fd"`` or ``"td"``
        Output domain. FD plugs directly into the sampler and projection
        layer; TD is supported by :meth:`get_hphc` for offline use and by
        the projection layer only after an explicit rFFT.
    parameter_names : sequence of str, optional
        Names of the physical parameters the function expects. If
        omitted, they are inferred from
        ``inspect.signature(waveform_function)`` (first argument is
        always taken to be the grid array).
    parameter_conversion : Callable, optional
        ``params_dict_in -> params_dict_out``. Runs before the function
        call; useful for sampling in a different parameter space than
        the function signature (e.g. (chirp_mass, mass_ratio) → (m1, m2)).
        Must be pure JAX.
    default_values : dict, optional
        Values applied to any parameter not present in the call. Lets
        you sample over a subset while keeping the rest fixed.
    name : str
        Human-readable identifier, used in ``__repr__``.

    Attributes
    ----------
    waveform_function : Callable
    domain : str
    parameter_names : tuple of str
    parameter_conversion : Callable or None
    default_values : dict
    name : str
    """

    def __init__(
        self,
        waveform_function:    Callable,
        domain:               str = "fd",
        parameter_names:      Optional[Sequence[str]] = None,
        parameter_conversion: Optional[Callable] = None,
        default_values:       Optional[dict] = None,
        name:                 str = "custom",
    ) -> None:
        if domain not in ("fd", "td"):
            raise ValueError(f"domain must be 'fd' or 'td', got {domain!r}")
        if not callable(waveform_function):
            raise TypeError("waveform_function must be callable")
        if parameter_conversion is not None and not callable(parameter_conversion):
            raise TypeError("parameter_conversion must be callable or None")

        self.waveform_function    = waveform_function
        self.domain               = domain
        self.parameter_conversion = parameter_conversion
        self.default_values       = dict(default_values or {})
        self.name                 = name

        if parameter_names is None:
            try:
                sig = inspect.signature(waveform_function)
                # Skip the first positional (grid) and any self/cls.
                names = []
                for i, p in enumerate(sig.parameters.values()):
                    if i == 0 or p.name in ("self", "cls"):
                        continue
                    if p.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue
                    names.append(p.name)
                parameter_names = names
            except (TypeError, ValueError):
                parameter_names = []
        self.parameter_names = tuple(parameter_names)

    # ── grid resolution ───────────────────────────────────────────────────────

    def _grid_array(self, grid) -> jnp.ndarray:
        """Return the 1-D array matching the wrapper's domain."""
        if isinstance(grid, TimeFrequencyGrid):
            return (grid.frequency_domain_array if self.domain == "fd"
                    else grid.time_domain_array)
        return grid

    def _prepare_kwargs(self, params: dict) -> dict:
        """Merge defaults, run parameter_conversion, and filter to the
        function's signature so unrelated keys (e.g. projection params
        like ``ra, dec, psi`` provided by the sampler) don't reach the
        waveform call."""
        full = {**self.default_values, **params}
        if self.parameter_conversion is not None:
            full = self.parameter_conversion(full)
        if self.parameter_names:
            return {k: full[k] for k in self.parameter_names if k in full}
        return full

    # ── single-event evaluation ───────────────────────────────────────────────

    def get_hphc(self, grid, **params):
        """Single-event waveform: ``hp, hc = wf(grid, **params)``.

        ``grid`` may be a :class:`TimeFrequencyGrid` (in which case the
        correct 1-D array is selected automatically) or a raw 1-D
        JAX/NumPy array.

        Any parameter missing from ``params`` is filled from
        :attr:`default_values`; keys outside :attr:`parameter_names` are
        ignored so this can be called with a full sampler particle dict.
        """
        grid_arr = self._grid_array(grid)
        return self.waveform_function(grid_arr, **self._prepare_kwargs(params))

    # ── batch evaluation (vmap over a dict of arrays) ─────────────────────────

    def get_hphc_batch(self, grid, params_batch: dict, jit: bool = True):
        """Batched evaluation via :func:`jax.vmap`.

        Parameters
        ----------
        grid : TimeFrequencyGrid or 1-D array
            Shared across all events.
        params_batch : dict[str, array]
            Mapping ``param_name -> array of shape (N,)``. All arrays
            must share the leading dimension.
        jit : bool
            Wrap the vmapped call in :func:`jax.jit` (default ``True``).
        """
        grid_arr  = self._grid_array(grid)
        wf        = self.waveform_function
        prepare   = self._prepare_kwargs

        def _one(p):
            return wf(grid_arr, **prepare(p))

        fn = jax.vmap(_one)
        if jit:
            fn = jax.jit(fn)
        return fn(params_batch)

    # ── sampler-compatible call signature ─────────────────────────────────────

    def __call__(self, params: dict, grid_array):
        """``waveform_fn(params, grid_array) -> (hp, hc)``.

        Matches the contract expected by
        :class:`~gwgpu_jax.gwgpu_jax_samplers.GWgpu_jaxNestedSampler`; you can pass
        a :class:`CustomWaveform` instance directly as
        ``waveform_fn=...``. Keys in ``params`` that the waveform
        function does not declare in its signature (e.g. projection
        params ``ra, dec, psi, tc``) are filtered out automatically.
        """
        return self.waveform_function(grid_array, **self._prepare_kwargs(params))

    # ── alias ────────────────────────────────────────────────────────────────

    def as_waveform_fn(self) -> Callable:
        """Return the ``(params, grid_array) -> (hp, hc)`` callable.

        Identical to ``self.__call__``; exposed for parity with the
        ``build_*_waveform_fn`` factories in
        :mod:`gwgpu_jax.gwgpu_jax_samplers`.
        """
        return self.__call__

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n = self.waveform_function.__name__ if hasattr(
            self.waveform_function, "__name__"
        ) else "<fn>"
        return (
            f"CustomWaveform(name={self.name!r}, domain={self.domain!r}, "
            f"fn={n}, params={list(self.parameter_names)})"
        )
