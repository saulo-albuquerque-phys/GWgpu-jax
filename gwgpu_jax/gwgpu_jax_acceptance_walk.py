"""
gwgpu_jax_acceptance_walk.py
============================
Conservative, GW-validated nested-sampling driver for gwgpu_jax.

This module lets you run gwgpu_jax parameter estimation with the **acceptance-walk**
nested-sampling kernel of Prathaban et al. (2025) — the ``bilby``/``dynesty``
constrained random-walk move (the LIGO/Virgo community standard), ported to
GPU inside the ``blackjax-ns`` framework and shown there to recover posteriors
and evidences *statistically identical* to the CPU ``bilby`` pipeline for
aligned-spin binary black holes.

Why this exists
---------------
:class:`~gwgpu_jax.GWgpu_jaxNestedSampler` uses BlackJAX-NS **Nested Slice
Sampling** (Yallup, Kroupa & Handley 2025). NSS is fast but, as a sampling
*kernel*, has not been separately validated on gravitational-wave data. For
results you want to defend to a PE-specialist audience, this driver swaps only
the inner move for the already-GW-validated acceptance-walk kernel while keeping
**your own likelihood, waveform, network, and priors unchanged** — the exact
same :meth:`GWgpu_jaxNestedSampler.loglikelihood_fn` and prior specs are reused,
so the only thing that differs from the NSS run is the sampler.

External dependency (not vendored)
----------------------------------
The acceptance-walk kernel lives in an external, separately-licensed repository
(`mrosep/blackjax_ns_gw <https://github.com/mrosep/blackjax_ns_gw>`_) and is
**not** bundled with gwgpu_jax. Fetch it once with::

    bash scripts/fetch_acceptance_walk_kernel.sh

which sparse-clones just the ``custom_kernels`` package into ``external/`` (kept
out of git via ``.gitignore``). At run time this module locates it via, in order:

1. an already-importable ``custom_kernels`` package;
2. the ``GWJAX_ACCEPTANCE_WALK_SRC`` environment variable (path to the ``src``
   dir that contains ``custom_kernels/``);
3. ``<repo>/external/blackjax_ns_gw/src``.

It pins the same BlackJAX commit gwgpu_jax already uses
(``dedbf11da33eb5ca286f6731e2c51f2b254b953f``), so there is no version conflict.

Citation
--------
If you use this driver, cite:

* Prathaban, Yallup, Alvey, Yang, Templeton & Handley (2025),
  "Gravitational-wave inference at GPU speed: A bilby-like nested sampling
  kernel within blackjax-ns", arXiv:2509.04336 — the acceptance-walk kernel;
* Cabezas et al. (2024), "BlackJAX", arXiv:2402.10797 — the framework;
* Speagle (2020), "dynesty", MNRAS 493, 3132; Ashton et al. (2019), "Bilby",
  ApJS 241, 27 — the acceptance-walk algorithm this kernel reproduces.
"""

from __future__ import annotations

import os
import sys
import time
import importlib
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp

import blackjax.ns as bns

from gwgpu_jax.gwgpu_jax_network_ifos_utils import Network
from gwgpu_jax.gwgpu_jax_samplers import GWgpu_jaxNestedSampler
from gwgpu_jax.gwgpu_jax_prior_definitions import resolve_priors


# ── Result container ─────────────────────────────────────────────────────────

class AcceptanceWalkResult(NamedTuple):
    """Output bundle from :meth:`GWgpu_jaxAcceptanceWalkSampler.run`.

    ``posterior_samples`` are in **physical** parameter space (the unit-cube
    draws are mapped back through the prior transform), so the interface matches
    :class:`~gwgpu_jax.NestedSamplingResult`.
    """
    posterior_samples: dict
    logZ:              float
    logZ_err:          float
    n_iterations:      int
    ess:               float
    live_state:        object
    dead_info:         object


# ── External-kernel discovery ─────────────────────────────────────────────────

_FETCH_HINT = (
    "The acceptance-walk kernel (mrosep/blackjax_ns_gw) is an external "
    "dependency and is not bundled with gwgpu_jax.\n"
    "Fetch it once with:\n\n"
    "    bash scripts/fetch_acceptance_walk_kernel.sh\n\n"
    "or set GWJAX_ACCEPTANCE_WALK_SRC to the 'src' directory that contains the "
    "'custom_kernels' package."
)


def _candidate_src_dirs() -> list[str]:
    """Ordered candidate directories that may contain ``custom_kernels/``."""
    cands: list[str] = []
    env = os.environ.get("GWJAX_ACCEPTANCE_WALK_SRC")
    if env:
        cands.append(env)
    # <repo_root>/external/blackjax_ns_gw/src  (repo_root = parent of this package)
    repo_root = Path(__file__).resolve().parent.parent
    cands.append(str(repo_root / "external" / "blackjax_ns_gw" / "src"))
    return cands


def import_custom_kernels():
    """Import and return the external ``custom_kernels`` module.

    Raises a helpful :class:`ModuleNotFoundError` (with fetch instructions) if
    the kernel package cannot be located.
    """
    try:
        return importlib.import_module("custom_kernels")
    except ModuleNotFoundError:
        pass
    for src in _candidate_src_dirs():
        if os.path.isdir(os.path.join(src, "custom_kernels")):
            if src not in sys.path:
                sys.path.insert(0, src)
            return importlib.import_module("custom_kernels")
    raise ModuleNotFoundError(_FETCH_HINT)


# ── Prior transform (unit cube -> physical) built from gwjax prior specs ───────

def build_prior_transform(prior_specs: dict, param_names: Sequence[str]) -> Callable:
    """Return ``u_dict -> x_dict`` using gwgpu_jax's own inverse-CDF (``icdf``).

    Because the transform is *the same* ``PriorSpec`` objects the NSS sampler
    draws from, the unit-cube prior is identical to the NSS prior — a required
    property for an apples-to-apples comparison.
    """
    specs = [prior_specs[name] for name in param_names]

    def prior_transform_fn(u_dict):
        return {name: spec.icdf(u_dict[name])
                for name, spec in zip(param_names, specs)}

    return prior_transform_fn


def build_periodic_mask(param_names: Sequence[str],
                        periodic_params: Sequence[str]) -> dict:
    """Boolean mask tree marking parameters whose unit coordinate wraps mod 1."""
    periodic = set(periodic_params)
    return {name: (name in periodic) for name in param_names}


# ── Sampler ────────────────────────────────────────────────────────────────────

class GWgpu_jaxAcceptanceWalkSampler:
    """Nested sampling with the GW-validated acceptance-walk kernel.

    The constructor signature mirrors
    :class:`~gwgpu_jax.GWgpu_jaxNestedSampler`; internally it builds one of those
    to reuse its exact ``loglikelihood_fn`` and resolved prior specs, then runs
    the acceptance-walk inner kernel on the unit hypercube instead of NSS.

    Parameters
    ----------
    network, waveform_fn, param_bounds, fixed_params, priors, gmst
        Passed straight through to :class:`GWgpu_jaxNestedSampler` (see its
        docstring). The likelihood and priors are therefore identical to the
        NSS path.
    periodic_params : sequence of str, optional
        Parameters whose *unit-cube* coordinate should wrap (mod 1) during the
        random walk — i.e. genuinely periodic angles. Defaults to
        ``("phi_c", "ra")``; any name not in ``param_bounds`` is ignored.
    """

    def __init__(
        self,
        network:         Network,
        waveform_fn:     Callable,
        param_bounds:    dict,
        fixed_params:    Optional[dict] = None,
        gmst:            Optional[float] = None,
        priors:          Optional[dict] = None,
        periodic_params: Sequence[str] = ("phi_c", "ra"),
    ) -> None:
        # Reuse the NSS sampler purely as the source of the (validated)
        # likelihood closure and the resolved prior specs — nothing about NSS
        # is executed here.
        self._ns = GWgpu_jaxNestedSampler(
            network=network, waveform_fn=waveform_fn,
            param_bounds=param_bounds, fixed_params=fixed_params,
            gmst=gmst, priors=priors,
        )
        self._init_from_parts(
            loglik_fn=self._ns.loglikelihood_fn,      # physical logL
            prior_specs=self._ns._prior_specs,
            param_bounds=param_bounds,
            periodic_params=periodic_params,
        )

    def _init_from_parts(self, loglik_fn, prior_specs, param_bounds,
                         periodic_params) -> None:
        self.param_bounds    = dict(param_bounds)
        self.param_names     = list(self.param_bounds.keys())
        self._prior_specs    = prior_specs
        self._loglik_fn      = loglik_fn
        self.periodic_params = tuple(p for p in periodic_params
                                     if p in self.param_bounds)

    @classmethod
    def from_loglikelihood(
        cls,
        loglikelihood_fn: Callable,
        param_bounds:     dict,
        priors:           Optional[dict] = None,
        periodic_params:  Sequence[str] = ("phi_c", "ra"),
    ) -> "GWgpu_jaxAcceptanceWalkSampler":
        """Build the sampler around an arbitrary physical log-likelihood.

        Use this when the likelihood is not a plain network+waveform Gaussian
        likelihood — e.g. a tc/φ_c-marginalised relative-binning likelihood.
        ``loglikelihood_fn(params)`` must take a dict keyed by ``param_bounds``
        names (physical space) and return a scalar log-likelihood; it must be
        JAX-traceable (jit/vmap-safe). Priors are resolved exactly as in the NSS
        sampler via :func:`resolve_priors`.
        """
        self = cls.__new__(cls)
        self._ns = None
        specs = resolve_priors(dict(param_bounds), dict(priors or {}))
        self._init_from_parts(
            loglik_fn=loglikelihood_fn, prior_specs=specs,
            param_bounds=param_bounds, periodic_params=periodic_params,
        )
        return self

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        rng_key:               jax.Array,
        num_live:              int = 500,
        n_target:              int = 60,
        max_mcmc:              int = 5000,
        num_delete:            int = 1,
        max_iterations:        int = 10000,
        log_dlogz_target:      float = -3.0,
        num_posterior_samples: int = 2000,
        verbose:               bool = False,
    ) -> AcceptanceWalkResult:
        """Run acceptance-walk nested sampling until the live evidence is small.

        Parameters
        ----------
        rng_key : PRNGKey
        num_live : int
            Number of live particles.
        n_target : int
            Target number of *accepted* MCMC steps per new point (bilby's
            ``nact``-style walk-length control). Higher = more decorrelated
            (safer), slower. 60 matches the kernel's default.
        max_mcmc : int
            Hard cap on MCMC proposals per new point.
        num_delete : int
            Live points replaced per NS step. ``1`` is the classical,
            lowest-bias choice; larger batches are faster on GPU.
        max_iterations : int
            Hard cap on NS iterations.
        log_dlogz_target : float
            Stop when ``state.logZ_live - state.logZ < log_dlogz_target``
            (same convention as :meth:`GWgpu_jaxNestedSampler.run`).
        num_posterior_samples : int
        verbose : bool

        Returns
        -------
        AcceptanceWalkResult
        """
        ck = import_custom_kernels()

        prior_transform_fn = build_prior_transform(self._prior_specs, self.param_names)
        mask_tree          = build_periodic_mask(self.param_names, self.periodic_params)

        uc = ck.create_unit_cube_functions(
            physical_loglikelihood_fn=self._loglik_fn,
            prior_transform_fn=prior_transform_fn,
            mask_tree=mask_tree,
        )

        k_init, k_loop, k_post = jax.random.split(rng_key, 3)

        example_params = {name: 0.0 for name in self.param_names}
        particles = ck.init_unit_cube_particles(k_init, example_params, num_live)

        algo = ck.acceptance_walk_sampler(
            logprior_fn      = uc["logprior_fn"],
            loglikelihood_fn = uc["loglikelihood_fn"],
            nlive            = num_live,
            n_target         = n_target,
            max_mcmc         = max_mcmc,
            num_delete       = num_delete,
            stepper_fn       = uc["stepper_fn"],
        )
        state = algo.init(particles)
        step  = jax.jit(algo.step)

        dead: list = []
        i = 0
        if verbose:
            print("JIT-compiling acceptance-walk NS kernel… "
                  "(one-off; first step is slow)", flush=True)
        t0 = time.perf_counter()
        while i < max_iterations:
            k_loop, k_step = jax.random.split(k_loop)
            state, info = step(k_step, state)
            if i == 0 and verbose:
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                    state,
                )
                print(f"  → compile + first step in {time.perf_counter()-t0:.1f} s.",
                      flush=True)
            dead.append(info)

            delta_logZ = float(state.logZ_live - state.logZ)
            if verbose and (i % 100 == 0):
                print(f"iter {i:5d}  logZ={float(state.logZ):+.3f}  "
                      f"logZ_live={float(state.logZ_live):+.3f}  Δ={delta_logZ:+.3f}")
            i += 1
            if delta_logZ < log_dlogz_target:
                break

        # Evidence + posterior: identical post-processing to the NSS path.
        dead_info = bns.utils.finalise(state, dead)

        k_w, k_s = jax.random.split(k_post)
        logw   = bns.utils.log_weights(k_w, dead_info, shape=200)
        logZ_r = jax.scipy.special.logsumexp(logw, axis=0)
        logZ     = float(jnp.mean(logZ_r))
        logZ_err = float(jnp.std(logZ_r))

        # Posterior draws come back in unit-cube coordinates → map to physical.
        post_u = bns.utils.sample(k_s, dead_info, shape=num_posterior_samples)
        post_x = jax.vmap(prior_transform_fn)(post_u)
        posterior_samples = {k: v for k, v in post_x.items()}

        ess = float(bns.utils.ess(k_w, dead_info))

        return AcceptanceWalkResult(
            posterior_samples = posterior_samples,
            logZ              = logZ,
            logZ_err          = logZ_err,
            n_iterations      = i,
            ess               = ess,
            live_state        = state,
            dead_info         = dead_info,
        )
