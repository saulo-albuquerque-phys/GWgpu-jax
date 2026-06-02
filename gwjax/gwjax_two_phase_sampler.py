"""
gwjax_two_phase_sampler.py
===========================
Two-phase nested-sampling driver on top of :class:`GWjaxNestedSampler`.

Phase 1 — *bulk exploration*
    Run NSS with a large ``num_delete`` (many particles replaced per
    iteration) until the live-evidence contribution

        Δ_logZ  =  state.logZ_live - state.logZ

    drops below ``phase1_delta_logz_threshold``. This phase contracts
    the prior volume cheaply: each NS iteration replaces a *batch* of
    particles via vmap, so wall-clock per iteration is roughly
    independent of ``num_delete`` while the evidence integral advances
    ``num_delete`` × faster.

Phase 2 — *accurate tail*
    Rebuild the algorithm with ``num_delete = 1`` (or any small value)
    and keep iterating from the same :class:`NSState` until
    ``Δ_logZ < log_dlogz_target``. This phase is Skilling's classical
    NS with negligible per-step bias and is responsible for the final,
    high-accuracy slice of the evidence integral and the bulk of the
    posterior mass.

The dead-particle history from phase 1 and phase 2 is concatenated
inside ``blackjax.ns.utils.finalise``, so the returned evidence and
posterior samples are computed against the *full* run.

Quickstart
----------
>>> import jax, gwjax
>>> sampler = gwjax.GWjaxTwoPhaseNestedSampler(
...     network       = net,
...     waveform_fn   = waveform_fn,
...     param_bounds  = {...},
...     fixed_params  = {...},
... )
>>> result = sampler.run_two_phase(
...     rng_key                      = jax.random.PRNGKey(0),
...     num_live                     = 500,
...     phase1_num_delete            = 25,    # 5 % of live; ~25x faster phase 1
...     phase1_delta_logz_threshold  = -1.0,  # switch when live evidence ~37 % of total
...     phase2_num_delete            = 1,     # classical Skilling
...     log_dlogz_target             = -3.0,  # final convergence
... )
>>> result.logZ, result.posterior_samples["m1"].shape
"""

from __future__ import annotations

import time
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

import blackjax.ns as bns

from gwjax.gwjax_samplers import GWjaxNestedSampler


# ── Result container ─────────────────────────────────────────────────────────

class TwoPhaseNestedSamplingResult(NamedTuple):
    """Output bundle from :meth:`GWjaxTwoPhaseNestedSampler.run_two_phase`."""
    posterior_samples:  dict
    logZ:               float
    logZ_err:           float
    phase1_iterations:  int
    phase2_iterations:  int
    n_iterations:       int    # phase1 + phase2
    ess:                float
    live_state:         object  # final NSState
    dead_info:          object  # combined NSInfo (phase 1 + phase 2 + final live)


# ── Sampler ───────────────────────────────────────────────────────────────────

class GWjaxTwoPhaseNestedSampler(GWjaxNestedSampler):
    """Two-phase variant of :class:`~gwjax.gwjax_samplers.GWjaxNestedSampler`.

    Inherits the constructor (network, waveform_fn, param_bounds,
    fixed_params, gmst) verbatim from the base class; the only new
    capability is :meth:`run_two_phase`, which orchestrates a
    high-``num_delete`` bulk phase followed by a low-``num_delete``
    accuracy phase.

    See the module docstring for the motivation and convergence
    behaviour.
    """

    # ── two-phase driver ─────────────────────────────────────────────────────

    def run_two_phase(
        self,
        rng_key:                       jax.Array,
        *,
        num_live:                      int            = 500,
        num_inner_steps:               Optional[int]  = None,
        phase1_num_inner_steps:        Optional[int]  = None,
        phase2_num_inner_steps:        Optional[int]  = None,
        # ── phase 1: aggressive batch deletion ──────────────────────────────
        phase1_num_delete:             int            = 25,
        phase1_delta_logz_threshold:   float          = -1.0,
        phase1_max_iterations:         int            = 1000,
        # ── phase 2: accurate tail ──────────────────────────────────────────
        phase2_num_delete:             int            = 1,
        phase2_max_iterations:         int            = 5000,
        log_dlogz_target:              float          = -3.0,
        # ── output ──────────────────────────────────────────────────────────
        num_posterior_samples:         int            = 2000,
        verbose:                       bool           = False,
    ) -> TwoPhaseNestedSamplingResult:
        """Run NS in two phases: bulk exploration → accurate tail.

        Parameters
        ----------
        rng_key : PRNGKey
            JAX random key seeding initial live points and both kernels.
        num_live : int
            Number of live particles (constant across both phases).
        num_inner_steps : int, optional
            Slice-sampling steps per NS iteration. Defaults to ``5·d``
            where ``d = len(self.param_bounds)``. Used for both phases
            unless overridden per phase below.
        phase1_num_inner_steps, phase2_num_inner_steps : int, optional
            Per-phase slice-step budgets; each defaults to
            ``num_inner_steps``. Typical use is a cheaper bulk phase and a
            more thorough tail, e.g. ``phase1_num_inner_steps=40,
            phase2_num_inner_steps=300`` to better decorrelate near the
            peak (helps narrow degeneracies). Note phase 2 runs many
            single-delete iterations, so raising its budget dominates the
            wall-clock; lowering phase 1 recovers little since it is
            vmap-amortised over ``phase1_num_delete``.
        phase1_num_delete : int
            Particles replaced per phase-1 step. Use ``num_live // 20``
            to ``num_live // 50`` for a strong speed-up with controlled
            bias; the live pool must stay much larger than this each
            iteration.
        phase1_delta_logz_threshold : float
            Switch from phase 1 to phase 2 when
            ``state.logZ_live - state.logZ < phase1_delta_logz_threshold``.
            ``-1.0`` means *switch when the live points still hold about
            37 % of the total evidence*; ``-2.0`` means ~14 %; ``0.0``
            means *switch immediately after phase 1 starts to converge*.
        phase1_max_iterations : int
            Hard cap on phase-1 iterations. Even if the threshold isn't
            crossed, we transition to phase 2.
        phase2_num_delete : int
            Particles replaced per phase-2 step. ``1`` is the classical
            unbiased choice; small values up to ``5`` are fine.
        phase2_max_iterations : int
            Hard cap on phase-2 iterations.
        log_dlogz_target : float
            Phase-2 termination tolerance. The loop stops when
            ``state.logZ_live - state.logZ < log_dlogz_target``.
        num_posterior_samples : int
            Size of the unweighted posterior sample returned in the result.
        verbose : bool
            Print progress (phase, iter, logZ, logZ_live, Δ) every 100 NS
            iterations and at phase transitions.

        Returns
        -------
        TwoPhaseNestedSamplingResult

        Notes
        -----
        Both phases share the *same* live pool. When the kernel is
        rebuilt for phase 2, the state's ``inner_kernel_params``
        (slice-direction covariance) is preserved — both phases use the
        default ``compute_covariance_from_particles`` adaptation
        function, so the parameter shape is identical.

        Phase 1 and phase 2 NSInfo records have different ``num_delete``
        leading axes (``phase1_num_delete`` vs ``phase2_num_delete``);
        they concatenate cleanly along axis 0 in
        :func:`blackjax.ns.utils.finalise`.
        """
        # Validate
        if phase1_num_delete < 1:
            raise ValueError("phase1_num_delete must be >= 1.")
        if phase2_num_delete < 1:
            raise ValueError("phase2_num_delete must be >= 1.")
        if phase1_num_delete > num_live // 2:
            raise ValueError(
                f"phase1_num_delete ({phase1_num_delete}) cannot exceed "
                f"num_live//2 ({num_live // 2}); the live pool would "
                "collapse in a single step."
            )

        d = len(self.param_bounds)
        if num_inner_steps is None:
            num_inner_steps = max(5 * d, 10)
        # Per-phase override of the slice-step budget. Phase 1 (batch delete)
        # is somewhat self-averaging and can use fewer steps; phase 2 (the
        # accurate tail, where the posterior — and narrow degeneracies like
        # chi_eff-q — is resolved) benefits from more thorough decorrelation.
        # Both default to ``num_inner_steps`` (unchanged behaviour).
        if phase1_num_inner_steps is None:
            phase1_num_inner_steps = num_inner_steps
        if phase2_num_inner_steps is None:
            phase2_num_inner_steps = num_inner_steps

        k_init, k_p1, k_p2, k_post = jax.random.split(rng_key, 4)

        # Initial live particles from the uniform prior
        particles, logprior_fn = bns.utils.uniform_prior(
            k_init, num_live, self.param_bounds,
        )

        # ── Phase 1: high num_delete, fast bulk exploration ──────────────────
        algo1 = bns.nss.as_top_level_api(
            logprior_fn      = logprior_fn,
            loglikelihood_fn = self._loglikelihood_fn,
            num_inner_steps  = phase1_num_inner_steps,
            num_delete       = phase1_num_delete,
        )
        state = algo1.init(particles)
        step1 = jax.jit(algo1.step)

        dead: list = []
        i1 = 0
        if verbose:
            print(
                f"=== Phase 1 (bulk): num_delete={phase1_num_delete}, "
                f"target Δlog Z < {phase1_delta_logz_threshold} ==="
            )
            print(
                "  JIT-compiling phase-1 kernel… "
                "(blackjax-ns NSS step; one-off, can take 30–90 s on CPU)",
                flush=True,
            )

        t_compile = time.perf_counter()
        while i1 < phase1_max_iterations:
            k_p1, kk = jax.random.split(k_p1)
            state, info = step1(kk, state)
            if i1 == 0 and verbose:
                # Block until the first iteration's arrays materialise, then
                # report the compile + first-step wall-clock so the user can
                # tell compile vs steady-state apart.
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
                    state,
                )
                print(
                    f"  → compile + first step finished in "
                    f"{time.perf_counter()-t_compile:.1f} s; subsequent steps are fast.",
                    flush=True,
                )
            dead.append(info)

            delta_logZ = float(state.logZ_live - state.logZ)
            if verbose and (i1 % 100 == 0):
                print(
                    f"  p1 iter {i1:5d}  logZ={float(state.logZ):+.3f}  "
                    f"logZ_live={float(state.logZ_live):+.3f}  Δ={delta_logZ:+.3f}"
                )
            i1 += 1
            if delta_logZ < phase1_delta_logz_threshold:
                break

        if verbose:
            print(
                f"  → Phase 1 finished after {i1} iterations.  "
                f"logZ={float(state.logZ):+.3f}, Δ={delta_logZ:+.3f}"
            )

        # ── Phase 2: low num_delete, accurate tail ───────────────────────────
        algo2 = bns.nss.as_top_level_api(
            logprior_fn      = logprior_fn,
            loglikelihood_fn = self._loglikelihood_fn,
            num_inner_steps  = phase2_num_inner_steps,
            num_delete       = phase2_num_delete,
        )
        step2 = jax.jit(algo2.step)

        i2 = 0
        if verbose:
            print(
                f"=== Phase 2 (tail): num_delete={phase2_num_delete}, "
                f"target Δlog Z < {log_dlogz_target} ==="
            )
            print(
                "  JIT-compiling phase-2 kernel… "
                "(different num_delete → separate XLA program)",
                flush=True,
            )

        t_compile = time.perf_counter()
        while i2 < phase2_max_iterations:
            k_p2, kk = jax.random.split(k_p2)
            state, info = step2(kk, state)
            if i2 == 0 and verbose:
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
                    state,
                )
                print(
                    f"  → compile + first step finished in "
                    f"{time.perf_counter()-t_compile:.1f} s.",
                    flush=True,
                )
            dead.append(info)

            delta_logZ = float(state.logZ_live - state.logZ)
            if verbose and (i2 % 100 == 0):
                print(
                    f"  p2 iter {i2:5d}  logZ={float(state.logZ):+.3f}  "
                    f"logZ_live={float(state.logZ_live):+.3f}  Δ={delta_logZ:+.3f}"
                )
            i2 += 1
            if delta_logZ < log_dlogz_target:
                break

        if verbose:
            print(
                f"  → Phase 2 finished after {i2} iterations.  "
                f"logZ={float(state.logZ):+.3f}, Δ={delta_logZ:+.3f}"
            )

        # ── Finalise + posterior sample ──────────────────────────────────────
        # ``finalise`` concatenates *every* NSInfo field across all dead
        # records, including the diagnostic ``inner_kernel_info`` whose shape is
        # (num_delete, num_inner_steps). When the two phases use different
        # ``num_inner_steps``, those per-record shapes differ along the
        # inner-steps axis and the concatenation raises. The field is unused by
        # the evidence/posterior, so drop it (set to None → an empty pytree node
        # that ``jax.tree.map`` skips) before finalising.
        dead = [info._replace(inner_kernel_info=None) for info in dead]
        dead_info = bns.utils.finalise(state, dead)

        k_w, k_s = jax.random.split(k_post)
        logw     = bns.utils.log_weights(k_w, dead_info, shape=200)
        logZ_r   = jax.scipy.special.logsumexp(logw, axis=0)
        logZ     = float(jnp.mean(logZ_r))
        logZ_err = float(jnp.std(logZ_r))

        posterior = bns.utils.sample(k_s, dead_info, shape=num_posterior_samples)
        posterior_samples = {k: v for k, v in posterior.items()}

        ess = float(bns.utils.ess(k_w, dead_info))

        return TwoPhaseNestedSamplingResult(
            posterior_samples  = posterior_samples,
            logZ               = logZ,
            logZ_err           = logZ_err,
            phase1_iterations  = i1,
            phase2_iterations  = i2,
            n_iterations       = i1 + i2,
            ess                = ess,
            live_state         = state,
            dead_info          = dead_info,
        )
