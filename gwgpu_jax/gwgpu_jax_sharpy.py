"""
gwgpu_jax_sharpy.py
===================
SHARPy — *Sequential HAmiltonian Riemann montecarlo PYthon sampler* — as a
gwgpu_jax sampling backend.

SHARPy (G. Demasi, W. Del Pozzo et al.; https://github.com/saulo-albuquerque-phys/sharpy,
branch ``saulo-lorenzo-adriano-patch-beta``) is an adaptively-tempered
Sequential Monte Carlo (SMC) sampler whose mutation move is NUTS with a
**per-particle, Hessian-based (SoftAbs) metric** and reflective / periodic
boundary handling. One SMC step is:

1. **Adaptive tempering.** Choose the next inverse temperature ``β_{t+1}`` by
   bisection so that the ESS of the incremental weights
   ``w_i = L(θ_i)^{β_{t+1} - β_t}`` equals ``alpha · N``.
2. **Evidence update.** ``log Z += log mean(w)``.
3. **Resample** multinomially with the normalised weights.
4. **Mutate** every particle with NUTS on ``β_{t+1} log L + log π``, using as
   inverse mass matrix the inverse of the SoftAbs-regularised negative Hessian
   of the tempered log-posterior at that particle (Betancourt 2013,
   arXiv:1212.4693).

It stops after the step that reaches ``β = 1``. The posterior is drawn by
rejection sampling from the *recycled* mixture of every tempered population
(SHARPy's ``draw_iid_samples``); the last population is also returned.

What is different from the upstream package
-------------------------------------------
This is a self-contained pure-JAX port of ``sharpy.smc_functions.run_sharpy``.
Upstream SHARPy needs a *fork* of BlackJAX (``gabrieledemasi/blackjax``, whose
``nuts.build_kernel`` takes ``bounds, boundary_conditions``) and ``netket``
(for ``vmap_chunked``). That fork cannot coexist with the BlackJAX commit
gwgpu_jax pins for ``blackjax.ns``. Here the bounded leapfrog is written as a
custom integrator for the **stock** ``blackjax.nuts.build_kernel``, and chunked
vmaps use ``jax.lax.map(batch_size=...)``, so no extra dependency is needed.

Further deliberate differences:

* **Sampling space.** Particles live on the unit hypercube and are mapped to
  physical space by the gwgpu_jax prior inverse-CDFs (as in
  :class:`~gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler`). The prior is then
  uniform on ``[0, 1]^d``, which is exactly SHARPy's set-up (uniform initial
  draws, ``prior(θ) = 0`` inside reflective/periodic bounds), and non-uniform
  gwgpu_jax priors (``"sin"``, ``"cos"``, ``"volumetric"``…) are supported for
  free. For uniform priors this is just an affine rescaling of the physical
  space, except that the SoftAbs floor ``1/softabs_alpha`` now acts in
  unit-cube units.
* **Tempering bisection** evaluates the ESS at the bisection midpoint.
  Upstream's ``find_next_beta`` evaluates it at the upper end, which turns it
  into a halving search that lands on a ``β`` with ESS at or above the target
  (smaller, more conservative steps).
* **The per-particle log-likelihood is cached** between steps rather than
  recomputed at every bisection iteration.
* ``metric="ensemble"`` is offered as an alternative to the Hessian metric:
  the inverse mass matrix is the covariance of the resampled population,
  shared by all particles. A metric that depends on the particle's own start
  point (``metric="hessian"``, the SHARPy default) makes the NUTS move only
  approximately invariant for the tempered target; the ensemble metric keeps it
  exact.

Citation
--------
If you use this driver, cite SHARPy (Demasi, Del Pozzo et al.), BlackJAX
(Cabezas et al. 2024, arXiv:2402.10797), NUTS (Hoffman & Gelman 2014) and
SoftAbs Riemannian HMC (Betancourt 2013, arXiv:1212.4693).
"""

from __future__ import annotations

import os
import time
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import blackjax
from blackjax.mcmc import hmc
from blackjax.mcmc.integrators import IntegratorState

from gwgpu_jax.gwgpu_jax_network_ifos_utils import Network
from gwgpu_jax.gwgpu_jax_samplers import GWgpu_jaxNestedSampler
from gwgpu_jax.gwgpu_jax_prior_definitions import resolve_priors
from gwgpu_jax.gwgpu_jax_acceptance_walk import build_prior_transform


# ── Result container ─────────────────────────────────────────────────────────

class SHARPyResult(NamedTuple):
    """Output bundle from :meth:`GWgpu_jaxSHARPySampler.run`.

    ``posterior_samples`` (recycled-mixture rejection draws) and
    ``final_samples`` (the β = 1 population) are dicts in **physical**
    parameter space, matching :class:`~gwgpu_jax.AcceptanceWalkResult`.
    """
    posterior_samples: dict        # recycled, unweighted draws (SHARPy default)
    final_samples:     dict        # last tempered population (β = 1), N draws
    logZ:              float
    logZ_err:          float
    n_steps:           int
    betas:             np.ndarray  # (n_steps,)
    ess:               np.ndarray  # (n_steps,) ESS of the incremental weights
    acceptance:        np.ndarray  # (n_steps,) mean NUTS acceptance rate
    divergences:       np.ndarray  # (n_steps,) fraction of divergent moves
    history:           dict        # per-step unit-cube particles / logL / log-weights


# ── SHARPy numerical building blocks ─────────────────────────────────────────

def softabs_inverse_metric(neg_hessian: jax.Array, alpha: float) -> jax.Array:
    """Inverse of the SoftAbs metric ``V diag(λ coth(αλ)) Vᵀ`` (arXiv:1212.4693).

    ``λ coth(αλ) ≥ 1/α`` for every ``λ``, so ``1/alpha`` is the stiffness floor
    for flat or negatively-curved directions. Non-finite Hessians (a
    non-differentiable point in the model) fall back to the isotropic floor.
    """
    h = 0.5 * (neg_hessian + neg_hessian.T)
    h = jnp.where(jnp.all(jnp.isfinite(h)), h, jnp.zeros_like(h))
    lam, vec = jnp.linalg.eigh(h)
    x = alpha * lam
    small = jnp.abs(x) < 1e-6
    x_safe = jnp.where(small, 1.0, x)
    soft = jnp.where(small, 1.0 / alpha + alpha * lam**2 / 3.0,
                     lam / jnp.tanh(x_safe))
    return (vec / soft) @ vec.T


def fold_into_bounds(q, lower, upper, periodic):
    """Map ``q`` back into ``[lower, upper]``.

    Periodic coordinates wrap; reflective ones are folded (a triangle wave, so
    any number of crossings is handled). Returns the folded position and a
    ``±1`` factor that flips the momentum of reflective coordinates that
    crossed an odd number of times.
    """
    width = upper - lower
    x = q - lower
    wrapped = jnp.mod(x, width)
    y = jnp.mod(x, 2.0 * width)
    reflected = jnp.where(y > width, 2.0 * width - y, y)
    odd = jnp.mod(jnp.floor(x / width), 2.0) == 1.0
    flip = jnp.where(periodic | ~odd, 1.0, -1.0)
    return lower + jnp.where(periodic, wrapped, reflected), flip


def bounded_velocity_verlet(lower, upper, periodic):
    """Velocity-Verlet integrator with SHARPy's boundary handling.

    Drop-in ``integrator`` for the stock ``blackjax.nuts.build_kernel``: after
    the drift the position is folded into the box and, for reflective
    coordinates, the momentum is flipped before the closing half-kick (the
    same ordering as the bounded integrator in ``gabrieledemasi/blackjax``).
    """
    def integrator(logdensity_fn: Callable, kinetic_energy_fn: Callable):
        logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
        kinetic_grad_fn        = jax.grad(kinetic_energy_fn)

        def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
            q, p, _, grad = state
            p = p + 0.5 * step_size * grad
            q = q + step_size * kinetic_grad_fn(p)
            q, flip = fold_into_bounds(q, lower, upper, periodic)
            p = p * flip
            logdensity, grad = logdensity_and_grad_fn(q)
            p = p + 0.5 * step_size * grad
            return IntegratorState(q, p, logdensity, grad)

        return one_step

    return integrator


def incremental_ess(logL: jax.Array, dbeta) -> jax.Array:
    """ESS of the incremental weights ``exp(dbeta · logL)``."""
    lw = dbeta * logL
    lw = lw - jnp.max(lw)
    return jnp.exp(2.0 * logsumexp(lw) - logsumexp(2.0 * lw))


@jax.jit
def next_beta(logL: jax.Array, beta_prev, ess_target) -> jax.Array:
    """Largest ``β ∈ (β_prev, 1]`` whose incremental ESS is ≥ ``ess_target``."""
    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        ok = incremental_ess(logL, mid - beta_prev) >= ess_target
        return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)

    lo, _ = jax.lax.fori_loop(0, 60, body, (beta_prev, jnp.asarray(1.0)))
    reach_one = incremental_ess(logL, 1.0 - beta_prev) >= ess_target
    return jnp.where(reach_one, 1.0, lo)


def recycled_posterior(key, history: dict, logZ_cumulative: np.ndarray):
    """SHARPy's ``draw_iid_samples``: rejection-sample the posterior from the
    equal-weight mixture of all tempered populations.

    Population ``k`` approximately follows ``π L^{β_k} / Z_k``; the posterior
    ``π L / Z`` is drawn from the mixture by rejection with acceptance ratio
    ``L / Σ_k (L^{β_k} / Z_k)``, normalised by its maximum.
    """
    U     = np.concatenate(history["particles"])           # (K·N, d)
    logL  = np.concatenate(history["logL"])                # (K·N,)
    betas = np.asarray(history["beta"])                    # (K,)
    log_mix = logsumexp(jnp.asarray(betas[:, None] * logL[None, :]
                                    - logZ_cumulative[:, None]), axis=0)
    log_ratio = np.asarray(logL - (log_mix - np.log(len(betas))))
    log_ratio = np.where(np.isfinite(log_ratio), log_ratio, -np.inf)
    log_u = np.log(np.asarray(jax.random.uniform(key, log_ratio.shape)))
    accepted = log_ratio - np.max(log_ratio) > log_u
    return U[accepted]


# ── Sampler ──────────────────────────────────────────────────────────────────

class GWgpu_jaxSHARPySampler:
    """SHARPy tempered SMC with Riemannian (SoftAbs-Hessian) NUTS moves.

    The constructor signature mirrors
    :class:`~gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler`; internally it builds a
    :class:`GWgpu_jaxNestedSampler` only to reuse its validated
    ``loglikelihood_fn`` and resolved prior specs, so the likelihood, waveform,
    network and priors are identical to the nested-sampling runs and the
    evidences are directly comparable.

    The likelihood must be differentiable (``jax.grad`` / ``jax.hessian``),
    because SHARPy is gradient-based.

    Parameters
    ----------
    network, waveform_fn, param_bounds, fixed_params, priors, gmst
        Passed straight through to :class:`GWgpu_jaxNestedSampler`.
    periodic_params : sequence of str, optional
        Parameters whose unit-cube coordinate wraps (periodic boundary);
        every other coordinate is reflective. Defaults to SHARPy's choice
        ``("ra", "phi_c", "psi")``; names not in ``param_bounds`` are ignored.
    """

    def __init__(
        self,
        network:         Network,
        waveform_fn:     Callable,
        param_bounds:    dict,
        fixed_params:    Optional[dict] = None,
        gmst:            Optional[float] = None,
        priors:          Optional[dict] = None,
        periodic_params: Sequence[str] = ("ra", "phi_c", "psi"),
    ) -> None:
        self._ns = GWgpu_jaxNestedSampler(
            network=network, waveform_fn=waveform_fn,
            param_bounds=param_bounds, fixed_params=fixed_params,
            gmst=gmst, priors=priors,
        )
        self._init_from_parts(
            loglik_fn=self._ns.loglikelihood_fn,
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
        periodic_params:  Sequence[str] = ("ra", "phi_c", "psi"),
    ) -> "GWgpu_jaxSHARPySampler":
        """Build the sampler around an arbitrary physical log-likelihood.

        ``loglikelihood_fn(params)`` takes a dict keyed by ``param_bounds``
        names (physical space) and returns a scalar; it must be
        jit/vmap/grad-safe.
        """
        self = cls.__new__(cls)
        self._ns = None
        specs = resolve_priors(dict(param_bounds), dict(priors or {}))
        self._init_from_parts(
            loglik_fn=loglikelihood_fn, prior_specs=specs,
            param_bounds=param_bounds, periodic_params=periodic_params,
        )
        return self

    # ── unit-cube <-> physical ────────────────────────────────────────────────

    def to_physical(self, U) -> dict:
        """Map unit-cube particles ``(n, d)`` to a dict of physical arrays."""
        prior_transform = build_prior_transform(self._prior_specs, self.param_names)
        U = jnp.asarray(U)
        return jax.vmap(prior_transform)(
            {name: U[:, i] for i, name in enumerate(self.param_names)})

    def _unit_cube_loglik(self) -> Callable:
        prior_transform = build_prior_transform(self._prior_specs, self.param_names)
        names, loglik   = self.param_names, self._loglik_fn

        def loglik_u(u):
            logl = loglik(prior_transform({n: u[i] for i, n in enumerate(names)}))
            return jnp.where(jnp.isnan(logl), -jnp.inf, logl)

        return loglik_u

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        rng_key:           jax.Array,
        num_particles:     int = 4000,
        alpha:             float = 0.95,
        step_size:         float = 0.3,
        max_num_doublings: int = 6,
        num_mcmc_steps:    int = 1,
        metric:            str = "hessian",
        softabs_alpha:     float = 5e-3,
        chunk_size:        int = 500,
        max_steps:         int = 500,
        initial_particles: Optional[jax.Array] = None,
        outdir:            Optional[str] = None,
        label:             str = "sharpy",
        verbose:           bool = False,
    ) -> SHARPyResult:
        """Run SHARPy from the prior (β = 0) to the posterior (β = 1).

        Parameters
        ----------
        rng_key : PRNGKey
        num_particles : int
            Number of SMC particles ``N``.
        alpha : float
            Target ESS fraction per tempering step (SHARPy's ``alpha``).
            Closer to 1 = more, smaller temperature steps.
        step_size : float
            NUTS leapfrog step size (fixed; the Hessian metric absorbs the
            scale). SHARPy default 0.3.
        max_num_doublings : int
            NUTS tree-depth cap (≤ ``2**max_num_doublings`` leapfrog steps).
            SHARPy default 6.
        num_mcmc_steps : int
            NUTS transitions per particle per SMC step. SHARPy uses 1; raise it
            if the posterior looks under-mixed (e.g. many duplicate particles).
        metric : {"hessian", "ensemble"}
            ``"hessian"``: per-particle SoftAbs(-∇² log p_β)⁻¹, recomputed at the
            start of every NUTS transition (SHARPy). ``"ensemble"``: covariance
            of the resampled population, shared by all particles (cheaper, and
            leaves the tempered target exactly invariant).
        softabs_alpha : float
            SoftAbs sharpness; ``1/softabs_alpha`` is the minimum metric
            eigenvalue (unit-cube units).
        chunk_size : int
            Particles per vectorised batch in the likelihood / Hessian / NUTS
            maps. Lower it if the GPU runs out of memory.
        max_steps : int
            Hard cap on the number of SMC steps.
        initial_particles : array, optional
            ``(N, d)`` unit-cube starting particles; default is prior draws.
        outdir, label : str, optional
            If ``outdir`` is given, save ``<outdir>/<label>_result.npz``.
        verbose : bool

        Returns
        -------
        SHARPyResult
        """
        if metric not in ("hessian", "ensemble"):
            raise ValueError(f"metric must be 'hessian' or 'ensemble', got {metric!r}")

        d        = len(self.param_names)
        N        = int(num_particles)
        lower    = jnp.zeros(d)
        upper    = jnp.ones(d)
        periodic = jnp.array([n in self.periodic_params for n in self.param_names])
        loglik_u = self._unit_cube_loglik()

        nuts_kernel = blackjax.nuts.build_kernel(
            integrator=bounded_velocity_verlet(lower, upper, periodic))

        @jax.jit
        def batched_loglik(U):
            return jax.lax.map(loglik_u, U, batch_size=chunk_size)

        def mutate_one(key, u, beta, shared_inv_mass):
            logdensity_fn = lambda x: beta * loglik_u(x)

            def transition(state, k):
                if metric == "hessian":
                    inv_mass = softabs_inverse_metric(
                        -jax.hessian(logdensity_fn)(state.position), softabs_alpha)
                else:
                    inv_mass = shared_inv_mass
                state, info = nuts_kernel(k, state, logdensity_fn, step_size,
                                          inv_mass, max_num_doublings)
                return state, (info.acceptance_rate, info.is_divergent)

            state = hmc.init(u, logdensity_fn)
            state, (acc, div) = jax.lax.scan(
                transition, state, jax.random.split(key, num_mcmc_steps))
            return state.position, jnp.mean(acc), jnp.mean(div)

        @jax.jit
        def mutate(keys, U, beta, shared_inv_mass):
            return jax.lax.map(
                lambda xs: mutate_one(xs[0], xs[1], beta, shared_inv_mass),
                (keys, U), batch_size=chunk_size)

        k_init, k_loop, k_post = jax.random.split(rng_key, 3)
        if initial_particles is None:
            U = jax.random.uniform(k_init, (N, d))
        else:
            U = jnp.asarray(initial_particles)
            if U.shape != (N, d):
                raise ValueError(f"initial_particles must have shape {(N, d)}, got {U.shape}")

        t0 = time.perf_counter()
        if verbose:
            print("Evaluating log-likelihood on the initial prior draws "
                  "(includes JIT compile)…", flush=True)
        logL = batched_loglik(U)

        history = {"particles": [], "logL": [], "log_weights": [], "beta": []}
        ess_hist, acc_hist, div_hist = [], [], []
        logZ, dlogZ2, logZ_cum = 0.0, 0.0, []
        beta = 0.0
        step = 0

        while beta < 1.0 - 1e-8 and step < max_steps:
            k_loop, k_res, k_mut = jax.random.split(k_loop, 3)

            beta_next = float(next_beta(logL, beta, alpha * N))
            log_w     = (beta_next - beta) * logL
            ess       = float(incremental_ess(logL, beta_next - beta))

            # evidence increment + SHARPy's delta-method variance
            logZ += float(logsumexp(log_w) - jnp.log(N))
            w = np.exp(np.asarray(log_w - jnp.max(log_w)))
            dlogZ2 += float(np.var(w) / (np.mean(w) ** 2 * N) * (1.0 + ess / N))

            idx = jax.random.choice(k_res, N, (N,), p=jnp.exp(log_w - logsumexp(log_w)))
            U   = U[idx]

            shared_inv_mass = (jnp.atleast_2d(jnp.cov(U, rowvar=False))
                               + 1e-10 * jnp.eye(d))
            U, acc, div = mutate(jax.random.split(k_mut, N), U,
                                 jnp.asarray(beta_next), shared_inv_mass)
            logL = batched_loglik(U)

            history["particles"].append(np.asarray(U))
            history["logL"].append(np.asarray(logL))
            history["log_weights"].append(np.asarray(log_w))
            history["beta"].append(beta_next)
            ess_hist.append(ess)
            acc_hist.append(float(jnp.mean(acc)))
            div_hist.append(float(jnp.mean(div)))
            logZ_cum.append(logZ)

            beta  = beta_next
            step += 1
            if verbose:
                print(f"step {step:3d}  β={beta:.3e}  ESS={ess:7.1f}  "
                      f"acc={acc_hist[-1]:.2f}  div={div_hist[-1]:.3f}  "
                      f"logZ={logZ:+.3f}  t={time.perf_counter()-t0:.0f}s",
                      flush=True)

        if beta < 1.0 - 1e-8:
            raise RuntimeError(f"SHARPy did not reach β = 1 within max_steps={max_steps} "
                               f"(stopped at β = {beta:.3e}).")

        U_post = recycled_posterior(k_post, history, np.asarray(logZ_cum))
        result = SHARPyResult(
            posterior_samples = {k: np.asarray(v) for k, v in self.to_physical(U_post).items()},
            final_samples     = {k: np.asarray(v) for k, v in self.to_physical(U).items()},
            logZ              = logZ,
            logZ_err          = float(np.sqrt(dlogZ2)),
            n_steps           = step,
            betas             = np.asarray(history["beta"]),
            ess               = np.asarray(ess_hist),
            acceptance        = np.asarray(acc_hist),
            divergences       = np.asarray(div_hist),
            history           = history,
        )
        if verbose:
            print(f"done: {step} steps, logZ = {result.logZ:+.3f} ± {result.logZ_err:.3f}, "
                  f"{len(U_post)} recycled posterior samples, "
                  f"{time.perf_counter()-t0:.0f} s", flush=True)

        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            np.savez(
                os.path.join(outdir, f"{label}_result.npz"),
                param_names=np.array(self.param_names),
                logZ=result.logZ, logZ_err=result.logZ_err,
                betas=result.betas, ess=result.ess,
                acceptance=result.acceptance, divergences=result.divergences,
                **{f"posterior_{k}": v for k, v in result.posterior_samples.items()},
                **{f"final_{k}": v for k, v in result.final_samples.items()},
            )
        return result
