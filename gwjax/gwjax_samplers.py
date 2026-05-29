"""
gwjax_samplers.py
==================
Nested-sampling interface for gravitational-wave parameter estimation,
built on top of ``blackjax.ns`` (BlackJAX-NS).

The high-level entry point is :class:`GWjaxNestedSampler`, which couples:

  * a :class:`~gwjax.gwjax_network_ifos_utils.Network` (data + PSDs),
  * a JAX-traceable ``waveform_fn(params, freqs) -> (hp, hc)``,
  * a dict of uniform-prior bounds for the sampled parameters,
  * a dict of fixed values for the remaining parameters,

into a single jit-compiled log-likelihood, samples its posterior with the
adaptive Nested Slice Sampler (NSS), and returns both an evidence estimate
and a set of unweighted posterior draws.

Quickstart
----------
>>> import jax
>>> from gwjax.gwjax_samplers import (
...     GWjaxNestedSampler, build_ripplegw_waveform_fn,
... )
>>> waveform_fn = build_ripplegw_waveform_fn(approximant="IMRPhenomD", f_ref=20.0)
>>> sampler = GWjaxNestedSampler(
...     network=net,
...     waveform_fn=waveform_fn,
...     param_bounds={
...         "m1":          (5.0, 50.0),
...         "m2":          (5.0, 50.0),
...         "distance":    (50.0, 1000.0),
...         "inclination": (0.0, 3.14159),
...         "ra":          (0.0, 6.28318),
...         "dec":        (-1.5708, 1.5708),
...         "psi":         (0.0, 3.14159),
...         "tc":         (-0.05, 0.05),
...     },
...     fixed_params={"chi_1": 0.0, "chi_2": 0.0, "phi_c": 0.0},
... )
>>> result = sampler.run(rng_key=jax.random.PRNGKey(0), num_live=500)
>>> result.logZ, result.posterior_samples["m1"].shape

The sampled likelihood is

    ℒ(θ) = Σ_i  -0.5 · ⟨d_i - h_i(θ) | d_i - h_i(θ)⟩_i,

where ``h_i = (Fp_i · h+ + Fc_i · h×) · exp(-2πi f (tc + Δt_i))``,
``(Fp_i, Fc_i)`` and ``Δt_i`` come from the IFO geometry, and ``tc``
shifts the geocenter time of arrival.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.ns as bns

from gwjax.gwjax_network_ifos_utils import Network
from gwjax.gwjax_likelihood_utils import (
    waveform_projection_fd,
    log_likelihood_ifo,
    precompute_weight,
)


# ── Result container ─────────────────────────────────────────────────────────

class NestedSamplingResult(NamedTuple):
    """Output bundle from :meth:`GWjaxNestedSampler.run`."""
    posterior_samples: dict   # dict[str, jnp.ndarray] of shape (n_samples,)
    logZ:              float  # marginal-likelihood estimate (log)
    logZ_err:          float  # 1-σ uncertainty on logZ
    n_iterations:      int    # number of NS iterations actually run
    ess:               float  # effective sample size of weighted dead points
    live_state:        object # final NSState (for advanced post-processing)
    dead_info:         object # NSInfo with dead + live concatenated


# ── Sky/time parameters consumed by the projection layer ─────────────────────
SKY_PARAMS: tuple[str, ...] = ("ra", "dec", "psi", "tc")


# ── Waveform-function factory: ripplegw ──────────────────────────────────────

def build_ripplegw_waveform_fn(
    approximant: str = "IMRPhenomD",
    f_ref:       float = 20.0,
) -> Callable:
    """Build a JAX-traceable ``(params, freqs) -> (hp, hc)`` for ripplegw.

    Parameters
    ----------
    approximant : str
        One of ``"IMRPhenomD"``, ``"IMRPhenomXAS"``,
        ``"IMRPhenomD_NRTidalv2"``, ``"IMRPhenomPv2"``.
    f_ref : float
        Reference frequency [Hz].

    Returns
    -------
    callable
        ``waveform_fn(params, freqs)`` where ``params`` is a dict of scalar
        JAX values with keys
        ``m1, m2, chi_1, chi_2, distance, inclination, tc, phi_c``.
        Returns complex FD ``(hp, hc)`` aligned with ``freqs``.

    Notes
    -----
    Sky angles (``ra, dec, psi``) and detector-specific time delays are NOT
    applied here; they are handled by the projection layer inside the
    sampler.
    """
    from ripplegw.waveforms import (
        IMRPhenomD, IMRPhenomXAS, IMRPhenomD_NRTidalv2, IMRPhenomPv2,
    )

    _MAP = {
        "IMRPhenomD":           IMRPhenomD.gen_IMRPhenomD_hphc,
        "IMRPhenomXAS":         IMRPhenomXAS.gen_IMRPhenomXAS_hphc,
        "IMRPhenomD_NRTidalv2": IMRPhenomD_NRTidalv2.gen_IMRPhenomD_NRTidalv2_hphc,
        "IMRPhenomPv2":         IMRPhenomPv2.gen_IMRPhenomPv2,
    }
    if approximant not in _MAP:
        raise ValueError(
            f"Unknown approximant '{approximant}'. Choose from {list(_MAP)}."
        )
    gen = _MAP[approximant]

    def waveform_fn(params, freqs):
        # Inline (m1, m2) → (Mc, eta); ripplegw's helper moved around the
        # 0.0.9 → 0.0.10 boundary so we don't import it.
        m1, m2 = params["m1"], params["m2"]
        Mc  = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        eta = m1 * m2 / (m1 + m2) ** 2
        # tc = 0 in the ripplegw theta. ripplegw applies tc as an FD shift
        # exp(-2πi·f·tc) internally; the projection layer
        # (gwjax_likelihood_utils.waveform_projection_fd) also applies tc via
        # (tc + dt_ifo). Passing params["tc"] here would double-count it. The
        # SHARPy/bilby/LAL convention is to keep tc *only* in the projection
        # layer — that's what the sampler likelihood expects.
        theta = jnp.stack([
            Mc, eta,
            params["chi_1"], params["chi_2"],
            params["distance"],
            jnp.zeros_like(m1),                       # tc = 0 (see comment above)
            params["phi_c"],
            params["inclination"],
        ])
        return gen(freqs, theta, f_ref)

    return waveform_fn


# ── Sampler ───────────────────────────────────────────────────────────────────

class GWjaxNestedSampler:
    """Adaptive Nested Slice Sampler bound to a GW data network.

    Parameters
    ----------
    network : Network
        A populated :class:`~gwjax.gwjax_network_ifos_utils.Network` with
        strain data and PSDs set on each interferometer.
    waveform_fn : callable or :class:`~gwjax.gwjax_custom_waveform.CustomWaveform`
        ``waveform_fn(params, freqs) -> (hp, hc)``. Must be JAX-traceable.
        ``params`` is a dict of scalar JAX values. A
        :class:`~gwjax.gwjax_custom_waveform.CustomWaveform` instance
        works directly (its ``__call__`` matches this contract); you can
        also pass the output of :func:`build_ripplegw_waveform_fn` or any
        function you wrote yourself.
    param_bounds : dict[str, tuple[float, float]]
        Uniform-prior bounds for the parameters that NS will sample. All
        other parameters consumed by ``waveform_fn`` or the projection
        layer must be supplied via ``fixed_params``.
    fixed_params : dict[str, float] or None
        Constant parameter values that are not sampled. Merged into the
        particle dict before each waveform call.
    gmst : float
        Greenwich Mean Sidereal Time [rad]. Defaults to 0.
    """

    def __init__(
        self,
        network:        Network,
        waveform_fn:    Callable,
        param_bounds:   dict,
        fixed_params:   dict | None = None,
        gmst:           float = 0.0,
    ) -> None:
        self.network      = network
        # Accept either a raw (params, grid_array) -> (hp, hc) callable
        # or a CustomWaveform instance. The latter is itself callable
        # with the right signature via its __call__.
        if not callable(waveform_fn):
            raise TypeError(
                "waveform_fn must be callable. Pass a function, a "
                "CustomWaveform, or the output of a build_*_waveform_fn helper."
            )
        self.waveform_fn  = waveform_fn
        self.param_bounds = dict(param_bounds)
        self.fixed_params = dict(fixed_params or {})
        self.gmst         = float(gmst)

        # The projection layer needs ra, dec, psi, tc.
        provided = set(self.param_bounds) | set(self.fixed_params)
        missing = set(SKY_PARAMS) - provided
        if missing:
            raise ValueError(
                "Missing sky/time parameters required by the projection "
                f"layer: {sorted(missing)}. Provide them either in "
                "`param_bounds` (to sample) or `fixed_params` (to fix)."
            )

        # Snapshot per-IFO data and pre-computed inner-product weight into
        # closures. The weight w = 4·df·mask/psd is constant across NS
        # iterations, so we compute it once here and keep only (d, w) on
        # the hot path.
        self._d_fd, self._weight = {}, {}
        for ifo in network.interferometers:
            if ifo.grid is None or ifo.psd is None or ifo.strain_data_fd is None:
                raise RuntimeError(
                    f"Interferometer '{ifo.name}' is missing grid/PSD/data."
                )
            self._d_fd[ifo.name]   = ifo.strain_data_fd
            self._weight[ifo.name] = precompute_weight(
                ifo.psd, ifo.grid.df, mask=ifo.grid.frequency_mask,
            )

        # All IFOs share the same FD grid.
        self._freqs = network.interferometers[0].grid.frequency_domain_array

        # Build the per-particle log-likelihood once.
        self._loglikelihood_fn = self._build_loglikelihood()

    # ── log-likelihood ────────────────────────────────────────────────────────

    def _build_loglikelihood(self) -> Callable:
        net      = self.network
        waveform = self.waveform_fn
        fixed    = self.fixed_params
        gmst     = self.gmst
        freqs    = self._freqs
        d_fd     = self._d_fd
        weight   = self._weight

        def loglikelihood_fn(particle):
            params = {**fixed, **particle}
            hp, hc = waveform(params, freqs)

            ra, dec, psi = params["ra"], params["dec"], params["psi"]
            tc           = params["tc"]

            logl = jnp.asarray(0.0)
            for ifo in net.interferometers:
                Fp, Fc = ifo.antenna_pattern(ra, dec, psi, gmst)
                dt_ifo = ifo.time_delay_from_geocenter(ra, dec, gmst)
                h_ifo  = waveform_projection_fd(
                    hp, hc, Fp, Fc, freqs, tc + dt_ifo,
                )
                logl   = logl + log_likelihood_ifo(
                    d_fd[ifo.name], h_ifo, weight[ifo.name],
                )
            return logl

        return loglikelihood_fn

    @property
    def loglikelihood_fn(self) -> Callable:
        """The per-particle log-likelihood closure (jit/vmap/grad-safe)."""
        return self._loglikelihood_fn

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        rng_key:               jax.Array,
        num_live:              int = 500,
        num_inner_steps:       int | None = None,
        num_delete:            int = 1,
        max_iterations:        int = 10000,
        log_dlogz_target:      float = -3.0,
        num_posterior_samples: int = 2000,
        verbose:               bool = False,
    ) -> NestedSamplingResult:
        """Run the nested sampler until live-point evidence is negligible.

        Parameters
        ----------
        rng_key : PRNGKey
            JAX random key.
        num_live : int
            Number of live particles.
        num_inner_steps : int, optional
            Slice-sampling steps per NS iteration. Defaults to ``5·d``.
        num_delete : int
            Particles replaced per NS step. ``1`` is the classical choice.
        max_iterations : int
            Hard cap on NS iterations.
        log_dlogz_target : float
            Termination tolerance. The loop stops when
            ``state.logZ_live - state.logZ < log_dlogz_target``.
            Default ``-3`` (≈5% remaining evidence).
        num_posterior_samples : int
            Size of the unweighted posterior sample.
        verbose : bool
            Print progress every 100 iterations.

        Returns
        -------
        NestedSamplingResult
        """
        d = len(self.param_bounds)
        if num_inner_steps is None:
            num_inner_steps = max(5 * d, 10)

        k_init, k_loop, k_post = jax.random.split(rng_key, 3)

        # Initial live particles drawn from the uniform prior.
        particles, logprior_fn = bns.utils.uniform_prior(
            k_init, num_live, self.param_bounds,
        )

        algo = bns.nss.as_top_level_api(
            logprior_fn      = logprior_fn,
            loglikelihood_fn = self._loglikelihood_fn,
            num_inner_steps  = num_inner_steps,
            num_delete       = num_delete,
        )
        state = algo.init(particles)
        step  = jax.jit(algo.step)

        dead: list = []
        i = 0
        if verbose:
            print(
                "JIT-compiling NS kernel… "
                "(one-off, can take 30–90 s on CPU; subsequent iterations are fast)",
                flush=True,
            )
        import time as _time
        t_compile = _time.perf_counter()
        while i < max_iterations:
            k_loop, k_step = jax.random.split(k_loop)
            state, info    = step(k_step, state)
            if i == 0 and verbose:
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
                    state,
                )
                print(
                    f"  → compile + first step finished in "
                    f"{_time.perf_counter()-t_compile:.1f} s.",
                    flush=True,
                )
            dead.append(info)

            delta_logZ = float(state.logZ_live - state.logZ)
            if verbose and (i % 100 == 0):
                print(
                    f"iter {i:5d}  logZ={float(state.logZ):+.3f}  "
                    f"logZ_live={float(state.logZ_live):+.3f}  "
                    f"Δ={delta_logZ:+.3f}"
                )
            i += 1
            if delta_logZ < log_dlogz_target:
                break

        # Combine dead + final live into a single NSInfo.
        dead_info = bns.utils.finalise(state, dead)

        # logZ estimate via Skilling's stochastic logX (averaged over realisations).
        k_w, k_s = jax.random.split(k_post)
        logw     = bns.utils.log_weights(k_w, dead_info, shape=200)
        logZ_r   = jax.scipy.special.logsumexp(logw, axis=0)
        logZ     = float(jnp.mean(logZ_r))
        logZ_err = float(jnp.std(logZ_r))

        posterior = bns.utils.sample(k_s, dead_info, shape=num_posterior_samples)
        posterior_samples = {k: v for k, v in posterior.items()}

        ess = float(bns.utils.ess(k_w, dead_info))

        return NestedSamplingResult(
            posterior_samples = posterior_samples,
            logZ              = logZ,
            logZ_err          = logZ_err,
            n_iterations      = i,
            ess               = ess,
            live_state        = state,
            dead_info         = dead_info,
        )
