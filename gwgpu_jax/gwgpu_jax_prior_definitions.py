"""
gwgpu_jax_prior_definitions.py
==========================
Non-uniform prior definitions for GWgpu_jax nested-sampling parameter
estimation — fully JAX / ``jit``-traceable, drop-in compatible with the
``blackjax.ns`` machinery already used across the pipeline.

Motivation
----------
The base GWgpu_jax samplers draw every sampled parameter from a *plain uniform*
prior via :func:`blackjax.ns.utils.uniform_prior`. That is fine for masses,
spins, tidal parameters, etc., but the *geometric* parameters have natural,
physically-motivated priors that are **not** uniform in the sampled
coordinate:

============  ====================  ===================================
Parameter     Physical prior        Density (in the sampled coordinate)
============  ====================  ===================================
inclination   isotropic orientation ``p(θ) ∝ sin θ``     (:class:`SinUniform`)
declination   isotropic sky         ``p(δ) ∝ cos δ``     (:class:`CosUniform`)
distance      uniform in volume     ``p(d) ∝ d²``        (:class:`Volumetric`)
============  ====================  ===================================

How it plugs into nested sampling
---------------------------------
The ``blackjax.ns`` Nested Slice Sampler uses ``logprior_fn`` as the *target
density* of its inner Hit-and-Run slice kernel (see
``blackjax/ns/nss.py``: ``logdensity_fn = logprior_fn``). Therefore a
non-uniform prior is obtained *exactly* — with **zero** hot-path overhead —
by simply:

  1. drawing the initial live points from the desired prior, and
  2. supplying a ``logprior_fn`` whose density matches it (normalised, and
     ``-inf`` outside the support).

This module provides both, through a single helper :func:`sample_prior`
whose signature is a **drop-in replacement** for
``blackjax.ns.utils.uniform_prior``::

    particles, logprior_fn = sample_prior(rng_key, num_live, prior_specs)

where ``prior_specs`` maps each sampled parameter name to a
:class:`PriorSpec`.

Each :class:`PriorSpec` exposes two scalar, traceable methods:

  * ``sample(rng_key) -> x``  — an inverse-CDF draw from the prior, so the
    initial live points are *exactly* prior-distributed (no rejection).
  * ``log_prob(x) -> logp``   — the **normalised** log-density, returning
    ``-inf`` outside ``[a, b]`` so the slice sampler respects the support.

Because the densities are normalised, the evidence ``logZ`` returned by the
sampler is directly comparable across different prior choices.

Quickstart
----------
>>> import jax
>>> from gwgpu_jax.gwgpu_jax_prior_definitions import (
...     Uniform, SinUniform, CosUniform, Volumetric, sample_prior,
... )
>>> specs = {
...     "m1":          Uniform(5.0, 50.0),
...     "distance":    Volumetric(50.0, 1000.0),   # p(d) ∝ d²
...     "inclination": SinUniform(0.0, 3.14159),   # p(θ) ∝ sin θ
...     "dec":         CosUniform(-1.5708, 1.5708),# p(δ) ∝ cos δ
... }
>>> particles, logprior_fn = sample_prior(jax.random.PRNGKey(0), 500, specs)

Convenience string aliases
--------------------------
For ergonomic notebook use, :func:`resolve_priors` lets you keep your
existing ``param_bounds`` dict and only name the *kind* of prior per
parameter::

    specs = resolve_priors(
        param_bounds = {"distance": (50.0, 1000.0),
                        "inclination": (0.0, 3.14159),
                        "dec": (-1.5708, 1.5708), ...},
        priors       = {"distance": "volumetric",
                        "inclination": "sin",
                        "dec": "cos"},
    )

Any parameter not named in ``priors`` falls back to :class:`Uniform` over
its ``param_bounds`` entry — i.e. unchanged behaviour.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
    "PriorSpec",
    "Uniform",
    "SinUniform",
    "CosUniform",
    "PowerLaw",
    "Volumetric",
    "UniformInComponentsChirpMass",
    "UniformInComponentsMassRatio",
    "AlignedSpin",
    "build_prior",
    "sample_prior",
    "resolve_priors",
    "PRIOR_ALIASES",
]


# ── Base class ────────────────────────────────────────────────────────────────

class PriorSpec:
    """Abstract 1-D prior on a closed interval ``[a, b]``.

    Subclasses implement two scalar, JAX-traceable methods:

    * :meth:`sample` — an inverse-CDF transform of a single uniform draw,
      so initial live points are *exactly* prior-distributed.
    * :meth:`log_prob` — the **normalised** log-density on ``[a, b]``,
      returning ``-inf`` outside the interval.

    Attributes
    ----------
    a, b : float
        Lower / upper bounds of the support. ``b > a`` is required.
    """

    def __init__(self, a: float, b: float) -> None:
        a = float(a)
        b = float(b)
        if not (b > a):
            raise ValueError(
                f"{type(self).__name__} requires b > a, got a={a}, b={b}."
            )
        self.a = a
        self.b = b

    @property
    def bounds(self) -> Tuple[float, float]:
        """The ``(a, b)`` support of the prior."""
        return (self.a, self.b)

    # subclasses override these two
    def sample(self, rng_key: jax.Array) -> jax.Array:  # pragma: no cover
        raise NotImplementedError

    def log_prob(self, x: jax.Array) -> jax.Array:  # pragma: no cover
        raise NotImplementedError

    # Optional: the deterministic inverse-CDF (percent-point) transform mapping a
    # uniform draw ``u ∈ [0, 1]`` to a physical value in ``[a, b]``. ``sample``
    # is exactly ``icdf(U(0,1))``; exposing ``icdf`` on its own lets unit-cube
    # samplers (e.g. the acceptance-walk kernel) reuse *this* prior definition as
    # their ``prior_transform`` instead of re-deriving the transforms. Purely
    # additive: ``sample`` is left untouched, so the existing NSS path is
    # numerically unchanged.
    def icdf(self, u: jax.Array) -> jax.Array:  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__} does not implement icdf(); it is only "
            "required for unit-hypercube samplers."
        )

    # shared helper: NaN-safe masking to -inf outside [a, b]
    def _mask_support(self, x: jax.Array, logp: jax.Array) -> jax.Array:
        inside = (x >= self.a) & (x <= self.b)
        return jnp.where(inside, logp, -jnp.inf)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(a={self.a}, b={self.b})"


# ── Uniform ───────────────────────────────────────────────────────────────────

class Uniform(PriorSpec):
    """Plain uniform prior, ``p(x) = 1 / (b - a)`` on ``[a, b]``.

    Provided for completeness so a single ``prior_specs`` dict can mix
    uniform and non-uniform parameters.
    """

    def sample(self, rng_key: jax.Array) -> jax.Array:
        return jax.random.uniform(rng_key, minval=self.a, maxval=self.b)

    def icdf(self, u: jax.Array) -> jax.Array:
        return self.a + jnp.asarray(u) * (self.b - self.a)

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        logp = jnp.full_like(jnp.asarray(x, dtype=jnp.result_type(float)),
                             -jnp.log(self.b - self.a))
        return self._mask_support(x, logp)


# ── Sine prior (isotropic inclination / zenith angle) ─────────────────────────

class SinUniform(PriorSpec):
    r"""Prior with density ``p(θ) ∝ sin θ`` on ``[a, b] ⊆ [0, π]``.

    This is the marginal of an *isotropic* orientation on the sphere for a
    polar/zenith angle (e.g. the inclination ``θ_JN``). The normalised
    density is

    .. math::
        p(\theta) = \frac{\sin\theta}{\cos a - \cos b}, \qquad a \le \theta \le b.

    Inverse-CDF sampling uses
    ``θ = arccos(cos a − u (cos a − cos b))`` for ``u ~ U(0, 1)``.
    """

    def __init__(self, a: float = 0.0, b: float = float(jnp.pi)) -> None:
        super().__init__(a, b)
        if a < -1e-9 or b > float(jnp.pi) + 1e-9:
            raise ValueError(
                "SinUniform support must lie within [0, π]; "
                f"got a={a}, b={b}."
            )
        # Normalisation constant Z = ∫ sin θ dθ = cos a - cos b  (> 0).
        self._cos_a = jnp.cos(self.a)
        self._cos_b = jnp.cos(self.b)
        self._logZ = jnp.log(self._cos_a - self._cos_b)

    def sample(self, rng_key: jax.Array) -> jax.Array:
        u = jax.random.uniform(rng_key)
        cos_x = self._cos_a - u * (self._cos_a - self._cos_b)
        # Guard against tiny FP excursions outside [-1, 1].
        return jnp.arccos(jnp.clip(cos_x, -1.0, 1.0))

    def icdf(self, u: jax.Array) -> jax.Array:
        cos_x = self._cos_a - jnp.asarray(u) * (self._cos_a - self._cos_b)
        return jnp.arccos(jnp.clip(cos_x, -1.0, 1.0))

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        logp = jnp.log(jnp.sin(x)) - self._logZ
        return self._mask_support(x, logp)


# ── Cosine prior (isotropic declination / latitude) ───────────────────────────

class CosUniform(PriorSpec):
    r"""Prior with density ``p(δ) ∝ cos δ`` on ``[a, b] ⊆ [-π/2, π/2]``.

    This is the marginal of an *isotropic* sky position for a
    latitude/declination angle ``δ`` measured from the equator. The
    normalised density is

    .. math::
        p(\delta) = \frac{\cos\delta}{\sin b - \sin a}, \qquad a \le \delta \le b.

    Inverse-CDF sampling uses
    ``δ = arcsin(sin a + u (sin b − sin a))`` for ``u ~ U(0, 1)``.
    """

    def __init__(
        self,
        a: float = -float(jnp.pi) / 2,
        b: float = float(jnp.pi) / 2,
    ) -> None:
        super().__init__(a, b)
        half_pi = float(jnp.pi) / 2
        if a < -half_pi - 1e-9 or b > half_pi + 1e-9:
            raise ValueError(
                "CosUniform support must lie within [-π/2, π/2]; "
                f"got a={a}, b={b}."
            )
        # Normalisation constant Z = ∫ cos δ dδ = sin b - sin a  (> 0).
        self._sin_a = jnp.sin(self.a)
        self._sin_b = jnp.sin(self.b)
        self._logZ = jnp.log(self._sin_b - self._sin_a)

    def sample(self, rng_key: jax.Array) -> jax.Array:
        u = jax.random.uniform(rng_key)
        sin_x = self._sin_a + u * (self._sin_b - self._sin_a)
        return jnp.arcsin(jnp.clip(sin_x, -1.0, 1.0))

    def icdf(self, u: jax.Array) -> jax.Array:
        sin_x = self._sin_a + jnp.asarray(u) * (self._sin_b - self._sin_a)
        return jnp.arcsin(jnp.clip(sin_x, -1.0, 1.0))

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        logp = jnp.log(jnp.cos(x)) - self._logZ
        return self._mask_support(x, logp)


# ── Power-law prior (volumetric distance is the n=2 case) ─────────────────────

class PowerLaw(PriorSpec):
    r"""Prior with density ``p(x) ∝ x ** n`` on ``[a, b]``, ``a > 0``.

    The normalised density for ``n ≠ -1`` is

    .. math::
        p(x) = \frac{(n+1)\, x^{n}}{b^{\,n+1} - a^{\,n+1}}.

    Inverse-CDF sampling uses
    ``x = (a^{n+1} + u (b^{n+1} − a^{n+1}))^{1/(n+1)}``.

    The luminosity-distance *uniform-in-comoving-volume (Euclidean)* prior
    ``p(d) ∝ d²`` is the ``n = 2`` special case; see :class:`Volumetric`.
    """

    def __init__(self, a: float, b: float, n: float = 2.0) -> None:
        super().__init__(a, b)
        n = float(n)
        if n == -1.0:
            raise ValueError(
                "PowerLaw with n == -1 is log-uniform; not supported here "
                "(use a dedicated LogUniform if needed)."
            )
        if a <= 0.0:
            raise ValueError(
                f"PowerLaw requires a > 0 (got a={a}); x ** n with x <= 0 is "
                "ill-defined for the distance use-case."
            )
        self.n = n
        self._np1 = n + 1.0
        # Normalisation: Z = (b^{n+1} - a^{n+1}) / (n+1).
        self._a_np1 = self.a ** self._np1
        self._b_np1 = self.b ** self._np1
        self._logZ = jnp.log((self._b_np1 - self._a_np1) / self._np1)

    def sample(self, rng_key: jax.Array) -> jax.Array:
        u = jax.random.uniform(rng_key)
        x_np1 = self._a_np1 + u * (self._b_np1 - self._a_np1)
        return x_np1 ** (1.0 / self._np1)

    def icdf(self, u: jax.Array) -> jax.Array:
        x_np1 = self._a_np1 + jnp.asarray(u) * (self._b_np1 - self._a_np1)
        return x_np1 ** (1.0 / self._np1)

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        logp = self.n * jnp.log(x) - self._logZ
        return self._mask_support(x, logp)


class Volumetric(PowerLaw):
    r"""Uniform-in-volume luminosity-distance prior, ``p(d) ∝ d²``.

    Equivalent to ``PowerLaw(a, b, n=2)`` — the Euclidean / constant-rate
    expectation for sources uniformly distributed in space. Use this for the
    ``distance`` parameter when you want the standard GW PE volumetric prior
    rather than the flat (uniform-in-distance) default.
    """

    def __init__(self, a: float, b: float) -> None:
        super().__init__(a, b, n=2.0)


# ── LVK-standard mass and aligned-spin priors ─────────────────────────────────

class UniformInComponentsChirpMass(PowerLaw):
    r"""Chirp-mass marginal of a prior uniform in the component masses, ``p(M_c) ∝ M_c``.

    Together with :class:`UniformInComponentsMassRatio` this reproduces the
    LVK / bilby default of sampling ``(M_c, q)`` with a prior that is uniform
    in ``(m_1, m_2)`` inside the ``(M_c, q)`` box: the Jacobian
    ``|∂(m_1, m_2)/∂(M_c, q)| ∝ M_c (1+q)^{2/5} q^{-6/5}`` factorises.
    Same density as ``bilby.gw.prior.UniformInComponentsChirpMass``.
    """

    def __init__(self, a: float, b: float) -> None:
        super().__init__(a, b, n=1.0)


class UniformInComponentsMassRatio(PriorSpec):
    r"""Mass-ratio marginal of a prior uniform in component masses, ``q = m_2/m_1 ≤ 1``.

    .. math::
        p(q) \propto (1+q)^{2/5}\, q^{-6/5}, \qquad a \le q \le b \le 1,

    the same density as ``bilby.gw.prior.UniformInComponentsMassRatio``. The
    inverse CDF is tabulated once (float64, 20001 nodes uniform in ``log q``,
    cumulative Simpson-accurate trapezoid) and evaluated with ``jnp.interp``;
    ``log_prob`` is analytic.
    """

    def __init__(self, a: float, b: float = 1.0, n_grid: int = 20001) -> None:
        super().__init__(a, b)
        if a <= 0.0 or b > 1.0 + 1e-12:
            raise ValueError(f"UniformInComponentsMassRatio needs 0 < a < b <= 1, got a={a}, b={b}.")
        x = np.exp(np.linspace(np.log(self.a), np.log(self.b), n_grid))
        pdf = (1.0 + x) ** 0.4 * x ** (-1.2)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))])
        self._logZ = float(np.log(cdf[-1]))
        self._cdf = jnp.asarray(cdf / cdf[-1])
        self._x = jnp.asarray(x)

    def icdf(self, u: jax.Array) -> jax.Array:
        return jnp.interp(jnp.asarray(u), self._cdf, self._x)

    def sample(self, rng_key: jax.Array) -> jax.Array:
        return self.icdf(jax.random.uniform(rng_key))

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        xs = jnp.clip(x, self.a, self.b)
        logp = 0.4 * jnp.log1p(xs) - 1.2 * jnp.log(xs) - self._logZ
        return self._mask_support(x, logp)


class AlignedSpin(PriorSpec):
    r"""z-component of an isotropic spin with magnitude uniform in ``[0, a_max]``.

    .. math::
        p(\chi) = \frac{1}{2 a_{\max}} \ln\frac{a_{\max}}{|\chi|}, \qquad |\chi| \le a_{\max},

    the standard LVK aligned-spin prior (``bilby.gw.prior.AlignedSpin`` with a
    uniform magnitude prior), optionally truncated to ``[a, b]``. ``a_max``
    defaults to ``max(|a|, |b|)``, so ``AlignedSpin(-0.9, 0.9)`` is exactly
    "isotropic spins with |a| ≤ 0.9" (e.g. a surrogate's validity limit).
    The CDF is analytic, ``F(χ) = 1/2 + sgn(χ) G(|χ|)``,
    ``G(x) = x (1 + ln(a_max/x)) / (2 a_max)``; its inverse is tabulated.
    """

    def __init__(self, a: float, b: float, a_max: float | None = None,
                 n_grid: int = 20001) -> None:
        super().__init__(a, b)
        self.a_max = float(a_max) if a_max is not None else max(abs(self.a), abs(self.b))
        if max(abs(self.a), abs(self.b)) > self.a_max + 1e-12:
            raise ValueError(f"AlignedSpin bounds [{a}, {b}] exceed a_max={self.a_max}.")
        A = self.a_max

        def F(c):
            c = np.asarray(c, float)
            ax = np.abs(c)
            g = np.where(ax > 0, ax * (1.0 + np.log(A / np.where(ax > 0, ax, 1.0))) / (2 * A), 0.0)
            return 0.5 + np.sign(c) * g

        # nodes dense near the integrable log-singularity at chi = 0
        s = np.linspace(-1.0, 1.0, n_grid)
        grid = np.sign(s) * np.abs(s) ** 3
        x = self.a + (grid + 1.0) / 2.0 * (self.b - self.a) if self.a >= 0 or self.b <= 0 else \
            np.where(grid < 0, -grid * self.a, grid * self.b)
        x = np.unique(np.concatenate([x, [self.a, self.b]]))
        cdf = F(x) - F(self.a)
        Z = float(F(self.b) - F(self.a))
        self._logZ = float(np.log(Z))
        self._cdf = jnp.asarray(cdf / Z)
        self._x = jnp.asarray(x)

    def icdf(self, u: jax.Array) -> jax.Array:
        return jnp.interp(jnp.asarray(u), self._cdf, self._x)

    def sample(self, rng_key: jax.Array) -> jax.Array:
        return self.icdf(jax.random.uniform(rng_key))

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        ax = jnp.clip(jnp.abs(x), 1e-300, self.a_max)
        logp = jnp.log(jnp.log(self.a_max / ax) / (2.0 * self.a_max)) - self._logZ
        return self._mask_support(x, logp)


# ── String aliases for ergonomic notebook configuration ───────────────────────

#: Maps short string names to the :class:`PriorSpec` subclass built from
#: ``param_bounds``. Used by :func:`resolve_priors`.
PRIOR_ALIASES: Dict[str, type] = {
    "uniform":     Uniform,
    "flat":        Uniform,
    "sin":         SinUniform,
    "sine":        SinUniform,
    "sin_uniform": SinUniform,
    "cos":         CosUniform,
    "cosine":      CosUniform,
    "cos_uniform": CosUniform,
    "volumetric":  Volumetric,
    "volume":      Volumetric,
    "d2":          Volumetric,
    # LVK / bilby-standard choices
    "uniform_in_components_chirp_mass": UniformInComponentsChirpMass,
    "uniform_in_components_mass_ratio": UniformInComponentsMassRatio,
    "aligned_spin":                     AlignedSpin,
}


PriorLike = Union[PriorSpec, str]


def resolve_priors(
    param_bounds: Mapping[str, Tuple[float, float]],
    priors: Mapping[str, PriorLike] | None = None,
) -> Dict[str, PriorSpec]:
    """Build a full ``{name: PriorSpec}`` dict from bounds + optional choices.

    For every parameter in ``param_bounds`` this returns a concrete
    :class:`PriorSpec`:

    * If ``priors[name]`` is a :class:`PriorSpec` instance, it is used as-is
      (its own ``(a, b)`` define the support; ``param_bounds[name]`` is
      ignored for that parameter — they should agree).
    * If ``priors[name]`` is a string (see :data:`PRIOR_ALIASES`, e.g.
      ``"sin"``, ``"cos"``, ``"volumetric"``), a spec of that kind is built
      over ``param_bounds[name]``.
    * Otherwise (parameter absent from ``priors``) it defaults to
      :class:`Uniform` over ``param_bounds[name]`` — i.e. unchanged
      behaviour.

    Parameters
    ----------
    param_bounds : dict[str, (float, float)]
        Bounds for every sampled parameter — the same dict already passed to
        the GWgpu_jax samplers.
    priors : dict[str, PriorSpec | str] or None
        Per-parameter prior choice. Keys must be a subset of
        ``param_bounds``.

    Returns
    -------
    dict[str, PriorSpec]
        One concrete spec per parameter in ``param_bounds`` (insertion order
        preserved).
    """
    priors = dict(priors or {})

    unknown = set(priors) - set(param_bounds)
    if unknown:
        raise ValueError(
            f"`priors` names parameters not in `param_bounds`: {sorted(unknown)}."
        )

    specs: Dict[str, PriorSpec] = {}
    for name, bounds in param_bounds.items():
        choice = priors.get(name, "uniform")
        if isinstance(choice, PriorSpec):
            specs[name] = choice
        elif isinstance(choice, str):
            key = choice.lower()
            if key not in PRIOR_ALIASES:
                raise ValueError(
                    f"Unknown prior alias {choice!r} for parameter {name!r}. "
                    f"Valid aliases: {sorted(PRIOR_ALIASES)}."
                )
            a, b = bounds
            specs[name] = PRIOR_ALIASES[key](float(a), float(b))
        else:
            raise TypeError(
                f"priors[{name!r}] must be a PriorSpec or a string alias, "
                f"got {type(choice).__name__}."
            )
    return specs


# ── blackjax-ns drop-in helpers ───────────────────────────────────────────────

def build_prior(
    prior_specs: Mapping[str, PriorSpec],
) -> Tuple[Callable[[jax.Array], Dict[str, jax.Array]], Callable[[Dict[str, jax.Array]], jax.Array]]:
    """Build the ``(prior_sample, logprior_fn)`` closures for a spec dict.

    Parameters
    ----------
    prior_specs : dict[str, PriorSpec]
        One :class:`PriorSpec` per sampled parameter.

    Returns
    -------
    prior_sample : callable
        ``prior_sample(rng_key) -> dict[str, scalar]`` — a single draw from
        the joint (independent) prior. ``vmap`` it over a batch of keys to
        get live points.
    logprior_fn : callable
        ``logprior_fn(params) -> scalar`` — the summed, normalised
        log-prior density; ``-inf`` outside any parameter's support. This is
        exactly the ``logprior_fn`` consumed by
        ``blackjax.ns.nss.as_top_level_api``.
    """
    specs = dict(prior_specs)

    def logprior_fn(params: Dict[str, jax.Array]) -> jax.Array:
        logp = jnp.asarray(0.0)
        for name, spec in specs.items():
            logp = logp + spec.log_prob(params[name])
        return logp

    def prior_sample(rng_key: jax.Array) -> Dict[str, jax.Array]:
        keys = jax.random.split(rng_key, len(specs))
        return {
            name: spec.sample(k)
            for k, (name, spec) in zip(keys, specs.items())
        }

    return prior_sample, logprior_fn


def sample_prior(
    rng_key: jax.Array,
    num_live: int,
    prior_specs: Mapping[str, PriorSpec],
) -> Tuple[Dict[str, jax.Array], Callable[[Dict[str, jax.Array]], jax.Array]]:
    """Draw ``num_live`` live points and return them with ``logprior_fn``.

    Drop-in replacement for :func:`blackjax.ns.utils.uniform_prior` — same
    ``(rng_key, num_live, spec_dict) -> (particles, logprior_fn)`` signature
    — except ``spec_dict`` maps names to :class:`PriorSpec` objects instead
    of ``(a, b)`` tuples, enabling arbitrary (sin / cos / volumetric / …)
    priors.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX random key.
    num_live : int
        Number of live particles to draw.
    prior_specs : dict[str, PriorSpec]
        One :class:`PriorSpec` per sampled parameter.

    Returns
    -------
    particles : dict[str, jnp.ndarray]
        Each leaf has shape ``(num_live,)``, exactly prior-distributed.
    logprior_fn : callable
        The normalised joint log-prior density consumed by the NS kernel.
    """
    prior_sample, logprior_fn = build_prior(prior_specs)
    keys = jax.random.split(rng_key, num_live)
    particles = jax.vmap(prior_sample)(keys)
    return particles, logprior_fn
