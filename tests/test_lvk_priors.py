"""Validate the LVK-standard priors (UniformInComponents*, AlignedSpin) against
analytic densities and bilby.  Run:  python -m pytest tests/test_lvk_priors.py -q"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy import stats
from scipy.integrate import quad

from gwgpu_jax import (UniformInComponentsChirpMass, UniformInComponentsMassRatio,
                       AlignedSpin, resolve_priors)

U = np.random.default_rng(0).uniform(size=200_000)

CASES = [UniformInComponentsChirpMass(15.0, 50.0), UniformInComponentsMassRatio(0.1, 1.0),
         AlignedSpin(-0.9, 0.9), AlignedSpin(-0.99, 0.99), AlignedSpin(-0.5, 0.8, a_max=0.99)]


def _pdf(p):
    return lambda x: float(np.exp(p.log_prob(jnp.asarray(x))))


def test_normalised():
    for p in CASES:
        brk = [0.0] if p.a < 0 < p.b else []
        z, _ = quad(_pdf(p), p.a, p.b, points=brk, limit=400)
        assert abs(z - 1) < 1e-6, (p, z)


def test_icdf_matches_density_ks():
    for p in CASES:
        x = np.asarray(p.icdf(jnp.asarray(U)))
        assert x.min() >= p.a - 1e-12 and x.max() <= p.b + 1e-12
        brk = [0.0] if p.a < 0 < p.b else []
        grid = np.linspace(p.a, p.b, 401)
        cdf_grid = np.concatenate([[0.0], np.cumsum(
            [quad(_pdf(p), lo, hi, points=[b for b in brk if lo < b < hi] or None)[0]
             for lo, hi in zip(grid[:-1], grid[1:])])])
        cdf = lambda v: np.interp(v, grid, cdf_grid)
        assert stats.kstest(x[:20000], cdf).pvalue > 1e-3, p


def test_outside_support_is_minus_inf():
    for p in CASES:
        assert np.isneginf(float(p.log_prob(p.a - 1e-3)))
        assert np.isneginf(float(p.log_prob(p.b + 1e-3)))


def test_joint_mass_prior_uniform_in_components():
    mc = np.asarray(UniformInComponentsChirpMass(20.0, 40.0).icdf(jnp.asarray(U)))
    q = np.asarray(UniformInComponentsMassRatio(0.2, 1.0).icdf(
        jnp.asarray(np.random.default_rng(1).uniform(size=U.size))))
    m1 = mc * (1 + q) ** 0.2 / q ** 0.6
    m2 = q * m1
    # inside a box fully contained in the (mc, q) support, (m1, m2) must be flat
    sel = (m1 > 30) & (m1 < 40) & (m2 > 25) & (m2 < 30)
    h, _, _ = np.histogram2d(m1[sel], m2[sel], bins=5)
    assert stats.chisquare(h.ravel()).pvalue > 1e-3


def test_aliases_resolve():
    specs = resolve_priors({"mc": (15, 50), "q": (0.1, 1), "chi_1": (-0.9, 0.9)},
                           {"mc": "uniform_in_components_chirp_mass",
                            "q": "uniform_in_components_mass_ratio", "chi_1": "aligned_spin"})
    assert isinstance(specs["chi_1"], AlignedSpin) and specs["chi_1"].a_max == 0.9


def test_against_bilby():
    try:
        import bilby
    except ImportError:
        import pytest; pytest.skip("bilby not installed")
    from bilby.gw.prior import (UniformInComponentsChirpMass as BMc,
                                UniformInComponentsMassRatio as Bq, AlignedSpin as BAS)
    from bilby.core.prior import Uniform
    pairs = [(UniformInComponentsChirpMass(15, 50), BMc(15, 50), np.linspace(15.5, 49.5, 50)),
             (UniformInComponentsMassRatio(0.1, 1), Bq(0.1, 1), np.linspace(0.11, 0.99, 50)),
             (AlignedSpin(-0.99, 0.99), BAS(a_prior=Uniform(0, 0.99)), np.linspace(-0.98, 0.98, 50))]
    for ours, theirs, xs in pairs:
        xs = xs[np.abs(xs) > 1e-3]
        np.testing.assert_allclose(np.asarray(ours.log_prob(jnp.asarray(xs))),
                                   theirs.ln_prob(xs), atol=1e-5)
