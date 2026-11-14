"""Calibration of the nested samplers on a problem with a known answer.

6-D Gaussian in the unit cube with two coordinates piled up against a prior edge. Both samplers must
recover the analytic evidence and the posterior widths. This catches the bias of an acceptance walk
that stops after a fixed number of ACCEPTED steps (ln Z +2.6, widths x0.90 on this problem; found in
the GW170817 revision runs, Nov 2026) -- gwgpu_jax uses a fixed number of steps, as bilby does.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.stats import norm, truncnorm

jax.config.update("jax_enable_x64", True)
import gwgpu_jax
from gwgpu_jax.gwgpu_jax_acceptance_walk import GWgpu_jaxAcceptanceWalkSampler, import_custom_kernels

D, SIG = 6, 0.03
MU = np.array([0.5, 0.5, 0.5, 0.01, 0.99, 0.5])
NAMES = [f"x{i}" for i in range(D)]
BOUNDS = {n: (0.0, 1.0) for n in NAMES}
TRUE_LOGZ = float(np.sum(np.log(SIG * np.sqrt(2 * np.pi) * (norm.cdf((1 - MU) / SIG) - norm.cdf(-MU / SIG)))))
TRUE_SD = np.array([truncnorm.std(-m / SIG, (1 - m) / SIG, loc=m, scale=SIG) for m in MU])


def logl(p):
    x = jnp.stack([p[n] for n in NAMES])
    return -0.5 * jnp.sum((x - MU) ** 2) / SIG**2


def check(result):
    sd = np.array([np.std(np.asarray(result.posterior_samples[n])) for n in NAMES]) / TRUE_SD
    assert abs(result.logZ - TRUE_LOGZ) < 0.6, f"ln Z {result.logZ:.3f} vs {TRUE_LOGZ:.3f}"
    assert np.all(np.abs(sd - 1.0) < 0.06), f"posterior std / true = {np.round(sd, 3)}"


def test_nss_calibration():
    s = gwgpu_jax.GWgpu_jaxNestedSampler.from_loglikelihood(logl, BOUNDS)
    check(s.run(jax.random.PRNGKey(0), num_live=1000, num_delete=50, num_inner_steps=4 * D,
                log_dlogz_target=float(np.log(0.1)), num_posterior_samples=10000))


def test_acceptance_walk_calibration():
    try:
        import_custom_kernels()
    except ModuleNotFoundError:
        pytest.skip("external acceptance-walk kernel not fetched")
    s = GWgpu_jaxAcceptanceWalkSampler.from_loglikelihood(logl, BOUNDS, periodic_params=())
    check(s.run(jax.random.PRNGKey(0), num_live=1000, num_delete=100, n_target=60, max_mcmc=5000,
                log_dlogz_target=float(np.log(np.expm1(0.1))), num_posterior_samples=10000))
