#!/usr/bin/env python
"""gwgpu_jax acceptance-walk PE on the frozen mlgw_bbh injection.

GPU half of the mlgw_bbh vs bilby benchmark. Consumes the frozen FD strain +
PSD arrays written by `generate_injection.py` (checksum-verified below) and
recovers with the SAME mlgw_bbh_jax (SEOBNRv5HM, (2,2)-mode) waveform used to
make the injection — same PARAM_BOUNDS / PRIORS / sampler settings as
`examples/NEWParameter_Estimation_mlgw_bbh_GW150914_acceptance_walk_VSCode.ipynb`,
just pointed at this frozen data instead of a real GWOSC fetch.

Output: <outdir>/{posterior_samples.npz,result.json} — same schema as
`frozen_benchmark/run_gwjax_frozen.py` — feed both to make_comparison_plot.py
together with the bilby result.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", default=None, help="frozen data dir (default: ./data next to this script)")
parser.add_argument("--outdir", default="gwjax_bbh_case")
parser.add_argument("--nlive", type=int, default=2000)
parser.add_argument("--n-target", type=int, default=200)
parser.add_argument("--max-mcmc", type=int, default=1200)
parser.add_argument("--num-delete", type=int, default=50)
parser.add_argument("--num-posterior", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-iters", type=int, default=10000,
                    help="cap on NS steps (lower it only for smoke tests)")
args, _unknown = parser.parse_known_args()

DATA = Path(args.data) if args.data else Path(__file__).resolve().parent / "data"
meta = json.load(open(DATA / "metadata.json"))


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


freqs_np = np.load(DATA / "frequency_array.npy")
for key, fname in [("frequency_array", "frequency_array.npy")] + [
        (f"{d}_{k}", f"{d}_{k}.npy") for d in meta["detectors"] for k in ("strain_fd", "psd")]:
    got = _sha(np.load(DATA / fname))
    want = meta["sha256_16"][key]
    if got != want:
        raise RuntimeError(f"checksum mismatch for {fname}: {got} != {want} "
                           "— data dir does not match the frozen injection metadata")
print("frozen mlgw_bbh injection verified against checksums.")

# ── gwjax setup ───────────────────────────────────────────────────────────────
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import gwgpu_jax
from gwgpu_jax.gwgpu_jax_compatibility import gmst_from_gps
from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator

grid = gwgpu_jax.TimeFrequencyGrid(
    meta["duration"], meta["sampling_frequency"],
    f_min=meta["minimum_frequency"], f_max=meta["maximum_frequency"])
assert np.array_equal(np.asarray(grid.frequency_domain_array), freqs_np)

net = gwgpu_jax.Network.from_names(meta["detectors"], grid)
for ifo in net.interferometers:
    ifo.load_data(jnp.asarray(np.load(DATA / f"{ifo.name}_strain_fd.npy")), domain="fd")
    ifo.psd = jnp.asarray(np.load(DATA / f"{ifo.name}_psd.npy"))
    ifo._weight = None
net.gmst = gmst_from_gps(meta["trigger_time"])

# ── mlgw_bbh (SEOBNRv5HM, (2,2) mode) TD -> FD waveform ─────────────────────
# Recipe identical to generate_injection.py / the real-GW150914 notebook.
# Must equal the value the injection was generated with (merger at tc - final_time).
# Older frozen data predates the key and was made with 0.01 s.
MLGW_FINAL_TIME = float(meta.get("mlgw_final_time", 0.01))
print(f"mlgw_final_time = {MLGW_FINAL_TIME} s")
TUKEY_ALPHA_LEFT, TUKEY_ALPHA_RIGHT = 0.2, 0.001


def _tukey_asymmetric(n, alpha_left, alpha_right):
    x = jnp.linspace(0.0, 1.0, n)
    w = jnp.ones_like(x)
    left = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * (x / alpha_left - 0.5)))
    right = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * ((x - 1.0) / alpha_right + 0.5)))
    w = jnp.where(x < alpha_left / 2.0, left, w)
    w = jnp.where(x > 1.0 - alpha_right / 2.0, right, w)
    return w


def build_mlgw_bbh_waveform_fn(grid, modes=(2, 2),
                                final_time=MLGW_FINAL_TIME,
                                alpha_left=TUKEY_ALPHA_LEFT,
                                alpha_right=TUKEY_ALPHA_RIGHT):
    bbh = MLGWBBHGenerator(model=4)
    dt, N = grid.dt, grid.n_samples
    t_grid = jnp.arange(N) * dt - grid.duration + final_time
    window = _tukey_asymmetric(N, alpha_left, alpha_right)
    gen_wf = bbh.generate

    def waveform_fn(params, freqs):
        mc, q = params["mc"], params["q"]
        m1 = mc * (1.0 + q) ** (1.0 / 5.0) / q ** (3.0 / 5.0)
        m2 = q * m1
        theta = jnp.stack([
            m1, m2, params["chi_1"], params["chi_2"],
            params["distance"], params["inclination"], params["phi_c"],
        ])
        hp_td, hc_td = gen_wf(theta, t_grid, modes=modes)
        hp_fd = jnp.fft.rfft(hp_td * window) * dt
        hc_fd = jnp.fft.rfft(hc_td * window) * dt
        return hp_fd, -hc_fd

    return waveform_fn


t0 = time.perf_counter()
waveform_fn = build_mlgw_bbh_waveform_fn(grid, modes=(2, 2))
print(f"mlgw_bbh_jax (SEOBNRv5HM, model_4, (2,2) mode) loaded in {time.perf_counter()-t0:.1f} s")

# ── Priors — identical to the real-GW150914 notebook, centred on this
#    injection's mc/q/tc instead of GW150914's ─────────────────────────────
TC_CENTER = meta["tc_in_segment"] + MLGW_FINAL_TIME   # mlgw tc that puts the merger on the trigger
inj = meta["injection"]
PARAM_BOUNDS = {
    "mc":          (inj["chirp_mass"] - 2.0, inj["chirp_mass"] + 2.0),
    "q":           (0.6, 1.0),
    "chi_1":       (-0.4, 0.4),
    "chi_2":       (-0.4, 0.4),
    "distance":    (50.0, 300.0),
    "inclination": (0.0, float(jnp.pi)),
    "ra":          (0.0, 2.0 * float(jnp.pi)),
    "dec":         (-float(jnp.pi) / 2, float(jnp.pi) / 2),
    "psi":         (0.0, float(jnp.pi)),
    "phi_c":       (0.0, 2.0 * float(jnp.pi)),
    "tc":          (TC_CENTER - 0.1, TC_CENTER + 0.1),
}
PRIORS = {
    "inclination": "sin",
    "dec":         "cos",
    "distance":    "volumetric",
}

sampler = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler(
    network=net, waveform_fn=waveform_fn,
    param_bounds=PARAM_BOUNDS, priors=PRIORS, gmst=None,
    periodic_params=("phi_c", "ra"),
)
print(f"sampling dimension : {len(sampler.param_bounds)}")
print(f"free parameters    : {list(sampler.param_bounds)}")

t0 = time.perf_counter()
result = sampler.run(
    rng_key               = jax.random.PRNGKey(args.seed),
    num_live              = args.nlive,
    n_target              = args.n_target,
    max_mcmc              = args.max_mcmc,
    num_delete            = args.num_delete,
    max_iterations        = args.max_iters,
    log_dlogz_target      = -3,
    num_posterior_samples = args.num_posterior,
    verbose               = True,
)
elapsed = time.perf_counter() - t0

noise_logl = float(sum(i.log_likelihood_data(jnp.zeros_like(i.strain_data_fd))
                       for i in net.interferometers))

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
post = {k: np.asarray(v) for k, v in result.posterior_samples.items()}
np.savez_compressed(outdir / "posterior_samples.npz", **post)
json.dump(dict(
    logZ=float(result.logZ), logZ_err=float(result.logZ_err),
    log_noise_evidence=noise_logl,
    log_bayes_factor=float(result.logZ) - noise_logl,
    ess=float(result.ess), n_iterations=int(result.n_iterations),
    elapsed_s=elapsed, start_time=meta["start_time"],
    trigger_time=meta["trigger_time"], injection=meta["injection"],
    config=dict(nlive=args.nlive, num_delete=args.num_delete, n_target=args.n_target,
                max_mcmc=args.max_mcmc, log_dlogz_target=-3, seed=args.seed,
                mlgw_final_time=MLGW_FINAL_TIME,
                waveform="mlgw_bbh_jax (SEOBNRv5HM, model_4, (2,2) mode)"),
), open(outdir / "result.json", "w"), indent=2)

print(f"\nlogZ             = {result.logZ:.4f} +/- {result.logZ_err:.4f}")
print(f"log_bayes_factor = {result.logZ - noise_logl:.4f}")
print(f"ESS = {result.ess:.0f}   steps = {result.n_iterations}   "
      f"wall time = {elapsed/3600:.2f} h")
print(f"Saved {outdir}/posterior_samples.npz and {outdir}/result.json")
