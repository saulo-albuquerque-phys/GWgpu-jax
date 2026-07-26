#!/usr/bin/env python
"""gwgpu_jax acceptance-walk run on the frozen 4s benchmark data.

GPU half of the red/blue benchmark. Consumes byte-identical inputs to
`run_bilby_frozen.py` (frozen FD strain + PSD arrays in `data/`) and runs the
external acceptance-walk kernel (mrosep/blackjax_ns_gw) with the reference
paper's Fig.-1 configuration:

  nlive = 1400, num_delete = 700  (effective 1000 live points, paper Sec 3.2.1)
  n_target = 60, max_mcmc = 5000
  termination: bilby's dlogZ < 0.1  <=>  logZ_live - logZ < ln(e^0.1 - 1)
  ripple IMRPhenomD, f_ref = 50 Hz, sampled in (Mc, q)
  priors: Table 1 (power-law-2 distance on [100, 5000] Mpc, spins U[-1, 1])

If `data/` is missing (e.g. a fresh Colab runtime), the frozen inputs are
rebuilt from the public reference repo (mrosep/blackjax_ns_gw@main) and
verified against the recorded checksums.

Output: <outdir>/posterior_samples.npz + <outdir>/result.json — feed both to
make_comparison_plot.py together with the bilby result.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", default=None, help="frozen data dir (default: ./data next to this script)")
parser.add_argument("--outdir", default="gwjax_frozen_4s")
parser.add_argument("--nlive", type=int, default=1400)
parser.add_argument("--num-delete", type=int, default=700)
parser.add_argument("--num-posterior", type=int, default=8000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-iters", type=int, default=20000,
                    help="cap on NS steps (lower it only for smoke tests)")
args, _unknown = parser.parse_known_args()

DATA = Path(args.data) if args.data else Path(__file__).resolve().parent / "data"

# ── Frozen-data provenance: expected sha256[:16] of the raw array bytes ──────
SHA = {
    "frequency_array": "280a86ac9211e670",
    "H1_strain_fd": "5fd731e51f6ee531", "H1_psd": "ca101fe79c42a09b",
    "L1_strain_fd": "62af05c25aad3901", "L1_psd": "ca101fe79c42a09b",
    "V1_strain_fd": "8439b06785024793", "V1_psd": "aed559f6b5dd4edb",
}
_RAW = "https://raw.githubusercontent.com/mrosep/blackjax_ns_gw/main/src/"
_ASD = {"H1": "aLIGO_O4_high_asd.txt", "L1": "aLIGO_O4_high_asd.txt", "V1": "AdV_asd.txt"}


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def rebuild_frozen_data(data_dir: Path) -> None:
    """Rebuild data/ from the public reference repo (same recipe as freezing)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    def fetch(name):
        dest = data_dir / ("_src_" + name)
        if not dest.exists():
            print("downloading", _RAW + name)
            urllib.request.urlretrieve(_RAW + name, dest)
        return dest
    freqs = np.load(fetch("4s_frequency_array.npy"))
    np.save(data_dir / "frequency_array.npy", freqs)
    for det in ["H1", "L1", "V1"]:
        s = np.load(fetch(f"4s_{det}_strain.npy"))
        fa, a = np.loadtxt(fetch(_ASD[det]), unpack=True)
        np.save(data_dir / f"{det}_strain_fd.npy", s)
        np.save(data_dir / f"{det}_psd.npy", np.interp(freqs, fa, a**2))
    meta = dict(duration=4.0, sampling_frequency=2048.0,
                minimum_frequency=20.0, maximum_frequency=1024.0,
                start_time=1126259640.413, trigger_time=1126259642.413,
                reference_frequency=50.0, detectors=["H1", "L1", "V1"],
                injection=dict(chirp_mass=35.0, mass_ratio=0.9, chi_1=0.4,
                               chi_2=-0.3, luminosity_distance=1000.0,
                               theta_jn=0.4, psi=2.659, phase=1.3,
                               geocent_time=1126259642.413, ra=1.375, dec=-1.2108),
                sha256_16=SHA, source="rebuilt from mrosep/blackjax_ns_gw@main")
    json.dump(meta, open(data_dir / "metadata.json", "w"), indent=2)


if not (DATA / "metadata.json").exists():
    rebuild_frozen_data(DATA)
meta = json.load(open(DATA / "metadata.json"))

freqs_np = np.load(DATA / "frequency_array.npy")
for key, fname in [("frequency_array", "frequency_array.npy")] + [
        (f"{d}_{k}", f"{d}_{k}.npy") for d in meta["detectors"] for k in ("strain_fd", "psd")]:
    got = _sha(np.load(DATA / fname))
    if got != SHA[key]:
        raise RuntimeError(f"checksum mismatch for {fname}: {got} != {SHA[key]} "
                           "— data dir is not the frozen benchmark set")
print("frozen data verified against checksums.")

# ── gwjax setup ───────────────────────────────────────────────────────────────
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import gwgpu_jax
from gwgpu_jax.gwgpu_jax_compatibility import gmst_from_gps

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

_base_wf = gwgpu_jax.build_ripplegw_waveform_fn(
    "IMRPhenomD", f_ref=meta["reference_frequency"])


def waveform_fn(params, freqs):
    """(Mc, q) -> (m1, m2) wrapper around the ripple IMRPhenomD builder."""
    Mc, q = params["Mc"], params["q"]
    eta = q / (1.0 + q) ** 2
    Mtot = Mc * eta ** (-3.0 / 5.0)
    p = dict(params)
    p["m1"], p["m2"] = Mtot / (1.0 + q), Mtot * q / (1.0 + q)
    return _base_wf(p, freqs)


# tc is measured from the segment start: bilby geocent_time = start_time + tc.
TC0 = meta["trigger_time"] - meta["start_time"]      # = 2.0 s
PARAM_BOUNDS = {
    "Mc":          (25.0, 50.0),
    "q":           (0.25, 1.0),
    "chi_1":       (-1.0, 1.0),
    "chi_2":       (-1.0, 1.0),
    "distance":    (100.0, 5000.0),
    "inclination": (0.0, float(np.pi)),
    "ra":          (0.0, 2.0 * float(np.pi)),
    "dec":         (-float(np.pi) / 2, float(np.pi) / 2),
    "psi":         (0.0, float(np.pi)),
    "phi_c":       (0.0, 2.0 * float(np.pi)),
    "tc":          (TC0 - 0.1, TC0 + 0.1),
}
PRIORS = {"inclination": "sin", "dec": "cos", "distance": "volumetric"}

sampler = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler(
    network=net, waveform_fn=waveform_fn,
    param_bounds=PARAM_BOUNDS, priors=PRIORS, gmst=None,
    periodic_params=("phi_c", "ra", "psi"),
)

# bilby's termination dlogZ = log(1 + Z_live/Z) < 0.1
# <=> logZ_live - logZ < ln(e^0.1 - 1) — the Fig.-1 criterion.
LOG_DLOGZ_TARGET = float(np.log(np.expm1(0.1)))

t0 = time.perf_counter()
result = sampler.run(
    rng_key               = jax.random.PRNGKey(args.seed),
    num_live              = args.nlive,
    n_target              = 60,
    max_mcmc              = 5000,
    num_delete            = args.num_delete,
    max_iterations        = args.max_iters,
    log_dlogz_target      = LOG_DLOGZ_TARGET,
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
    config=dict(nlive=args.nlive, num_delete=args.num_delete, n_target=60,
                max_mcmc=5000, log_dlogz_target=LOG_DLOGZ_TARGET,
                seed=args.seed, f_ref=meta["reference_frequency"]),
), open(outdir / "result.json", "w"), indent=2)

print(f"\nlogZ             = {result.logZ:.4f} +/- {result.logZ_err:.4f}")
print(f"log_bayes_factor = {result.logZ - noise_logl:.4f}")
print(f"ESS = {result.ess:.0f}   steps = {result.n_iterations}   "
      f"wall time = {elapsed/3600:.2f} h")
print(f"Saved {outdir}/posterior_samples.npz and {outdir}/result.json")
