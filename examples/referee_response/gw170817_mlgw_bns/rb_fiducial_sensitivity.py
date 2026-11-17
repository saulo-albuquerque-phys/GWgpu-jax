#!/usr/bin/env python
"""Sensitivity of the GW170817 relative-binning (RB) likelihood to the choice of fiducial waveform
(referee 2, minor comment 3).

For several fiducials -- the production one and deliberately offset ones -- builds the RB likelihood
marginalised over tc and phi_c, and compares it with the exact full-band likelihood marginalised over
the same tc grid, at posterior samples of the production run.

A constant offset between the two likelihoods does not change the posterior; what matters is how much
Delta = lnL_RB - lnL_full varies across the posterior. Reported per fiducial:
  * max |Delta - median(Delta)| and std(Delta) over the samples;
  * the reweighting efficiency ESS/N of weights exp(Delta - max) -- the fraction of the RB posterior
    that survives when it is reweighted to the exact likelihood (1 = identical posteriors);
  * the largest shift of a posterior median after reweighting, in units of the posterior std.

    python rb_fiducial_sensitivity.py RUN_DIR [--data data_gwtc1] [--n 200] [--n-tc 51]

RUN_DIR holds posterior_samples.npz of a gwjax GW170817 run. A coarser tc grid than production
(default 51 points over the same +-10 ms) keeps the full-band phase table small (~150 MB); RB and
full band use the same grid, so the comparison is like for like.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir", type=Path)
ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data_gwtc1")
ap.add_argument("--n", type=int, default=200, help="number of posterior samples")
ap.add_argument("--n-tc", type=int, default=51, help="tc grid points over +-10 ms")
ap.add_argument("--n-bins", type=int, default=1600, help="RB bins requested (production: 1600)")
ap.add_argument("--out", default="rb_fiducial_sensitivity.json")
a = ap.parse_args()

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import gwgpu_jax
from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator
from gwgpu_jax.gwgpu_jax_relative_binning import build_rb_likelihood_tc_phi_marg, build_full_likelihood_tc_phi_marg

meta = json.load(open(a.data / "metadata.json"))
grid = gwgpu_jax.TimeFrequencyGrid(meta["duration"], meta["sampling_frequency"],
                                   f_min=meta["minimum_frequency"], f_max=meta["maximum_frequency"])
network = gwgpu_jax.Network.from_names(meta["detectors"], grid)
for ifo in network.interferometers:
    ifo.load_data(jnp.asarray(np.load(a.data / f"{ifo.name}_strain_fd.npy")), domain="fd")
    ifo.psd = jnp.asarray(np.load(a.data / f"{ifo.name}_psd.npy")); ifo._weight = None
network.gmst = meta["gmst"]
predict = MLGWBNSGenerator().predict


def waveform_fn(p, freqs):                           # identical to the GW170817 notebook
    m1 = p["mc"] * (1.0 + p["q"]) ** 0.2 / p["q"] ** 0.6; m2 = p["q"] * m1
    hp, hc = predict(jnp.stack([m1 / m2, p["lambda_1"], p["lambda_2"], p["chi_1"], p["chi_2"]]), freqs,
                     total_mass=m1 + m2, distance_mpc=p["distance"], inclination=p["inclination"])
    ph = jnp.exp(-1j * p["phi_c"])
    return hp * ph, hc * ph


TC0 = meta["tc_in_segment"]
TC_GRID = TC0 + np.linspace(-0.01, 0.01, a.n_tc)
NOMINAL = {"mc": 1.1975, "q": 0.85, "chi_1": 0.0, "chi_2": 0.0, "lambda_1": 250.0, "lambda_2": 250.0,
           "distance": 37.0, "inclination": 2.545, "ra": 3.44616, "dec": -0.408084, "psi": 0.0, "tc": TC0, "phi_c": 0.0}
K11 = ["mc", "q", "chi_1", "chi_2", "lambda_1", "lambda_2", "distance", "inclination", "psi", "ra", "dec"]

post = np.load(a.run_dir / "posterior_samples.npz")
rng = np.random.default_rng(1)
idx = rng.choice(len(post["mc"]), a.n, replace=False)
S = {k: np.asarray(post[k])[idx] for k in K11}
std = {k: float(np.std(post[k])) for k in K11}

# fiducials: production + offsets of several posterior standard deviations (sigma_Mc ~ 7e-5 Msun)
FIDUCIALS = {
    "production": {},
    "Mc +3 sigma": {"mc": NOMINAL["mc"] + 3 * std["mc"]},
    "Mc -3 sigma": {"mc": NOMINAL["mc"] - 3 * std["mc"]},
    "q = 0.65": {"q": 0.65},
    "Lambda1,2 = 1000": {"lambda_1": 1000.0, "lambda_2": 1000.0},
    "chi1,2 = +0.03": {"chi_1": 0.03, "chi_2": 0.03},
    "sky +5 deg": {"ra": NOMINAL["ra"] + np.radians(5), "dec": NOMINAL["dec"] + np.radians(5)},
    "all of Mc +3 sigma, q, Lambda, chi": {"mc": NOMINAL["mc"] + 3 * std["mc"], "q": 0.65,
                                          "lambda_1": 1000.0, "lambda_2": 1000.0, "chi_1": 0.03, "chi_2": 0.03},
}

print(f"{a.n} posterior samples, tc grid {a.n_tc} points over +-10 ms", flush=True)
t0 = time.perf_counter()
full = jax.jit(build_full_likelihood_tc_phi_marg(network=network, waveform_fn=waveform_fn, fiducial_params=NOMINAL,
                                                 tc_grid=TC_GRID, gmst=network.gmst))
L_full = np.array([float(full({k: jnp.asarray(S[k][i]) for k in K11})) for i in range(a.n)])
print(f"full-band likelihood at {a.n} samples: {time.perf_counter() - t0:.0f} s", flush=True)


def ess_fraction(d):
    w = np.exp(d - d.max()); w /= w.sum()
    return float(1.0 / np.sum(w**2) / len(w))


res = {}
for name, off in FIDUCIALS.items():
    fid = dict(NOMINAL, **off)
    ll, rb = build_rb_likelihood_tc_phi_marg(network=network, waveform_fn=waveform_fn, fiducial_params=fid,
                                             tc_grid=TC_GRID, n_bins=a.n_bins, gmst=network.gmst, verbose=False)
    L_rb = np.asarray(jax.jit(jax.vmap(ll))({k: jnp.asarray(S[k]) for k in K11}))
    d = L_rb - L_full
    dc = d - np.median(d)
    w = np.exp(-(d - d.max())); w /= w.sum()          # reweight RB samples to the exact likelihood
    shift = max(abs(float(np.sum(w * S[k]) - np.mean(S[k]))) / std[k] for k in K11)
    res[name] = dict(fiducial={k: float(v) for k, v in fid.items()}, n_bins=int(len(rb.f_bins)),
                     max_abs_dev=float(np.max(np.abs(dc))), std=float(np.std(d)), median_offset=float(np.median(d)),
                     ess_fraction=ess_fraction(-d), max_mean_shift_sigma=shift)
    r = res[name]
    print(f"{name:36s} bins {r['n_bins']:5d}  std(dlnL) {r['std']:.4f}  max|dlnL-med| {r['max_abs_dev']:.4f}  "
          f"ESS/N {r['ess_fraction']:.4f}  max mean shift {r['max_mean_shift_sigma']:.3f} sigma", flush=True)

json.dump(dict(run=str(a.run_dir), n=a.n, n_tc=a.n_tc, results=res), open(a.out, "w"), indent=2)
print("wrote", a.out)
