#!/usr/bin/env python
"""Sampler-independent consistency check for a GW170817 gwjax run (run from the repo root).

Evaluates the gwjax tc/phi-marginalised RB likelihood (CPU) on the gwjax and bilby posterior samples and
re-estimates ln BF from each sample set with a Gelfand-Dey estimator (truncated-Gaussian reference density).
A correct run reproduces its own nested-sampling evidence. Nov 2026, run 20261114_112633 (nlive 1000,
num_delete 500): bilby samples -> 490.2 (= bilby NS 490.20), gwjax samples -> 489.4-489.9 vs gwjax NS 494.04,
i.e. the likelihoods agree and the gwjax NS run was over-concentrated. Paths to the run are hard-coded below.
(Assembled from the notebook cells of COLAB_GW170817_mlgw_bns_LVKpriors.ipynb.)
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import time, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import gwgpu_jax
print("gwgpu_jax version :", gwgpu_jax.__version__)
print("JAX devices       :", jax.devices())

PSD_MODE  = "gwtc1"     # "gwtc1": LVK GWTC-1 BayesWave PSDs (default)  |  "welch": the published run's PSD
DATA_ROOT = "examples/referee_response/gw170817_mlgw_bns"

DATA = Path(DATA_ROOT) / f"data_{PSD_MODE}"
meta = json.load(open(DATA / "metadata.json"))

def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

arrays = {"frequency_array": np.load(DATA / "frequency_array.npy")}
for det in meta["detectors"]:
    arrays[f"{det}_strain_fd"] = np.load(DATA / f"{det}_strain_fd.npy")
    arrays[f"{det}_psd"] = np.load(DATA / f"{det}_psd.npy")
bad = [k for k, a in arrays.items() if _sha(a) != meta["sha256_16"][k]]
assert not bad, f"checksum mismatch for {bad}: re-copy {DATA}"

grid = gwgpu_jax.TimeFrequencyGrid(meta["duration"], meta["sampling_frequency"],
                                   f_min=meta["minimum_frequency"], f_max=meta["maximum_frequency"])
assert np.array_equal(np.asarray(grid.frequency_domain_array), arrays["frequency_array"])
network = gwgpu_jax.Network.from_names(meta["detectors"], grid)
for ifo in network.interferometers:
    ifo.load_data(jnp.asarray(arrays[f"{ifo.name}_strain_fd"]), domain="fd")
    ifo.psd = jnp.asarray(arrays[f"{ifo.name}_psd"])
    ifo._weight = None
network.gmst = meta["gmst"]
print(f"verified {DATA.name}: {meta['psd_method']}")
print(f"  {meta['detectors']}, {meta['duration']:g} s @ {meta['sampling_frequency']:g} Hz, "
      f"{meta['minimum_frequency']:g}-{meta['maximum_frequency']:g} Hz, merger at {meta['tc_in_segment']:.3f} s")
print("  whitened power (expect ~1):", {d: [round(x, 2) for x in meta["whitened_power_median"][d]] for d in meta["detectors"]})

from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator

_bns_gen = MLGWBNSGenerator()
_predict = _bns_gen.predict   # raw JAX callable (jit/vmap-safe)

def mc_q_to_m1_m2(mc, q):
    """(Mc, q=m2/m1 ∈ (0,1]) → (m1, m2) with m1 ≥ m2."""
    m1 = mc * (1.0 + q) ** (1.0 / 5.0) / q ** (3.0 / 5.0)
    m2 = q * m1
    return m1, m2

def waveform_fn(params, freqs):
    """GWgpu_jax-compatible mlgw_bns_jax waveform with phi_c as a uniform phase."""
    m1, m2     = mc_q_to_m1_m2(params["mc"], params["q"])
    total_mass = m1 + m2
    q_mlgw     = m1 / m2                                    # ≥ 1 for the surrogate
    mlgw_p     = jnp.stack([
        q_mlgw,
        params["lambda_1"], params["lambda_2"],
        params["chi_1"],    params["chi_2"],
    ])
    hp, hc = _predict(
        mlgw_p, freqs,
        total_mass   = total_mass,
        distance_mpc = params["distance"],
        inclination  = params["inclination"],
    )
    phase = jnp.exp(-1j * params["phi_c"])
    return hp * phase, hc * phase

# Sanity smoke-test on the analysis grid (cell completes ≤ 1 minute on first JIT).
_smoke = waveform_fn(
    {"mc": 1.1975, "q": 0.85,
     "chi_1": 0.0, "chi_2": 0.0,
     "lambda_1": 300.0, "lambda_2": 300.0,
     "distance": 40.0, "inclination": 2.5,
     "phi_c": 0.0},
    grid.frequency_domain_array,
)
print("hp/hc shapes:", _smoke[0].shape, _smoke[1].shape)
print("max |hp|, |hc|:", float(jnp.max(jnp.abs(_smoke[0]))),
      float(jnp.max(jnp.abs(_smoke[1]))))

from gwgpu_jax.gwgpu_jax_relative_binning import (
    build_rb_likelihood_tc_phi_marg, build_full_likelihood_tc_phi_marg, compute_matched_filter_snr)

NGC4993_RA, NGC4993_DEC = 3.44616, -0.408084
TC0 = meta["tc_in_segment"]
FIDUCIAL = {"mc": 1.1975, "q": 0.85, "chi_1": 0.0, "chi_2": 0.0, "lambda_1": 250.0, "lambda_2": 250.0,
            "distance": 37.0, "inclination": 2.545, "ra": NGC4993_RA, "dec": NGC4993_DEC, "psi": 0.0,
            "tc": TC0, "phi_c": 0.0}
TC_HALF_WIDTH, TC_SPACING = 0.01, 1e-5          # ±10 ms (published run: ±1 ms, too narrow), 10 µs
TC_GRID = TC0 + TC_SPACING * np.arange(-round(TC_HALF_WIDTH / TC_SPACING), round(TC_HALF_WIDTH / TC_SPACING) + 1)
N_BINS = 1600

t0 = time.perf_counter()
log_L_tp_marg, rb = build_rb_likelihood_tc_phi_marg(
    network=network, waveform_fn=waveform_fn, fiducial_params=FIDUCIAL, tc_grid=TC_GRID,
    n_bins=N_BINS, gmst=network.gmst, verbose=True)
T_RB_SETUP = time.perf_counter() - t0
print(f"RB built in {T_RB_SETUP:.1f} s")
snr = compute_matched_filter_snr(network, waveform_fn, FIDUCIAL, gmst=network.gmst)
print("optimal SNR at the fiducial:", {k: round(float(v), 2) for k, v in snr["per_ifo"].items()},
      f"network {float(snr['network']):.2f}")

SPIN_PRIOR = "low"      # "low": |a| <= 0.05 (GWTC-1 low-spin)  |  "high": |a| <= 0.5 (surrogate limit)
A_MAX = {"low": 0.05, "high": 0.5}[SPIN_PRIOR]
PI = float(np.pi)
PARAM_BOUNDS = {
    "mc":          (1.18, 1.21),
    "q":           (0.34, 1.0),
    "chi_1":       (-A_MAX, A_MAX),
    "chi_2":       (-A_MAX, A_MAX),
    "lambda_1":    (5.0, 5000.0),
    "lambda_2":    (5.0, 5000.0),
    "distance":    (1.0, 75.0),
    "inclination": (0.0, PI),
    "psi":         (0.0, PI),
    "ra":          (0.0, 2.0 * PI),
    "dec":         (-PI / 2, PI / 2),
}
PRIORS = {
    "mc":          "uniform_in_components_chirp_mass",
    "q":           "uniform_in_components_mass_ratio",
    "chi_1":       "aligned_spin",
    "chi_2":       "aligned_spin",
    "distance":    "volumetric",
    "inclination": "sin",
    "dec":         "cos",
}
PERIODIC = ("ra", "psi")
from gwgpu_jax import resolve_priors
print({k: type(v).__name__ for k, v in resolve_priors(PARAM_BOUNDS, PRIORS).items()})


from scipy.stats import chi2
specs = resolve_priors(PARAM_BOUNDS, PRIORS); names = list(PARAM_BOUNDS)
noise = float(sum(i.log_likelihood_data(jnp.zeros_like(i.strain_data_fd)) for i in network.interferometers))
vll = jax.jit(jax.vmap(log_L_tp_marg))
def lnL(S):
    out = []
    for s in range(0, len(S[names[0]]), 250):
        out.append(np.asarray(vll({k: jnp.asarray(S[k][s:s+250]) for k in names})))
        print(".", end="", flush=True)
    return np.concatenate(out) - noise
def lnprior(S):
    return sum(np.asarray(specs[k].log_prob(jnp.asarray(S[k]))) for k in names)
G = np.load("examples/referee_response/gwjax_runs/20261114_112633_gwtc1_lowspin/posterior_samples.npz")
Sg = {k: np.asarray(G[k]) for k in names}
B = json.load(open("examples/referee_response/gw170817_mlgw_bns/bilby_gw170817_TEOBResumSPA_gwtc1_lowspin/gw170817_TEOBResumSPA_result.json"))["posterior"]["content"]
mp = dict(mc="chirp_mass", q="mass_ratio", distance="luminosity_distance", inclination="theta_jn")
Sb = {k: np.asarray(B[mp.get(k, k)], float) for k in names}
rng = np.random.default_rng(1)
res = {}
for tag, S in (("gwjax", Sg), ("bilby", Sb)):
    X = np.column_stack([S[k] for k in names]); n, d = X.shape
    T = lnL(S); lp = lnprior(S)
    print(f"\n{tag}: n={n}  lnL(gwjax tc/phi-marg, ratio vs noise): mean {T.mean():.3f}  median {np.median(T):.3f}  std {T.std():.3f}  max {T.max():.3f}; finite prior {np.isfinite(lp).mean():.3f}")
    res[tag] = dict(T=T, lp=lp)
    mu, C = X.mean(0), np.cov(X.T); Ci = np.linalg.inv(C); Lc = np.linalg.cholesky(C)
    m2 = np.einsum("ij,jk,ik->i", X - mu, Ci, X - mu)
    lo = np.array([PARAM_BOUNDS[k][0] for k in names]); hi = np.array([PARAM_BOUNDS[k][1] for k in names])
    for frac in (0.2, 0.5, 0.8):
        r2 = chi2.ppf(frac, d)
        # mass of the truncated Gaussian that lies inside the prior box (Monte Carlo)
        Z_ = rng.standard_normal((400000, d)); Z_ = Z_[np.sum(Z_**2, 1) <= r2]
        Y = mu + Z_ @ Lc.T; f_in = np.mean(np.all((Y >= lo) & (Y <= hi), 1))
        logq = -0.5 * m2 - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(C)[1] - np.log(frac * f_in)
        sel = m2 <= r2
        terms = np.where(sel, logq - T - lp, -np.inf)
        est = []
        for b in range(200):                      # bootstrap error
            ii = rng.integers(0, n, n); t = terms[ii]; mx = t.max()
            est.append(-(mx + np.log(np.mean(np.exp(t - mx)))))
        t = terms; mx = t.max(); lnZ = -(mx + np.log(np.mean(np.exp(t - mx))))
        print(f"   Gelfand-Dey lnBF (ellipsoid {frac:.0%}, {sel.mean():.2f} of samples inside, box frac {f_in:.3f}): {lnZ:.3f} +- {np.std(est):.3f}")
np.savez('/private/tmp/claude-501/-Users-ut-Documents-GitHub-GWjax/521970a6-d8b1-4abd-94db-9585f2a02f6c/scratchpad/wfcheck/gd.npz', Tg=res["gwjax"]["T"], Tb=res["bilby"]["T"])
print("NS-reported lnBF: gwjax 494.04 +- 0.17, bilby 490.20 +- 0.19")
