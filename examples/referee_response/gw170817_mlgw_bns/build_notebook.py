#!/usr/bin/env python
"""Generate COLAB_GW170817_mlgw_bns_LVKpriors.ipynb (the GPU side of the GW170817 revision run).

    python build_notebook.py

The analysis cells (waveform, RB likelihood, sampler) are the published notebook's
(examples/public_Parameter_Estimation_MLGW_BNS_JAX_GW170817_tc95_acceptance_walk.ipynb); the data,
priors, sampler budget, timing and output cells are new.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "COLAB_GW170817_mlgw_bns_LVKpriors.ipynb"
SRC = HERE.parents[1] / "public_Parameter_Estimation_MLGW_BNS_JAX_GW170817_tc95_acceptance_walk.ipynb"
src_cells = json.load(open(SRC))["cells"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def src_code(pred):
    hits = [c for c in src_cells if c["cell_type"] == "code" and pred("".join(c["source"]))]
    assert len(hits) == 1, len(hits)
    return code("".join(hits[0]["source"]))


cells = []
cells.append(md(
    '<a href="https://colab.research.google.com/github/saulo-albuquerque-phys/GWgpu-jax/blob/'
    'acceptance_walk_sampler/examples/referee_response/gw170817_mlgw_bns/COLAB_GW170817_mlgw_bns_LVKpriors.ipynb"'
    ' target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'))

cells.append(md(r"""
# GW170817 — mlgw_bns_jax + relative binning + acceptance-walk NS, **LVK-standard priors** (paper revision)

Revision of `examples/public_Parameter_Estimation_MLGW_BNS_JAX_GW170817_tc95_acceptance_walk.ipynb` for the
referee's PE comments. Same surrogate (mlgw_bns_jax, a TEOBResumSPA surrogate), same tc/φ_c-marginalised
relative-binning likelihood and sampler; differences:

| | published run | this notebook |
|---|---|---|
| data conditioning | gwpy `highpass(23 Hz)` before cropping — **Chebyshev-I, 2 dB ripple, filtfilt: in-band gain 0.63–1.0**, data whitened to 0.5–0.6 at 40–200 Hz (fixed in gwgpu_jax, Nov 2026) | no time-domain filter (bilby/LVK convention); whitened power 0.96–1.08 in every band |
| PSD (comment 1d) | median Welch over the full 1024 s BWCLEANED file (128 s segments, 15 averages, includes the event) | `PSD_MODE = "gwtc1"`: the **GWTC-1 GW170817 PSDs** released by the LVK (BayesWave on-source, DCC P1900011) — or `"welch"` (the published PSD) |
| band (comment 1e) | 23–1500 Hz | 23–1500 Hz (GWOSC 4 kHz files are anti-alias filtered above ~1650 Hz, so GWTC-1's 2048 Hz is not usable with them) |
| M_c (detector frame) | U(1.19, 1.20) | uniform in components, [1.18, 1.21] |
| q = m2/m1 | U(0.68, 1) | uniform in components, [0.34, 1] (surrogate limit m1/m2 ≤ 3) |
| χ₁, χ₂ | U(−0.15, 0.15) | LVK **AlignedSpin**, \|a\| ≤ 0.05 (GWTC-1 low-spin; `SPIN_PRIOR = "high"`: 0.5, surrogate limit) |
| Λ₁, Λ₂ (comment 1c) | U(5, 1200) | **U(5, 5000)** (GWTC-1: 0–5000; surrogate trained on 5–5000) |
| d_L | ∝ d² on [1, 50] Mpc | ∝ d² on [1, 75] Mpc |
| sampler | nlive 2000, n_target 80, maxmcmc 1500, num_delete 150 | nlive 1000, n_target 60, maxmcmc 5000, num_delete 500, stop ≡ bilby dlogz < 0.1 (= the bilby reference run) |

The bilby reference (`run_bilby_gw170817.py`, TEOBResumSPA, same frozen data, priors and settings) runs on
LDG; `compare_gwjax_bilby.sh` turns the two into the JSD table. Section 9 is the run-time breakdown asked
in comment 6. Not changed: (2,2) mode, aligned spins, no calibration marginalisation.

**Before running:** copy `data_gwtc1/` (and `data_welch/` if you want the PSD comparison), produced by
`freeze_gw170817_data.py`, to `MyDrive/gwjax_referee/gw170817_data/`.
"""))

cells.append(md("## 1. Install GWgpu_jax"))
cells.append(src_code(lambda s: "RUNNING_ON_COLAB" in s and "pip install" in s))
cells.append(code(r"""
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
"""))

cells.append(md(r"""
## 2. Frozen GW170817 data (byte-identical to the bilby run)

`freeze_gw170817_data.py` builds the H1+L1+V1 segment from the BayesWave-deglitched 4 kHz strain (128 s,
merger at 0.95·T, Tukey 0.2 s, **no time-domain filter**) and writes FD strain + PSD with sha256 checksums;
both this notebook and `run_bilby_gw170817.py` verify them.
"""))
cells.append(code(r"""
PSD_MODE  = "gwtc1"     # "gwtc1": LVK GWTC-1 BayesWave PSDs (default)  |  "welch": the published run's PSD
DATA_ROOT = "/content/drive/MyDrive/gwjax_referee/gw170817_data"

from google.colab import drive
if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")
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
"""))

cells.append(md("## 3. mlgw_bns_jax waveform (unchanged)"))
cells.append(src_code(lambda s: "MLGWBNSGenerator()" in s and "def waveform_fn" in s))

cells.append(md(r"""
## 4. tc + φ_c-marginalised relative-binning likelihood

Fiducial point and RB bins as in the paper; **the tc window is widened from ±1 ms to ±10 ms** (2001 points,
10 µs spacing). For a BNS the merger time is extrapolated from the inspiral and is correlated with Λ̃
(r ≈ −0.6 in the bilby reference run): the published ±1 ms grid truncated the tc posterior and with it the
high-Λ̃ tail. Validated on the frozen data at bilby posterior points (incl. Λ̃ ≈ 800–1000): RB vs brute
force on the ±10 ms grid |Δln L| ≤ 0.004; 10 µs vs 2.5 µs spacing differs by a constant 4e-4 (grid
normalisation only). The bilby run samples geocent_time over a window of the same width placed at the
same position relative to its own likelihood peak.
"""))
cells.append(code(r"""
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
"""))

cells.append(md(r"""
## 5. RB accuracy check (now over the wider Λ prior)

RB vs the full-grid tc/φ_c-marginalised likelihood at the fiducial and at offsets inside the expected
posterior (same offsets as the bilby RB check in `run_bilby_gw170817.py --benchmark`), plus a high-Λ point.
Both are evaluated on every 4th point of `TC_GRID` (the full-grid variant stores an N_tc × N_band phase
matrix: 501 × 189k fits a T4, 2001 × 189k would need 6 GB).
"""))
cells.append(code(r"""
log_L_jit = jax.jit(log_L_tp_marg)
TC_CHECK = TC_GRID[::4]
_ll_rb_check, _ = build_rb_likelihood_tc_phi_marg(
    network=network, waveform_fn=waveform_fn, fiducial_params=FIDUCIAL, tc_grid=TC_CHECK,
    n_bins=N_BINS, gmst=network.gmst, verbose=False)
log_L_rb_check = jax.jit(_ll_rb_check)
log_L_full_tp = jax.jit(build_full_likelihood_tc_phi_marg(
    network=network, waveform_fn=waveform_fn, fiducial_params=FIDUCIAL, tc_grid=TC_CHECK, gmst=network.gmst))
fid11 = {k: FIDUCIAL[k] for k in ("mc", "q", "chi_1", "chi_2", "lambda_1", "lambda_2",
                                  "distance", "inclination", "psi", "ra", "dec")}
t0 = time.perf_counter(); _ = float(log_L_jit(fid11)); T_JIT_RB = time.perf_counter() - t0
t0 = time.perf_counter(); _ = float(log_L_full_tp(fid11)); T_JIT_FULL = time.perf_counter() - t0
print(f"JIT compile + first call: RB {T_JIT_RB:.1f} s, full {T_JIT_FULL:.1f} s")
OFFSETS = [{}, dict(mc=1.19755), dict(mc=1.1974, q=0.75), dict(lambda_1=300.0, lambda_2=900.0),
           dict(lambda_1=1500.0, lambda_2=2500.0), dict(chi_1=0.02, chi_2=-0.02),
           dict(distance=30.0, inclination=2.2), dict(ra=3.42, dec=-0.38, psi=1.0)]
rb_check = []
for o in OFFSETS:
    p = {**fid11, **o}
    a, b = float(log_L_rb_check(p)), float(log_L_full_tp(p))
    rb_check.append(dict(offset=o, rb=a, full=b, diff=a - b))
    print(f"{str(o):45s} RB {a:10.3f}  full {b:10.3f}  diff {a - b:+.4f}")
"""))

cells.append(md(r"""
## 6. LVK-standard priors (referee comment 1c)

11 sampled parameters (tc, φ_c marginalised). Masses uniform in the component masses inside the
(M_c, q) box, LVK AlignedSpin spins, Λ uniform to 5000, isotropic orientation and sky, distance ∝ d².
"""))
cells.append(code(r"""
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
"""))
cells.append(src_code(lambda s: "blackjax_ns_gw" in s and "custom_kernels" in s))

cells.append(md(r"""
## 7. Run the acceptance-walk sampler

Same settings as the bilby reference: 1000 live points, 60 accepted steps per new point (maxmcmc
5000), stop at bilby's `dlogz < 0.1` ≡ `ln Z_live − ln Z < ln(e^0.1 − 1)`. gwjax replaces 500 points
per (batched) iteration where dynesty replaces one.
"""))
cells.append(code(r"""
NUM_LIVE, NUM_DELETE, N_TARGET, MAX_MCMC = 1000, 500, 60, 5000
LOG_DLOGZ_TARGET = float(np.log(np.expm1(0.1)))
SEED = 0
sampler = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler.from_loglikelihood(
    loglikelihood_fn=log_L_tp_marg, param_bounds=PARAM_BOUNDS, priors=PRIORS, periodic_params=PERIODIC)
t0 = time.perf_counter()
result = sampler.run(
    rng_key=jax.random.PRNGKey(SEED), num_live=NUM_LIVE, n_target=N_TARGET, max_mcmc=MAX_MCMC,
    num_delete=NUM_DELETE, max_iterations=20000, log_dlogz_target=LOG_DLOGZ_TARGET,
    num_posterior_samples=10000, verbose=True)
elapsed = time.perf_counter() - t0
print(f"\nwall-clock {elapsed:.1f} s ({elapsed/60:.1f} min), {result.n_iterations} iterations, "
      f"log Z = {result.logZ:+.3f} ± {result.logZ_err:.3f}, ESS = {result.ess:.0f}")
"""))

cells.append(md("## 8. Save to Google Drive"))
cells.append(code(r"""
import datetime, pandas as pd
RUN_DIR = (f"/content/drive/MyDrive/gwjax_referee/gw170817_mlgw_bns_lvkpriors/"
           f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{PSD_MODE}_{SPIN_PRIOR}spin")
os.makedirs(RUN_DIR, exist_ok=True)
post = {k: np.asarray(v) for k, v in result.posterior_samples.items()}
m1 = post["mc"] * (1 + post["q"]) ** 0.2 / post["q"] ** 0.6
m2 = post["q"] * m1
post["mass_1"], post["mass_2"] = m1, m2
post["lambda_tilde"] = 16 / 13 * ((m1 + 12 * m2) * m1**4 * post["lambda_1"]
                                  + (m2 + 12 * m1) * m2**4 * post["lambda_2"]) / (m1 + m2) ** 5
post["chi_eff"] = (m1 * post["chi_1"] + m2 * post["chi_2"]) / (m1 + m2)
np.savez_compressed(f"{RUN_DIR}/posterior_samples.npz", **post)
pd.DataFrame(post).to_csv(f"{RUN_DIR}/posterior_samples.txt", sep="\t", index=False)
noise_logl = float(sum(i.log_likelihood_data(jnp.zeros_like(i.strain_data_fd)) for i in network.interferometers))
json.dump(dict(
    logZ=float(result.logZ), logZ_err=float(result.logZ_err), log_noise_evidence=noise_logl,
    log_bayes_factor=float(result.logZ) - noise_logl, ess=float(result.ess),
    n_iterations=int(result.n_iterations), elapsed_s=elapsed, rb_check=rb_check,
    config=dict(event="GW170817", data=DATA.name, psd_mode=PSD_MODE, psd_method=meta["psd_method"],
                sha256_16=meta["sha256_16"], start_time=meta["start_time"], trigger_time=meta["trigger_time"],
                tc_in_segment=TC0, tc_half_width=TC_HALF_WIDTH, n_tc=len(TC_GRID), n_bins=N_BINS,
                f_min=meta["minimum_frequency"], f_max=meta["maximum_frequency"],
                waveform="mlgw_bns_jax default (TEOBResumSPA surrogate), (2,2)",
                likelihood="relative binning, tc + phi_c marginalised", fiducial=FIDUCIAL, tc_spacing=TC_SPACING,
                param_bounds=PARAM_BOUNDS, priors=PRIORS, periodic=list(PERIODIC), spin_prior=SPIN_PRIOR,
                sampler="acceptance-walk NS", num_live=NUM_LIVE, num_delete=NUM_DELETE, n_target=N_TARGET,
                max_mcmc=MAX_MCMC, log_dlogz_target=LOG_DLOGZ_TARGET, seed=SEED,
                gwgpu_jax_version=getattr(gwgpu_jax, "__version__", "?"), device=str(jax.devices()[0]))),
    open(f"{RUN_DIR}/result.json", "w"), indent=2, default=str)
print("saved to", RUN_DIR)
"""))

cells.append(md(r"""
## 9. Run-time breakdown (referee comment 6)

Where the wall-clock time goes: (a) the mlgw_bns_jax waveform alone, at the RB bin centres and on the
full band; (b) one RB tc/φ_c-marginalised likelihood; (c) the full-grid tc/φ_c-marginalised likelihood
(what the run would cost without RB); (d) the number of likelihood calls the sampler made (kernel
counter). Batched numbers are per evaluation in a vmapped batch, as the sampler evaluates them.
CPU reference: `run_bilby_gw170817.py --benchmark` (TEOBResumSPA, bilby RB) and the bilby run's log.
"""))
cells.append(code(r"""
def _bench(fn, arg, n=10):
    jax.block_until_ready(fn(arg))                       # compile
    t = time.perf_counter()
    for _ in range(n):
        out = fn(arg)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / n

def _count_evals(tree):
    tot, stack, seen = 0, [tree], set()
    while stack:
        x = stack.pop()
        if id(x) in seen or x is None:
            continue
        seen.add(id(x))
        if hasattr(x, "_fields"):
            for f in x._fields:
                v = getattr(x, f)
                if f == "n_likelihood_evals":
                    tot += int(np.sum(np.asarray(v)))
                else:
                    stack.append(v)
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, dict):
            stack.extend(x.values())
    return tot or None

B = 1000
P = {k: jnp.asarray(post[k][:B]) for k in PARAM_BOUNDS}
Pw = {**P, "tc": jnp.full(B, TC0), "phi_c": jnp.zeros(B)}
f_band = jnp.asarray(np.asarray(grid.frequency_domain_array)[np.asarray(grid.frequency_mask)])
Bf = 32
Pw_f = {k: v[:Bf] for k, v in Pw.items()}
t_wf_bins = _bench(jax.jit(jax.vmap(lambda p: waveform_fn(p, rb.f_bins))), Pw) / B
t_wf_full = _bench(jax.jit(jax.vmap(lambda p: waveform_fn(p, f_band))), Pw_f) / Bf
t_rb      = _bench(jax.jit(jax.vmap(log_L_tp_marg)), P) / B
t_full    = _bench(log_L_full_tp, {k: v[0] for k, v in P.items()}, n=5)
n_evals   = _count_evals(result.dead_info)
timing = dict(device=str(jax.devices()[0]), batch=B, n_rb_bins=int(len(rb.f_bins)), n_band_bins=int(f_band.size),
              waveform_rb_bins_s=t_wf_bins, waveform_full_band_s=t_wf_full,
              rb_likelihood_s=t_rb, full_likelihood_tc_phi_marg_s=t_full,
              jit_first_call_rb_s=T_JIT_RB, jit_first_call_full_s=T_JIT_FULL, rb_setup_s=T_RB_SETUP,
              sampler_wall_s=elapsed, n_iterations=int(result.n_iterations), n_likelihood_evals=n_evals)
if n_evals:
    timing["wall_per_likelihood_eval_s"] = elapsed / n_evals
    timing["projected_wall_without_rb_s"] = n_evals * t_full
json.dump(timing, open(f"{RUN_DIR}/timing.json", "w"), indent=2)
print(f"device {timing['device']}   ({timing['n_rb_bins']} RB bins vs {timing['n_band_bins']} band bins)")
print(f"  waveform at RB bins        {1e6*t_wf_bins:9.2f} µs / eval (batch {B})")
print(f"  waveform on full band      {1e6*t_wf_full:9.2f} µs / eval (batch {Bf})")
print(f"  RB tc/φ-marg likelihood    {1e6*t_rb:9.2f} µs / eval (batch {B})  -> waveform share {t_wf_bins/t_rb:.0%}")
print(f"  full tc/φ-marg likelihood  {1e3*t_full:9.2f} ms / eval (single, {len(TC_CHECK)}-point tc grid)")
print(f"  JIT compile + first call:  RB {T_JIT_RB:.1f} s, full {T_JIT_FULL:.1f} s; RB setup {T_RB_SETUP:.1f} s")
if n_evals:
    print(f"  sampler: {n_evals:.3g} likelihood calls in {elapsed:.0f} s -> {1e6*elapsed/n_evals:.1f} µs / call "
          f"effective; without RB ≈ {n_evals*t_full/3600:.1f} h")
"""))

cells.append(md("## 10. Posterior summary and corner plot"))
cells.append(code(r"""
!pip install -q corner
import corner
show = ["mc", "q", "chi_eff", "lambda_tilde", "lambda_1", "lambda_2", "distance", "inclination", "ra", "dec"]
for n in show:
    lo, mid, hi = np.percentile(post[n], [5, 50, 95])
    print(f"{n:13s} {mid:10.4f}  -{mid-lo:.4f} +{hi-mid:.4f}  (90%)")
X = np.column_stack([post[n] for n in show])
fig = corner.corner(X, labels=show, quantiles=[0.05, 0.5, 0.95], show_titles=True,
                    title_kwargs={"fontsize": 9}, plot_datapoints=False, smooth=0.9)
fig.savefig(f"{RUN_DIR}/corner.png", dpi=110, bbox_inches="tight"); plt.show()
"""))

cells.append(md(r"""
## 11. Comparison with GWTC-1 (visual + JSD)

GWTC-1 `IMRPhenomPv2NRT_lowSpin` (DCC P1800370). **Caveat:** precessing PhenomPv2_NRTidal vs aligned-spin
TEOBResumSPA surrogate, different sampler, calibration marginalisation, and the sky fixed to NGC 4993
(so ra/dec are not compared) — this mixes those effects; the
surrogate/sampler-only comparison is the bilby TEOBResumSPA run (`compare_gwjax_bilby.sh`).
"""))
cells.append(code(r"""
import h5py, urllib.request
from scipy.stats import gaussian_kde
G1 = "GW170817_GWTC-1.hdf5"
if not os.path.exists(G1):
    urllib.request.urlretrieve("https://dcc.ligo.org/public/0157/P1800370/005/GW170817_GWTC-1.hdf5", G1)
with h5py.File(G1, "r") as f:
    g = f[{"low": "IMRPhenomPv2NRT_lowSpin_posterior", "high": "IMRPhenomPv2NRT_highSpin_posterior"}[SPIN_PRIOR]][()]
gm1, gm2 = g["m1_detector_frame_Msun"], g["m2_detector_frame_Msun"]
ref = dict(mc=(gm1 * gm2) ** 0.6 / (gm1 + gm2) ** 0.2, q=gm2 / gm1,
           chi_eff=(gm1 * g["spin1"] * g["costilt1"] + gm2 * g["spin2"] * g["costilt2"]) / (gm1 + gm2),
           lambda_tilde=16 / 13 * ((gm1 + 12 * gm2) * gm1**4 * g["lambda1"] + (gm2 + 12 * gm1) * gm2**4 * g["lambda2"]) / (gm1 + gm2) ** 5,
           distance=g["luminosity_distance_Mpc"], inclination=np.arccos(g["costheta_jn"]))
# (GWTC-1 fixed the sky to NGC 4993 for GW170817: ra/dec are constant there, so they are not compared)

def _pdf(x, xs, bounds=None, period=None):
    k = gaussian_kde(x); p = k(xs)
    if period: p += k(xs - period) + k(xs + period)
    elif bounds:
        if bounds[0] is not None: p += k(2 * bounds[0] - xs)
        if bounds[1] is not None: p += k(2 * bounds[1] - xs)
    p = np.clip(p, 0, None); return p / np.trapz(p, xs)

def jsd_bits(x, y, bounds=None, period=None, n=2000):
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); span = hi - lo
    lo, hi = lo - 0.1 * span, hi + 0.1 * span
    if period: lo, hi = 0.0, period
    elif bounds:
        lo = max(lo, bounds[0]) if bounds[0] is not None else lo
        hi = min(hi, bounds[1]) if bounds[1] is not None else hi
    xs = np.linspace(lo, hi, n); p, q = _pdf(x, xs, bounds, period), _pdf(y, xs, bounds, period); m = 0.5 * (p + q)
    kl = lambda a, b: np.trapz(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / b), 0), xs)
    return float(np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0, 1))

BND = dict(q=(None, 1.0), lambda_tilde=(0.0, None), inclination=(0.0, PI), distance=(0.0, None))
rng = np.random.default_rng(0); table = {}
print(f"{'param':13s} {'JSD [bits]':>10s} {'90% CI':>19s}  {'floor':>7s}")
for n in ref:
    kw = dict(period=2 * PI) if n == "ra" else dict(bounds=BND.get(n))
    x, y = post[n] if n != "ra" else np.mod(post[n], 2 * PI), np.asarray(ref[n])
    ns = min(3000, len(x), len(y))
    bs = [jsd_bits(rng.choice(x, ns, False), rng.choice(y, ns, False), **kw) for _ in range(40)]
    perm = rng.permutation(len(x)); h = min(3000, len(x) // 2)
    floor = jsd_bits(x[perm[:h]], x[perm[h:2 * h]], **kw)
    lo, med, hi = np.percentile(bs, [5, 50, 95])
    table[n] = dict(jsd_median=med, jsd_5=lo, jsd_95=hi, floor=floor)
    print(f"{n:13s} {med:10.4f} [{lo:8.4f},{hi:8.4f}]  {floor:7.4f}")
json.dump(table, open(f"{RUN_DIR}/jsd_vs_gwtc1.json", "w"), indent=2)

names = ["mc", "q", "chi_eff", "lambda_tilde", "distance", "inclination"]
Y = np.column_stack([np.asarray(ref[n]) for n in names])
rngs = [(min(np.percentile(post[n], 0.5), np.percentile(ref[n], 0.5)), max(np.percentile(post[n], 99.5), np.percentile(ref[n], 99.5))) for n in names]
fig = corner.corner(Y, labels=names, range=rngs, color="C0", plot_datapoints=False, smooth=1.0,
                    hist_kwargs=dict(density=True), weights=np.full(len(Y), 1 / len(Y)))
corner.corner(np.column_stack([post[n] for n in names]), fig=fig, range=rngs, color="C3", plot_datapoints=False,
              smooth=1.0, hist_kwargs=dict(density=True), weights=np.full(len(post["mc"]), 1 / len(post["mc"])))
fig.legend(handles=[plt.Line2D([], [], color="C0", label="GWTC-1 PhenomPv2NRT"),
                    plt.Line2D([], [], color="C3", label="this run (mlgw_bns_jax)")], loc="upper right")
fig.savefig(f"{RUN_DIR}/corner_vs_gwtc1.png", dpi=110, bbox_inches="tight"); plt.show()
"""))

cells.append(md(r"""
## 12. Next: JSD against the bilby TEOBResumSPA reference

Download `RUN_DIR` from Drive, then on a machine with this repo:

    bash examples/referee_response/gw170817_mlgw_bns/compare_gwjax_bilby.sh <RUN_DIR> <bilby ..._result.json>
"""))

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": [],
                                                  "include_colab_link": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}}}
json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"wrote {OUT.name}: {len(cells)} cells")
