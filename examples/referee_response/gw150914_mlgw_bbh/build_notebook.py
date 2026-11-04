#!/usr/bin/env python
"""Generate COLAB_GW150914_mlgw_bbh_LVKpriors.ipynb from the public GW150914 notebook.

Re-run after changing the public notebook so the two stay in sync:
    python build_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "public_Parameter_Estimation_mlgw_bbh_GW150914_acceptance_walk.ipynb"
OUT = HERE / "COLAB_GW150914_mlgw_bbh_LVKpriors.ipynb"

nb = json.load(open(SRC))
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def replace_once(i, a, b):
    s = src(i)
    assert s.count(a) == 1, (i, a[:70])
    cells[i]["source"] = s.replace(a, b).splitlines(True)


# ── locate cells by content (robust to small upstream edits) ─────────────────
def find(pred, kind="code"):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == kind and pred("".join(c["source"]))]
    assert len(hits) == 1, (pred, hits)
    return hits[0]

i_title = find(lambda s: s.startswith("# GWgpu_jax"), "markdown")
i_badge = find(lambda s: "colab.research.google.com" in s, "markdown")
i_psd = find(lambda s: "attach_event_to_network(" in s)
i_wf = find(lambda s: "MLGW_FINAL_TIME =" in s and "def build_mlgw_bbh_waveform_fn" in s)
i_prior_md = find(lambda s: s.startswith("## 5."), "markdown")
i_prior = find(lambda s: "PARAM_BOUNDS = {" in s)
i_run_md = find(lambda s: s.startswith("## 6."), "markdown")
i_run = find(lambda s: "result = sampler.run(" in s)
i_time = find(lambda s: s.startswith("minutes = elapsed"))
i_corner_md = find(lambda s: s.startswith("## 8. Corner"), "markdown")
i_gwtc_load = find(lambda s: s.startswith("from pesummary.io import read") and "samples_dict" not in s)
gwtc_plot = [i for i, c in enumerate(cells) if c["cell_type"] == "code"
             and "samples_dict['C01:Mixed']" in "".join(c["source"])]

# ── title / badge ────────────────────────────────────────────────────────────
cells[i_badge] = md('<a href="https://colab.research.google.com/github/saulo-albuquerque-phys/GWgpu-jax/blob/'
                    'acceptance_walk_sampler/examples/referee_response/gw150914_mlgw_bbh/'
                    'COLAB_GW150914_mlgw_bbh_LVKpriors.ipynb" target="_parent">'
                    '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>')
cells[i_title] = md("""
# GW150914 — mlgw_bbh_jax (SEOBNRv5HM) + acceptance-walk NS, **LVK-standard priors** (paper revision)

Revision of `examples/public_Parameter_Estimation_mlgw_bbh_GW150914_acceptance_walk.ipynb` that addresses
the referee's PE comments. Same data, likelihood and surrogate; differences:

| | published run | this notebook |
|---|---|---|
| post-merger span on the mlgw TD grid | 0.01 s (**truncated ringdown**, mismatch ≈ 5.6e-3 vs SEOBNRv5HM) | 0.12 s (whole ringdown, ≈1e-4) |
| M_c [M☉] | U(26, 32) | uniform in components, [15, 50] (lower edge: 20 Hz inspiral must fit the 4 s segment) |
| q = m2/m1 | U(0.6, 1) | uniform in components, [0.1, 1] (surrogate limit q ≤ 10) |
| χ₁, χ₂ | U(−0.4, 0.4) | LVK **AlignedSpin**: isotropic spins, \\|a\\| ≤ 0.9 (surrogate limit) |
| d_L [Mpc] | ∝ d² on [150, 700] | ∝ d² on [50, 2000] |
| PSD | off-source median Welch, 32 s ending 8 s before trigger, 4 s Hann segments (15 periodograms) | `PSD_MODE`: `"lvk_welch"` 128 s, 4 s segments (63 periodograms, default) or `"paper"` |
| sampler | n_target 100, maxmcmc 1200, num_delete 100, stop Δln Z < e⁻³ | n_target 60, maxmcmc 5000, num_delete 1000, stop ≡ bilby dlogz < 0.1 |

Sampler: **acceptance-walk nested sampling** (bilby/dynesty kernel of Prathaban et al. 2025, validated
against bilby at the JSD noise floor) — not NSS. Not changed here: (2,2) mode only (as the paper), no
calibration marginalisation, Euclidean d² rather than uniform-in-comoving-volume distance prior.

Results are written to Google Drive (`RUN_DIR`): posterior samples (`.npz` + `.txt`), `result.json` with
the full configuration, and the GWTC-2.1 JS-divergence table.
""")

# ── PSD switch ───────────────────────────────────────────────────────────────
cells[i_psd] = code('''
import time

# Referee comment 1(d). Both modes are OFF-SOURCE median Welch estimates (Hann, 50% overlap) from
# data ending PSD_OFFSET s before the trigger — the on-source 4 s segment is NOT included.
#   "lvk_welch": 128 s, 4 s segments -> 63 periodograms (off-source convention: 32 x segment length,
#                fftlength = analysis duration, so the PSD lives exactly on the analysis grid)
#   "paper"    :  32 s, 4 s segments -> 15 periodograms (the published run; N//8 default)
PSD_MODE   = "lvk_welch"
PSD_OFFSET = 8.0
PSD_SEGMENT_DURATION = {"lvk_welch": 128.0, "paper": 32.0}[PSD_MODE]
PSD_NPERSEG = {"lvk_welch": int(grid.n_samples), "paper": None}[PSD_MODE]

t0 = time.perf_counter()
gwgpu_jax.compat.attach_event_to_network(
    network, "GW150914",
    estimate_psd          = True,
    psd_segment_duration  = PSD_SEGMENT_DURATION,
    psd_offset            = PSD_OFFSET,
    psd_nperseg           = PSD_NPERSEG,
    verbose               = False,
)
print(f"GW150914 strain + PSDs attached in {time.perf_counter()-t0:.1f} s")
print(f"  populated detectors : {[ifo.name for ifo in network.interferometers]}")
print(f"  PSD_MODE={PSD_MODE}: off-source Welch window = [t_trig - {PSD_OFFSET + PSD_SEGMENT_DURATION:.0f} s, "
      f"t_trig - {PSD_OFFSET:.0f} s]; analysis segment = [t_trig - {0.875*grid.duration:.1f} s, "
      f"t_trig + {0.125*grid.duration:.1f} s]")
''')

# ── waveform: whole ringdown for the wide prior ──────────────────────────────
replace_once(i_wf,
    "MLGW_FINAL_TIME = 0.1    # s of post-merger time kept (whole ringdown; 0.01 truncated it, MM~3e-3)",
    "MLGW_FINAL_TIME = 0.12   # s of post-merger time kept: covers mlgw's native end (0.00052 s/Msun x M_tot)\n"
    "                         # up to M_tot ~ 230 Msun, i.e. the whole wide prior. 0.01 s truncated the ringdown.")

# ── priors ───────────────────────────────────────────────────────────────────
cells[i_prior_md] = md("""
## 5. LVK-standard priors + acceptance-walk nested sampler

11 sampled parameters. Masses are sampled in `(M_c, q = m2/m1 ≤ 1)` with the LVK/bilby prior that is
**uniform in the component masses** inside the box (`UniformInComponentsChirpMass` × `UniformInComponentsMassRatio`,
identical densities to bilby — see `tests/test_lvk_priors.py`). Spins use the LVK **AlignedSpin** prior
(z-projection of isotropic spins with magnitude uniform in [0, 0.9]); 0.9 and q ≥ 0.1 are the SEOBNRv5HM
surrogate's training limits. Orientation and sky are isotropic, distance uniform in Euclidean volume.
""")
cells[i_prior] = code('''
from gwgpu_jax import UniformInComponentsChirpMass, UniformInComponentsMassRatio, AlignedSpin  # needs gwgpu_jax >= Nov 2026

# mlgw's template merger sits at tc - MLGW_FINAL_TIME, so centre the tc window on
# (trigger position in the segment) + MLGW_FINAL_TIME.
TC_CENTER = 0.875 * grid.duration + MLGW_FINAL_TIME
PI = float(jnp.pi)

PARAM_BOUNDS = {
    "mc":          (15.0, 50.0),
    "q":           (0.1,  1.0),
    "chi_1":       (-0.9, 0.9),
    "chi_2":       (-0.9, 0.9),
    "distance":    (50.0, 2000.0),
    "inclination": (0.0,  PI),
    "ra":          (0.0,  2.0 * PI),
    "dec":         (-PI / 2, PI / 2),
    "psi":         (0.0,  PI),
    "phi_c":       (0.0,  2.0 * PI),
    "tc":          (TC_CENTER - 0.1, TC_CENTER + 0.1),
}
FIXED_PARAMS = {}
PRIORS = {
    "mc":          "uniform_in_components_chirp_mass",   # p(Mc) ∝ Mc
    "q":           "uniform_in_components_mass_ratio",   # p(q)  ∝ (1+q)^(2/5) q^(-6/5)
    "chi_1":       "aligned_spin",                       # p(χ)  ∝ ln(0.9/|χ|)
    "chi_2":       "aligned_spin",
    "inclination": "sin",
    "dec":         "cos",
    "distance":    "volumetric",
}

# GWTC-1 medians, visual reference only (not a truth)
PUBLISHED_MEDIAN = dict(mc=28.71, q=0.860, chi_1=0.0, chi_2=0.0, distance=440.0, inclination=2.5,
                        ra=2.21, dec=-1.25, psi=0.0, phi_c=0.0, tc=TC_CENTER)

sampler = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler(
    network         = network,
    waveform_fn     = waveform_fn,
    param_bounds    = PARAM_BOUNDS,
    fixed_params    = FIXED_PARAMS,
    priors          = PRIORS,
    gmst            = None,                          # auto-uses network.gmst
    periodic_params = ("phi_c", "ra", "psi"),        # wrapped on the unit cube
)
print(f"sampling dimension : {len(sampler.param_bounds)}")
print("priors             :", {k: type(v).__name__ for k, v in sampler._prior_specs.items()})
''')

# ── sampler budget ───────────────────────────────────────────────────────────
cells[i_run_md] = md("""
## 6. Run the acceptance-walk sampler

Settings of the bilby-validated acceptance-walk study (Prathaban et al. 2025): `n_target = 60` accepted
steps per new point with up to `max_mcmc = 5000` proposals (the old cap of 1200 clipped the chains at the
~2–3% acceptance rates typical here), `num_delete = 0.5·num_live`, and the stopping rule equivalent to
bilby's `dlogz < 0.1`, i.e. `ln Z_live − ln Z < ln(e^0.1 − 1)`. Expect the wider prior to need more
iterations than the published run.
""")
cells[i_run] = code('''
import numpy as np
NUM_LIVE         = 2000
NUM_DELETE       = 1000                          # 0.5 * NUM_LIVE
N_TARGET         = 60
MAX_MCMC         = 5000
LOG_DLOGZ_TARGET = float(np.log(np.expm1(0.1)))  # = -2.252, bilby dlogz < 0.1
SEED             = 0

t0 = time.perf_counter()
result = sampler.run(
    rng_key               = jax.random.PRNGKey(SEED),
    num_live              = NUM_LIVE,
    n_target              = N_TARGET,
    max_mcmc              = MAX_MCMC,
    num_delete            = NUM_DELETE,
    max_iterations        = 20000,
    log_dlogz_target      = LOG_DLOGZ_TARGET,
    num_posterior_samples = 10000,
    verbose               = True,
)
elapsed = time.perf_counter() - t0

print(f"\\ntotal wall-clock   = {elapsed:.1f} s")
print(f"  iterations       = {result.n_iterations}")
print(f"  log Z            = {result.logZ:+.3f} ± {result.logZ_err:.3f}")
print(f"  ESS              = {result.ess:.1f}")
''')

save_cell = code('''
# ── Save everything to Google Drive (one folder per run) ─────────────────────
import json, os, datetime
import numpy as np
from google.colab import drive
if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")
RUN_DIR = f"/content/drive/MyDrive/gwjax_referee/gw150914_mlgw_bbh_lvkpriors/{datetime.datetime.now():%Y%m%d_%H%M%S}_{PSD_MODE}"
os.makedirs(RUN_DIR, exist_ok=True)

post = {k: np.asarray(v) for k, v in result.posterior_samples.items()}
np.savez_compressed(f"{RUN_DIR}/posterior_samples.npz", **post)
noise_logl = float(sum(i.log_likelihood_data(jnp.zeros_like(i.strain_data_fd)) for i in network.interferometers))
json.dump(dict(
    logZ=float(result.logZ), logZ_err=float(result.logZ_err), log_noise_evidence=noise_logl,
    log_bayes_factor=float(result.logZ) - noise_logl, ess=float(result.ess),
    n_iterations=int(result.n_iterations), elapsed_s=elapsed,
    config=dict(event="GW150914", detectors=[i.name for i in network.interferometers],
                duration=grid.duration, sampling_rate=grid.sampling_rate, f_min=20.0, f_max=1024.0,
                psd_mode=PSD_MODE, psd_segment_duration=PSD_SEGMENT_DURATION, psd_offset=PSD_OFFSET,
                waveform="mlgw_bbh_jax model_4 (SEOBNRv5HM), (2,2) mode", mlgw_final_time=MLGW_FINAL_TIME,
                param_bounds=PARAM_BOUNDS, priors=PRIORS, periodic=["phi_c", "ra", "psi"],
                sampler="acceptance-walk NS", num_live=NUM_LIVE, num_delete=NUM_DELETE,
                n_target=N_TARGET, max_mcmc=MAX_MCMC, log_dlogz_target=LOG_DLOGZ_TARGET, seed=SEED,
                gwgpu_jax_version=getattr(gwgpu_jax, "__version__", "?")),
), open(f"{RUN_DIR}/result.json", "w"), indent=2, default=list)
import pandas as pd
pd.DataFrame(post).to_csv(f"{RUN_DIR}/posterior_samples.txt", sep="\\t", index=False)
print("saved to", RUN_DIR)
''')

corner_cell = code('''
import corner, matplotlib.pyplot as plt
names = list(PARAM_BOUNDS)
X = np.column_stack([post[n] for n in names])
lo, hi = np.percentile(X, 0.2, axis=0), np.percentile(X, 99.8, axis=0)
pad = 0.1 * (hi - lo)
fig = corner.corner(X, labels=names, range=list(zip(lo - pad, hi + pad)),
                    truths=[PUBLISHED_MEDIAN[n] for n in names], truth_color="C3",
                    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 9},
                    plot_datapoints=False, smooth=0.9)
fig.savefig(f"{RUN_DIR}/corner.png", dpi=110, bbox_inches="tight")
plt.show()
''')

cells[i_corner_md] = md("""
## 8. Corner plot

Ranges follow the posterior (0.2–99.8% + padding), since the prior box is now much wider than the
posterior. Red lines: GWTC-1 medians, visual reference only.
""")

gwtc_fetch = code('''
# ── GWTC-2.1 release file (Zenodo 6513631); upload it manually if the download fails ──
import os, subprocess
GWTC_FILE = "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
if not os.path.exists(GWTC_FILE):
    url = f"https://zenodo.org/records/6513631/files/{GWTC_FILE}?download=1"
    r = subprocess.run(["wget", "-q", "-O", GWTC_FILE, url])
    if r.returncode != 0 or os.path.getsize(GWTC_FILE) < 1e6:
        os.remove(GWTC_FILE) if os.path.exists(GWTC_FILE) else None
        raise FileNotFoundError(f"download failed — upload {GWTC_FILE} to /content manually")
print("GWTC-2.1 file:", GWTC_FILE, f"{os.path.getsize(GWTC_FILE)/1e6:.0f} MB")
''')

jsd_md = md("""
## 10. JS divergence vs GWTC-2.1 (referee comment 1b)

LVK convention: 1-D Gaussian KDEs on a shared grid (reflected at hard bounds, wrapped for ra), JSD in
**bits**, bootstrap median and 90% interval, plus the split-half noise floor of our own run.
**Caveat (as the referee notes):** GWTC-2.1 uses precessing higher-mode models, a different sampler,
BayesWave PSDs and calibration marginalisation, so this comparison mixes all of those effects; the
surrogate/sampler-only numbers come from the injection benchmarks (`examples/referee_response/jsd_results/`).
""")
jsd_cell = code('''
from scipy.stats import gaussian_kde
from pesummary.io import read as pes_read

def _pdf(x, grid, bounds=None, period=None):
    k = gaussian_kde(x); p = k(grid)
    if period: p += k(grid - period) + k(grid + period)
    elif bounds:
        if bounds[0] is not None: p += k(2 * bounds[0] - grid)
        if bounds[1] is not None: p += k(2 * bounds[1] - grid)
    p = np.clip(p, 0, None); return p / np.trapz(p, grid)

def jsd_bits(x, y, bounds=None, period=None, n=2000):
    span = max(x.max(), y.max()) - min(x.min(), y.min())
    lo, hi = min(x.min(), y.min()) - 0.1 * span, max(x.max(), y.max()) + 0.1 * span
    if period: lo, hi = 0.0, period
    elif bounds:
        lo = max(lo, bounds[0]) if bounds[0] is not None else lo
        hi = min(hi, bounds[1]) if bounds[1] is not None else hi
    g = np.linspace(lo, hi, n); p, q = _pdf(x, g, bounds, period), _pdf(y, g, bounds, period); m = 0.5 * (p + q)
    kl = lambda a, b: np.trapz(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / b), 0), g)
    return float(np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0, 1))

def boot(x, y, rng, nb=40, ns=3000, **kw):
    ns = min(ns, len(x), len(y))
    return np.percentile([jsd_bits(rng.choice(x, ns, False), rng.choice(y, ns, False), **kw) for _ in range(nb)], [5, 50, 95])

ours = dict(mc=post["mc"], q=post["q"], chi_eff=(post["chi_1"] + post["q"] * post["chi_2"]) / (1 + post["q"]),
            distance=post["distance"], theta_jn=post["inclination"], ra=np.mod(post["ra"], 2 * np.pi), dec=post["dec"])
BND = dict(q=(None, 1.0), chi_eff=(-1.0, 1.0), theta_jn=(0.0, np.pi))
pe = pes_read(GWTC_FILE, disable_prior=True)
rng = np.random.default_rng(0)
table = {}
for label in ["C01:Mixed", "C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM"]:
    if label not in pe.samples_dict: continue
    s = pe.samples_dict[label]
    ref = dict(mc=np.asarray(s["chirp_mass"]), q=np.asarray(s["mass_ratio"]), chi_eff=np.asarray(s["chi_eff"]),
               distance=np.asarray(s["luminosity_distance"]), theta_jn=np.asarray(s["theta_jn"]),
               ra=np.mod(np.asarray(s["ra"]), 2 * np.pi), dec=np.asarray(s["dec"]))
    print(f"\\n=== mlgw_bbh (this run) vs GWTC-2.1 {label}")
    print(f"{'param':10s} {'JSD [bits]':>10s} {'90% CI':>19s}  {'floor':>7s}")
    table[label] = {}
    for n in ours:
        kw = dict(period=2 * np.pi) if n == "ra" else dict(bounds=BND.get(n))
        lo, med, hi = boot(ours[n], ref[n], rng, **kw)
        perm = rng.permutation(len(ours[n])); h = min(3000, len(perm) // 2)
        floor = jsd_bits(ours[n][perm[:h]], ours[n][perm[h:2 * h]], **kw)
        table[label][n] = dict(jsd_median=med, jsd_5=lo, jsd_95=hi, floor=floor)
        print(f"{n:10s} {med:10.4f} [{lo:8.4f},{hi:8.4f}]  {floor:7.4f}")
json.dump(table, open(f"{RUN_DIR}/jsd_vs_gwtc2p1.json", "w"), indent=2)
print("\\nsaved", f"{RUN_DIR}/jsd_vs_gwtc2p1.json")
''')

# stale upstream comment: the waveform_fn samples q = m2/m1 in (0, 1], not m1/m2 >= 1
i_summ = find(lambda s: s.startswith("# No heavier-first relabelling needed"))
replace_once(i_summ, "we sampled q = m1/m2 >= 1 directly", "we sampled q = m2/m1 <= 1 directly")

# ── assemble: keep install/setup/network/waveform/summary cells, rebuild the tail ──
head = cells[:i_time + 1]
summary = [c for c in cells[i_time + 1:i_corner_md]]
gwtc_plots = [cells[i] for i in gwtc_plot]
for c in gwtc_plots:
    s = "".join(c["source"])
    s = s.replace("filename = 'IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo (3).h5'", "filename = GWTC_FILE")
    c["source"] = s.splitlines(True)

new_cells = (head + [save_cell] + summary + [cells[i_corner_md], corner_cell,
             md("## 9. Comparison with GWTC-2.1 (visual)"), code("!pip install -q pesummary"), gwtc_fetch]
             + gwtc_plots + [jsd_md, jsd_cell])
for c in new_cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
nb["cells"] = new_cells
json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"wrote {OUT.name}: {len(new_cells)} cells")
