#!/usr/bin/env python
"""Generate COLAB_GW150914_mlgw_bbh_HM_vs_bilby8s.ipynb.

GW150914 with all 7 modes of mlgw_bbh_jax model_4, in EXACTLY the configuration of the earlier
bilby/dynesty SEOBNRv5HM 7-mode run (data_bilby_hm_8s/README.md): same conditioned data and PSD,
same priors, same sampler settings -> like-for-like JS divergences.

Install / sub-repo / self-check / kernel cells are taken from COLAB_GW150914_mlgw_bbh_LVKpriors_HM.ipynb
(run `python build_notebook.py --hm` first if that one changed).

    python build_notebook_hm_bilby8s.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "COLAB_GW150914_mlgw_bbh_LVKpriors_HM.ipynb"
OUT = HERE / "COLAB_GW150914_mlgw_bbh_HM_vs_bilby8s.ipynb"
BRANCH = "py312_modern_jax"      # where data_bilby_hm_8s/ lives until merged
RAW = ("https://raw.githubusercontent.com/saulo-albuquerque-phys/GWgpu-jax/" + BRANCH +
       "/examples/referee_response/gw150914_mlgw_bbh/data_bilby_hm_8s")

src = json.load(open(SRC))["cells"]
S = lambda c: "".join(c["source"])


def find(pred, kind="code"):
    hits = [c for c in src if c["cell_type"] == kind and pred(S(c))]
    assert len(hits) == 1, (pred, len(hits))
    return hits[0]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


c_install_md = find(lambda s: s.startswith("## 1. Install"), "markdown")
c_install = find(lambda s: "gwjax_constraints.txt" in s and "pip install" in s)
c_sub_md = find(lambda s: s.startswith("### 1b."), "markdown")
c_sub = find(lambda s: "MLGW-JAX.git" in s)
c_check = find(lambda s: s.startswith("# Waveform self-check:"))
assert "modes=None" in S(c_check)
c_imports = find(lambda s: 'jax.config.update("jax_enable_x64", True)' in s and "import gwgpu_jax" in s
                 and not s.startswith("# Waveform self-check:"))
c_kernel = find(lambda s: "blackjax_ns_gw.git" in s)

title = md(f"""
# GW150914 — mlgw_bbh_jax (SEOBNRv5HM surrogate, all 7 modes) vs bilby/dynesty SEOBNRv5HM, like for like

GPU run in **exactly the configuration of the bilby/dynesty SEOBNRv5HM 7-mode reference run**
(`data_bilby_hm_8s/`), so the two posteriors can be compared with JS divergences:

| | value (identical in both runs) |
|---|---|
| data | the same conditioned frequency-domain strain and PSD arrays (H1, L1), read from one file |
| segment | 8 s at 2048 Hz, trigger 6 s into the segment; 20–1000 Hz |
| PSD | median Welch over 512 s |
| modes | (2,2) (2,1) (3,3) (3,2) (4,4) (4,3) (5,5) and their m < 0 partners |
| M_tot | uniform on [55, 120] M☉ (detector frame) |
| q = m1/m2 | uniform on [1, 8] |
| χ₁, χ₂ | uniform on [−0.8, 0.8] |
| d_L | ∝ d² on [100, 3000] Mpc |
| orientation, sky | isotropic; ψ ∈ [0, π]; φ_c ∈ [0, 2π] sampled (no marginalisation) |
| t_c | ±0.05 s around the trigger |
| sampler | acceptance-walk NS, 1000 live points, 60 accepted steps targeted, stop at dlogz 0.1 |

Differences that remain: the waveform (surrogate here, SEOBNRv5HM itself in bilby) and the sampler
implementation. `tc` here is the time of the (2,2) amplitude peak, which agrees with bilby's
`geocent_time` to ≈0.1 ms on average (0.3 ms scatter).

bilby reference: ln BF = 261.85 ± 0.21, 97.8 h on 16 cores, 4.73e7 likelihood calls.

**Order:** run the install cell, restart the session, then run everything from the top.
Results go to Google Drive (`RUN_DIR`).
""")

data_cell = code(f'''
import time, json, urllib.request
from pathlib import Path
from gwgpu_jax.gwgpu_jax_compatibility import gmst_from_gps

# The exact inputs and posterior of the bilby reference run (see data_bilby_hm_8s/README.md).
RAW = "{RAW}"
DATA_DIR = Path(os.environ.get("GWJAX_HM8_DATA_DIR", "/content/data_bilby_hm_8s"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
for name in ("gw150914_data.npz", "bilby_SEOBNRv5HM_7modes_posterior.npz"):
    if not (DATA_DIR / name).exists():
        urllib.request.urlretrieve(f"{{RAW}}/{{name}}", DATA_DIR / name)
D = np.load(DATA_DIR / "gw150914_data.npz")
B = np.load(DATA_DIR / "bilby_SEOBNRv5HM_7modes_posterior.npz")

DT, N = float(D["dt"]), len(D["times"])
DURATION, SAMPLING_RATE = N * DT, 1.0 / DT
F_MIN, F_MAX = float(D["f_low"]), float(D["f_high"])
T_REF = float(D["t_ref"])                          # trigger position in the segment [s]
grid = gwgpu_jax.TimeFrequencyGrid(duration=DURATION, sampling_rate=SAMPLING_RATE, f_min=F_MIN, f_max=F_MAX)
assert np.array_equal(np.asarray(grid.frequency_domain_array), D["freqs"]), "frequency grids differ"
network = gwgpu_jax.Network.from_names([str(n) for n in D["names"]], grid)
for k, ifo in enumerate(network.interferometers):
    ifo.load_data(jnp.asarray(D["data_fd"][k]), domain="fd")
    ifo.psd = jnp.asarray(D["psd"][k]); ifo._weight = None
network.gmst = float(gmst_from_gps(float(D["gps"])))
print(grid); print(network)

# Same data <=> same noise likelihood. bilby's log_noise_evidence is stored with its posterior.
noise_logl = float(sum(i.log_likelihood_data(jnp.zeros_like(i.strain_data_fd)) for i in network.interferometers))
print(f"noise ln L: gwjax {{noise_logl:.3f}}   bilby {{float(B['log_noise_evidence']):.3f}}")
assert abs(noise_logl - float(B["log_noise_evidence"])) < 1e-2, "data/PSD do not match the bilby run"
''')

wf_cell = code('''
from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator

# TD -> FD recipe of the (2,2) notebook (zero-crossing time grid + asymmetric Tukey taper), with all
# 7 modes. The grid starts 7.88 s before the merger: for these masses the (2,2) mode is then below
# 9 Hz, so every mode up to m = 5 enters the 20 Hz band inside the tapered region or before it.
MLGW_FINAL_TIME = 0.12                      # s kept after the (2,2) peak (whole ringdown up to M_tot ~ 230)
TUKEY_ALPHA_LEFT, TUKEY_ALPHA_RIGHT = 0.2, 0.001
MODES = None                                # None = all 7 modes: 22 21 32 33 43 44 55

def _tukey_asymmetric(n, alpha_left, alpha_right):
    x = jnp.linspace(0.0, 1.0, n)
    w = jnp.ones_like(x)
    left  = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * (x / alpha_left - 0.5)))
    right = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * ((x - 1.0) / alpha_right + 0.5)))
    w = jnp.where(x < alpha_left / 2.0, left, w)
    return jnp.where(x > 1.0 - alpha_right / 2.0, right, w)

t0 = time.perf_counter()
_gen    = MLGWBBHGenerator(model=4).generate
_T_GRID = jnp.arange(N) * DT - DURATION + MLGW_FINAL_TIME
_WINDOW = _tukey_asymmetric(N, TUKEY_ALPHA_LEFT, TUKEY_ALPHA_RIGHT)
# Shift the template so that `tc` is the time of the (2,2) peak (= bilby's geocent_time convention);
# without it the peak would sit at tc - MLGW_FINAL_TIME.
_SHIFT  = jnp.exp(-2j * jnp.pi * jnp.asarray(D["freqs"]) * MLGW_FINAL_TIME)

def waveform_fn(params, freqs):
    q  = params["q"]                        # q = m1/m2 >= 1, as in the bilby run
    m2 = params["mtot"] / (1.0 + q)
    m1 = q * m2
    theta = jnp.stack([m1, m2, params["chi_1"], params["chi_2"],
                       params["distance"], params["inclination"], params["phi_c"]])
    hp_td, hc_td = _gen(theta, _T_GRID, modes=MODES)
    return (jnp.fft.rfft(hp_td * _WINDOW) * DT * _SHIFT,
            -jnp.fft.rfft(hc_td * _WINDOW) * DT * _SHIFT)

print(f"mlgw_bbh_jax (SEOBNRv5HM, model_4, all 7 modes) loaded in {time.perf_counter()-t0:.1f} s")
''')

prior_cell = code('''
PI = float(jnp.pi)
PARAM_BOUNDS = {                            # identical to the bilby run
    "mtot":        (55.0, 120.0),           # uniform
    "q":           (1.0, 8.0),              # uniform in q = m1/m2
    "chi_1":       (-0.8, 0.8),             # uniform
    "chi_2":       (-0.8, 0.8),
    "distance":    (100.0, 3000.0),         # ~ d^2
    "inclination": (0.0, PI),               # sin
    "ra":          (0.0, 2.0 * PI),
    "dec":         (-PI / 2, PI / 2),       # cos
    "psi":         (0.0, PI),
    "phi_c":       (0.0, 2.0 * PI),
    "tc":          (T_REF - 0.05, T_REF + 0.05),
}
PRIORS = {"inclination": "sin", "dec": "cos", "distance": "volumetric"}   # everything else uniform

sampler = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler(
    network=network, waveform_fn=waveform_fn, param_bounds=PARAM_BOUNDS, priors=PRIORS,
    gmst=None, periodic_params=("phi_c", "ra", "psi"))
print(f"sampling dimension : {len(sampler.param_bounds)}")
print("priors             :", {k: type(v).__name__ for k, v in sampler._prior_specs.items()})

# bilby posterior in this notebook's parameter names
_b = dict(zip([str(n) for n in B["names"]], B["samples"].T))
BILBY = dict(mtot=_b["mtot"], q=_b["q"], chi_1=_b["chi1"], chi_2=_b["chi2"], distance=_b["d_l"],
             inclination=np.arccos(_b["cos_iota"]), ra=_b["ra"], dec=np.arcsin(_b["sin_dec"]),
             psi=_b["psi"], tc=_b["t_c"])
BILBY_LNBF, BILBY_LNBF_ERR = float(B["log_z"]), float(B["log_z_err"])

# Likelihood cross-check before spending GPU time: this notebook's likelihood, marginalised over
# phi_c on a grid, evaluated at bilby posterior samples. They must be high-likelihood points
# (ln L ratio ~ 270-290); a convention error in time, sky or masses would give values far below.
_idx = np.random.default_rng(0).choice(len(BILBY["mtot"]), 200, replace=False)
_vll = jax.jit(jax.vmap(sampler._loglik_fn))
_L = np.array([np.asarray(_vll({**{k: jnp.asarray(v[_idx]) for k, v in BILBY.items()},
                                "phi_c": jnp.full(len(_idx), p)})) for p in np.linspace(0, 2 * np.pi, 16, endpoint=False)]) - noise_logl
_m = _L.max(0); _marg = _m + np.log(np.mean(np.exp(_L - _m), 0))
print(f"phase-marginalised ln L ratio at 200 bilby samples: median {np.median(_marg):.1f}, "
      f"5-95% {np.quantile(_marg, 0.05):.1f} to {np.quantile(_marg, 0.95):.1f}   (bilby ln BF {BILBY_LNBF:.2f})")
assert np.median(_marg) > BILBY_LNBF, "bilby posterior samples are not high-likelihood points here: check conventions"
''')

run_md = md("""
## Run the acceptance-walk sampler

Same settings as the bilby run and as the validated (2,2) runs: 1000 live points, 10% replaced per
iteration, 60 accepted steps targeted per chain with a fixed number of steps (bilby's rule), stop at
bilby's `dlogz < 0.1`. The 7-mode waveform costs about 4× the (2,2) one.
""")

run_cell = code('''
from gwgpu_jax.gwgpu_jax_acceptance_walk import WALK_RULE
print("acceptance-walk rule:", WALK_RULE)
NUM_LIVE         = 1000
NUM_DELETE       = 100
N_TARGET         = 60
MAX_MCMC         = 5000
LOG_DLOGZ_TARGET = float(np.log(np.expm1(0.1)))  # bilby dlogz < 0.1
MAX_ITERATIONS   = 20000
SEED             = 0

t0 = time.perf_counter()
result = sampler.run(
    rng_key=jax.random.PRNGKey(SEED), num_live=NUM_LIVE, n_target=N_TARGET, max_mcmc=MAX_MCMC,
    num_delete=NUM_DELETE, max_iterations=MAX_ITERATIONS, log_dlogz_target=LOG_DLOGZ_TARGET,
    num_posterior_samples=10000, verbose=True)
elapsed = time.perf_counter() - t0
print(f"\\nTotal time for PE: {elapsed/60:.2f} minutes   iterations {result.n_iterations}")
print(f"ln BF = {float(result.logZ) - noise_logl:.3f} +- {float(result.logZ_err):.3f}   (bilby {BILBY_LNBF:.3f} +- {BILBY_LNBF_ERR:.3f})")
''')

save_cell = code('''
import datetime
try:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    _base = "/content/drive/MyDrive/gwjax_referee/gw150914_mlgw_bbh_hm_vs_bilby8s"
except ImportError:
    _base = os.environ.get("GWJAX_HM8_OUT", "gw150914_mlgw_bbh_hm_vs_bilby8s")
RUN_DIR = f"{_base}/{datetime.datetime.now():%Y%m%d_%H%M%S}_HM_bilby8s"
os.makedirs(RUN_DIR, exist_ok=True)

post = {k: np.asarray(v) for k, v in result.posterior_samples.items()}
np.savez_compressed(f"{RUN_DIR}/posterior_samples.npz", **post)
json.dump(dict(
    logZ=float(result.logZ), logZ_err=float(result.logZ_err), log_noise_evidence=noise_logl,
    log_bayes_factor=float(result.logZ) - noise_logl, ess=float(result.ess),
    n_iterations=int(result.n_iterations), elapsed_s=elapsed,
    bilby_log_bayes_factor=BILBY_LNBF, bilby_log_bayes_factor_err=BILBY_LNBF_ERR,
    config=dict(event="GW150914", detectors=[i.name for i in network.interferometers],
                duration=DURATION, sampling_rate=SAMPLING_RATE, f_min=F_MIN, f_max=F_MAX,
                data="data_bilby_hm_8s/gw150914_data.npz (inputs of the bilby SEOBNRv5HM 7-mode run)",
                waveform="mlgw_bbh_jax model_4 (SEOBNRv5HM), all 7 modes (22 21 32 33 43 44 55)",
                mlgw_final_time=MLGW_FINAL_TIME, tc_convention="time of the (2,2) peak in the segment",
                param_bounds=PARAM_BOUNDS, priors=PRIORS, periodic=["phi_c", "ra", "psi"],
                sampler="acceptance-walk NS, " + WALK_RULE, num_live=NUM_LIVE, num_delete=NUM_DELETE,
                n_target=N_TARGET, max_mcmc=MAX_MCMC, log_dlogz_target=LOG_DLOGZ_TARGET, seed=SEED,
                gwgpu_jax_version=getattr(gwgpu_jax, "__version__", "?")),
), open(f"{RUN_DIR}/result.json", "w"), indent=2, default=list)
print("saved to", RUN_DIR)
''')

cmp_md = md("""
## Comparison with the bilby/dynesty SEOBNRv5HM run

Median and 90% interval of both runs, and the JS divergence (bits) between the 1D marginals, with its
spread over bootstrap resamplings. `floor` is the divergence between two halves of the bilby samples,
i.e. what sampling noise alone produces at this sample size.
""")

cmp_cell = code('''
from scipy.stats import gaussian_kde

def jsd_bits(a, b, n_grid=600):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    g = np.linspace(lo, hi, n_grid)
    p, q = gaussian_kde(a)(g), gaussian_kde(b)(g)
    p, q = p / p.sum(), q / q.sum(); m = 0.5 * (p + q)
    kl = lambda x, y: np.sum(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1.0) / y), 0.0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def with_derived(s):
    s = dict(s); q = s["q"]
    s["mc"] = s["mtot"] * q ** 0.6 / (1.0 + q) ** 1.2
    s["chi_eff"] = (q * s["chi_1"] + s["chi_2"]) / (1.0 + q)
    return s

GJ, BI = with_derived(post), with_derived(BILBY)
CMP = ["mtot", "mc", "q", "chi_1", "chi_2", "chi_eff", "distance", "inclination", "ra", "dec", "psi", "tc"]
rng = np.random.default_rng(1); table = {}
print(f"{'param':12s} {'gwjax median (90%)':>30s} {'bilby median (90%)':>30s}   JSD [bit] (5-95%)      floor")
for k in CMP:
    a, b = GJ[k], BI[k]
    boot = [jsd_bits(rng.choice(a, 3000), rng.choice(b, 3000)) for _ in range(20)]
    h = rng.permutation(len(b)); floor = jsd_bits(b[h[:len(b) // 2]], b[h[len(b) // 2:]])
    qa, qb = np.quantile(a, [0.05, 0.5, 0.95]), np.quantile(b, [0.05, 0.5, 0.95])
    table[k] = dict(jsd_median=float(np.median(boot)), jsd_5=float(np.quantile(boot, 0.05)), jsd_95=float(np.quantile(boot, 0.95)),
                    floor=float(floor), gwjax=qa.tolist(), bilby=qb.tolist())
    fmt = lambda v: f"{v[1]:.4g} (+{v[2]-v[1]:.3g} / -{v[1]-v[0]:.3g})"
    print(f"{k:12s} {fmt(qa):>30s} {fmt(qb):>30s}   {np.median(boot):.4f} ({np.quantile(boot, .05):.4f}-{np.quantile(boot, .95):.4f})   {floor:.4f}")
print(f"\\nln BF: gwjax {float(result.logZ) - noise_logl:.2f} +- {float(result.logZ_err):.2f}   bilby {BILBY_LNBF:.2f} +- {BILBY_LNBF_ERR:.2f}")
json.dump(table, open(f"{RUN_DIR}/jsd_vs_bilby_SEOBNRv5HM_7modes.json", "w"), indent=2)
''')

corner_cell = code('''
!pip install -q corner
import corner, matplotlib.lines as mlines
names = ["mc", "q", "chi_eff", "distance", "inclination", "ra", "dec"]
XA = np.column_stack([GJ[n] for n in names]); XB = np.column_stack([BI[n] for n in names])
both = np.vstack([XA, XB]); lo, hi = np.percentile(both, 0.2, axis=0), np.percentile(both, 99.8, axis=0)
rngs = list(zip(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)))
kw = dict(labels=names, range=rngs, bins=40, smooth=1.2, levels=(0.5, 0.9), plot_datapoints=False, plot_density=False,
          hist_kwargs=dict(density=True))
fig = corner.corner(XA, color="#1b4f8a", fill_contours=True, **kw)
corner.corner(XB, fig=fig, color="#d1495b", **kw)
fig.legend(handles=[mlines.Line2D([], [], color="#1b4f8a", label="GWgpu-jax + mlgw_bbh_jax, 7 modes (GPU)"),
                    mlines.Line2D([], [], color="#d1495b", label="bilby/dynesty + SEOBNRv5HM, 7 modes (CPU)")],
           loc="upper right", fontsize=14, frameon=False)
fig.savefig(f"{RUN_DIR}/corner_vs_bilby.png", dpi=120, bbox_inches="tight")
plt.show()
''')

cells = [title, c_install_md, c_install, c_sub_md, c_sub, c_check, c_imports,
         md("## 2. Data: the inputs of the bilby reference run"), data_cell,
         md("## 3. Waveform: mlgw_bbh_jax model_4, all 7 modes"), wf_cell,
         md("## 4. Priors (identical to the bilby run), sampler, likelihood cross-check"), c_kernel, prior_cell,
         run_md, run_cell, md("## Save"), save_cell, cmp_md, cmp_cell, md("## Corner plot: GPU run vs bilby"), corner_cell]
import copy
cells = copy.deepcopy(cells)
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
nb = json.load(open(SRC)); nb["cells"] = cells
json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"wrote {OUT.name}: {len(cells)} cells")
