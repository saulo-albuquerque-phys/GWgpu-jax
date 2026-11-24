#!/usr/bin/env python
"""P-P plot + calibration statistics from a P-P campaign directory.

Works on any run dir whose inj_NNN/result.json files hold `truth` and
`p_values` (run_pp_mlgw_bbh.py, and the earlier ripplegw P-P notebooks).

Reports, as in the LVK/bilby P-P convention (e.g. Romero-Shaw et al. 2020):
  * empirical CDF of the credible level of the truth, per parameter,
    with 1/2/3-sigma binomial bands;
  * per-parameter KS p-value against U(0,1) and Fisher-combined p-value;
  * a derived chi_eff curve, recomputed from the stored posteriors
    (shows the best-measured spin combination is calibrated too); with --params also the
    derived m1, m2, mtot.

    python make_pp_plot.py --run-dir pp_mlgw_bbh_run [--out pp_plot.pdf]

Paper figure (physical parameters only; component masses m1, m2 and chi_eff derived from the
sampled mc, q, chi_1, chi_2; nuisance t_c, phi_c left out):

    python make_pp_plot.py --run-dir pp_mlgw_bbh_fixedwalk \
        --params m1 m2 chi_eff distance inclination ra dec psi --out pp_mlgw_bbh_fixedwalk/pp_plot_physical.pdf

With --params the curves and the Fisher-combined p-value use exactly those parameters, and the
summary goes to <out stem>_summary.json.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--out", default=None, help="default: <run-dir>/pp_plot.pdf")
ap.add_argument("--no-chi-eff", action="store_true")
ap.add_argument("--params", nargs="+", default=None,
                help="plot and combine only these (may include the derived chi_eff)")
args = ap.parse_args()

run = Path(args.run_dir)
recs = [json.load(open(f)) for f in sorted(glob.glob(str(run / "inj_*" / "result.json")))]
recs = [r for r in recs if r.get("completed")]
if len(recs) < 2:
    raise SystemExit(f"only {len(recs)} completed injections in {run}")
names = list(recs[0]["p_values"])

LABELS = {"mc": r"$\mathcal{M}_c$", "Mc": r"$\mathcal{M}_c$", "q": r"$q$",
          "chi_1": r"$\chi_1$", "chi_2": r"$\chi_2$", "distance": r"$d_L$",
          "inclination": r"$\theta_{JN}$", "ra": r"$\alpha$", "dec": r"$\delta$",
          "psi": r"$\psi$", "phi_c": r"$\phi_c$", "tc": r"$t_c$",
          "chi_eff": r"$\chi_{\rm eff}$", "m1": r"$m_1$", "m2": r"$m_2$", "mtot": r"$M$"}

pvals = {n: np.array([r["p_values"][n] for r in recs]) for n in names}

# derived parameters, recomputed from the stored posteriors (q = m2/m1 <= 1):
#   chi_eff = (chi1 + q chi2)/(1+q);  m1 = Mc (1+q)^0.2 / q^0.6;  m2 = q m1;  M = m1 + m2
def _derived(d, qn):
    q = d[qn]
    out = {}
    if "chi_1" in d and "chi_2" in d:
        out["chi_eff"] = (d["chi_1"] + q * d["chi_2"]) / (1 + q)
    if "mc" in d:
        m1 = d["mc"] * (1 + q) ** 0.2 / q ** 0.6
        out.update(m1=m1, m2=q * m1, mtot=m1 * (1 + q))
    return out


qn = "q" if "q" in names else None
if qn and not args.no_chi_eff:
    pe = {}
    for r in recs:
        f = run / f"inj_{r['index']:03d}" / "posterior_samples.npz"
        if not f.exists():
            pe = None
            break
        post = _derived(np.load(f), qn)
        tru = _derived({k: np.asarray(v) for k, v in r["truth"].items()}, qn)
        for k in post:
            pe.setdefault(k, []).append(np.mean(post[k] < tru[k]))
    if pe:
        if not args.params:      # default plot: only chi_eff, as before
            pe = {k: v for k, v in pe.items() if k == "chi_eff"}
        pvals.update({k: np.array(v) for k, v in pe.items()})

if args.params:
    missing = [n for n in args.params if n not in pvals]
    if missing:
        raise SystemExit(f"unknown/unavailable parameters: {missing}")
    pvals = {n: pvals[n] for n in args.params}
    names = list(args.params)

N = len(recs)
x = np.linspace(0, 1, 1001)
# shown one column wide in the paper: large fonts so that labels stay legible in print
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 15,
                     "axes.labelsize": 19, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for sig, a in [(1, 0.25), (2, 0.15), (3, 0.08)]:
    e = stats.norm.cdf(sig)
    ax.fill_between(x, stats.binom.ppf(1 - e, N, x) / N, stats.binom.ppf(e, N, x) / N,
                    color="0.4", alpha=a, lw=0)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)

ks = {}
cmap = plt.get_cmap("tab10" if len(pvals) <= 10 else "tab20")
for j, (n, p) in enumerate(pvals.items()):
    ks[n] = float(stats.kstest(p, "uniform").pvalue)
    ax.plot(x, np.searchsorted(np.sort(p), x, side="right") / N, lw=1.8,
            color=cmap(j % cmap.N), label=f"{LABELS.get(n, n)} ({ks[n]:.2f})")
comb = float(stats.combine_pvalues([ks[n] for n in names], method="fisher").pvalue)

ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("Credible interval"); ax.set_ylabel("Fraction of events in credible interval")
ax.set_title(f"N = {N}, combined $p$-value = {comb:.3f}", fontsize=17)
ax.tick_params(top=True, right=True)
ax.legend(fontsize=13.5, loc="upper left", title="KS $p$-value", title_fontsize=13.5, frameon=False,
          handlelength=1.4, labelspacing=0.3, borderaxespad=0.3)
out = Path(args.out) if args.out else run / "pp_plot.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")

snr = np.array([r["network_snr"] for r in recs])
el = np.array([r["elapsed_s"] for r in recs])
summary = dict(n=N, ks_pvalues=ks, combined_pvalue_fisher=comb,
               combined_over=names,
               snr_min=float(snr.min()), snr_median=float(np.median(snr)),
               snr_max=float(snr.max()), n_snr_ge_8=int((snr >= 8).sum()),
               wall_time_median_s=float(np.median(el)), wall_time_total_h=float(el.sum() / 3600))
summary_file = out.with_name(out.stem + "_summary.json") if args.params else run / "pp_summary.json"
json.dump(summary, open(summary_file, "w"), indent=2)

print(f"N = {N}   SNR {snr.min():.1f}–{snr.max():.1f} (median {np.median(snr):.1f}, "
      f"{(snr >= 8).sum()} with SNR>=8)")
for n, p in ks.items():
    print(f"  {n:12s} KS p = {p:.3f}")
print(f"combined (Fisher, {'selected' if args.params else 'sampled'} params) p = {comb:.3f}")
print(f"wall time per injection: median {np.median(el)/60:.1f} min, total {el.sum()/3600:.1f} h")
print("saved", out, "and", summary_file)
