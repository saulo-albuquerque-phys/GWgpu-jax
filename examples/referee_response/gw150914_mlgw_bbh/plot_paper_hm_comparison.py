#!/usr/bin/env python
"""Paper figure + JS-divergence table: GW150914, GPU run (mlgw_bbh_jax, 7 modes, GWgpu-jax) vs the
bilby/dynesty SEOBNRv5HM 7-mode reference, same data / PSD / priors (data_bilby_hm_8s/README.md).

    python plot_paper_hm_comparison.py RUN_DIR [--bilby POSTERIOR.npz] [--out NAME]

RUN_DIR is the Drive folder written by COLAB_GW150914_mlgw_bbh_HM_vs_bilby8s.ipynb. Writes
RUN_DIR/NAME.pdf/.png and RUN_DIR/jsd_gwjax_vs_bilby_SEOBNRv5HM_7modes.json.
JSD: same estimator as the GW170817 table (../compute_jsd.py: KDE with reflection at prior bounds,
wrapping for periodic angles, bits, bootstrap median and 5-95%, split-half noise floors).
Mass ratio is quoted as q = m2/m1 <= 1 (the runs sampled 1/q).
"""
import argparse
import json
import sys
from pathlib import Path

import corner
import matplotlib as mpl
import matplotlib.lines as mlines
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from compute_jsd import bootstrap_jsd, noise_floor  # noqa: E402

PI = np.pi
# name: (label, summary format, tick format, hard bounds (reflection), period)
P = {
    "mtot":        (r"$M^{\rm det}\,[M_\odot]$", "{:.1f}", "%.0f", (None, None), None),
    "mc":          (r"$\mathcal{M}^{\rm det}\,[M_\odot]$", "{:.1f}", "%.0f", (None, None), None),
    "q":           (r"$q$", "{:.2f}", "%.2f", (0.125, 1.0), None),
    "chi_1":       (r"$\chi_1$", "{:.2f}", "%.1f", (-0.8, 0.8), None),
    "chi_2":       (r"$\chi_2$", "{:.2f}", "%.1f", (-0.8, 0.8), None),
    "chi_eff":     (r"$\chi_{\rm eff}$", "{:.2f}", "%.1f", (None, None), None),
    "distance":    (r"$d_L\,[\mathrm{Mpc}]$", "{:.0f}", "%.0f", (100.0, 3000.0), None),
    "inclination": (r"$\theta_{JN}\,[\mathrm{rad}]$", "{:.2f}", "%.1f", (0.0, PI), None),
    "ra":          (r"$\alpha\,[\mathrm{rad}]$", "{:.2f}", "%.1f", (None, None), 2 * PI),
    "dec":         (r"$\delta\,[\mathrm{rad}]$", "{:.2f}", "%.1f", (-PI / 2, PI / 2), None),
    "psi":         (r"$\psi\,[\mathrm{rad}]$", "{:.2f}", "%.1f", (None, None), PI),
    "tc":          (r"$t_c\,[\mathrm{s}]$", "{:.4f}", "%.3f", (None, None), None),
}
FIG = ["mc", "q", "chi_eff", "distance", "inclination", "ra", "dec"]
C_GWJAX, C_BILBY = "#1b4f8a", "#d1495b"


def derived(mtot, qinv, c1, c2, **rest):
    """qinv = m1/m2 >= 1 (sampled) -> q = m2/m1, chirp mass, chi_eff."""
    return dict(mtot=mtot, mc=mtot * qinv**0.6 / (1 + qinv) ** 1.2, q=1.0 / qinv, chi_1=c1, chi_2=c2,
                chi_eff=(qinv * c1 + c2) / (1 + qinv), **rest)


def load_gwjax(run_dir):
    p = {k: np.asarray(v, float) for k, v in np.load(run_dir / "posterior_samples.npz").items()}
    return derived(p["mtot"], p["q"], p["chi_1"], p["chi_2"], distance=p["distance"], inclination=p["inclination"],
                   ra=p["ra"], dec=p["dec"], psi=p["psi"], tc=p["tc"])


def load_bilby(path):
    b = np.load(path)
    s = dict(zip([str(n) for n in b["names"]], np.asarray(b["samples"], float).T))
    return derived(s["mtot"], s["q"], s["chi1"], s["chi2"], distance=s["d_l"], inclination=np.arccos(s["cos_iota"]),
                   ra=s["ra"], dec=np.arcsin(s["sin_dec"]), psi=s["psi"], tc=s["t_c"]), b


def kde1d(x, grid, lo=None, hi=None):
    k = gaussian_kde(x)
    p = k(grid)
    if lo is not None:
        p += k(2 * lo - grid)
    if hi is not None:
        p += k(2 * hi - grid)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--bilby", type=Path, default=HERE / "data_bilby_hm_8s/bilby_SEOBNRv5HM_7modes_posterior.npz")
    ap.add_argument("--smooth", type=float, default=1.5)
    ap.add_argument("--n-sub", type=int, default=2500)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--out", default="paper_gw150914_hm_comparison")
    a = ap.parse_args()

    G = load_gwjax(a.run_dir)
    B, braw = load_bilby(a.bilby)
    R = json.load(open(a.run_dir / "result.json"))

    # ── JS divergences ──────────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    table = {}
    print(f"{'param':12s} {'gwjax median +/- (90%)':>28s} {'bilby median +/- (90%)':>28s}  JSD [bit] (5-95%)        floor a/b   w90 a/b")
    for k, (_, f, _, bounds, period) in P.items():
        x, y = G[k], B[k]
        bnd = None if bounds == (None, None) else bounds
        j5, j50, j95 = bootstrap_jsd(x, y, a.n_boot, a.n_sub, rng, bnd, period=period)
        fa = noise_floor(x, a.n_boot, a.n_sub, rng, bnd, period=period)
        fb = noise_floor(y, a.n_boot, a.n_sub, rng, bnd, period=period)
        qa, qb = np.quantile(x, [0.05, 0.5, 0.95]), np.quantile(y, [0.05, 0.5, 0.95])
        table[k] = dict(jsd_median=float(j50), jsd_5=float(j5), jsd_95=float(j95), floor_a=fa, floor_b=fb,
                        gwjax_q05_q50_q95=qa.tolist(), bilby_q05_q50_q95=qb.tolist(),
                        width90_ratio=float((qa[2] - qa[0]) / (qb[2] - qb[0])))
        s = lambda q: f"{f.format(q[1])} +{f.format(q[2]-q[1])} -{f.format(q[1]-q[0])}"
        print(f"{k:12s} {s(qa):>28s} {s(qb):>28s}  {j50:.4f} ({j5:.4f}-{j95:.4f})   {fa:.4f}/{fb:.4f}   {table[k]['width90_ratio']:.2f}")
    out = dict(a=f"gwjax:{a.run_dir.name}", b=f"bilby:{a.bilby.name}", n_sub=a.n_sub, n_boot=a.n_boot,
               log_bayes_factor=dict(gwjax=[R["log_bayes_factor"], R["logZ_err"]],
                                     bilby=[float(braw["log_z"]), float(braw["log_z_err"])]),
               wall_time=dict(gwjax_s=R["elapsed_s"], gwjax_iterations=R["n_iterations"],
                              bilby_s=float(braw["sampling_time_s"]), bilby_likelihood_calls=int(braw["n_likelihood"])),
               n_samples=dict(gwjax=len(G["mc"]), bilby=len(B["mc"])), jsd=table)
    json.dump(out, open(a.run_dir / "jsd_gwjax_vs_bilby_SEOBNRv5HM_7modes.json", "w"), indent=2)
    print(f"ln BF: gwjax {R['log_bayes_factor']:.2f} +- {R['logZ_err']:.2f}   bilby {float(braw['log_z']):.2f} +- {float(braw['log_z_err']):.2f}")
    print(f"max JSD {max(v['jsd_median'] for v in table.values()):.4f} bit")

    # ── figure ───────────────────────────────────────────────────────────────────────────────
    sets = [("GWgpu-jax + mlgw_bbh_jax, 7 modes (GPU)", G, C_GWJAX, True),
            ("bilby/dynesty + SEOBNRv5HM, 7 modes (CPU)", B, C_BILBY, False)]
    n = len(FIG)
    rngs = []
    for p in FIG:
        # 0.3-99.7%: a few face-on samples (theta_JN ~ 0) would otherwise stretch the axis over [0, pi]
        lo = min(np.quantile(s[1][p], 0.003) for s in sets); hi = max(np.quantile(s[1][p], 0.997) for s in sets)
        pad = 0.04 * (hi - lo); blo, bhi = P[p][3]
        rngs.append((max(lo - pad, blo) if blo is not None else lo - pad, min(hi + pad, bhi) if bhi is not None else hi + pad))
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13, "axes.labelsize": 19,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 0.8,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig = None
    for label, s, col, fill in sets:
        fig = corner.corner(np.column_stack([s[p] for p in FIG]), fig=fig, range=rngs, bins=40, smooth=a.smooth,
                            levels=(0.5, 0.9), labels=[P[p][0] for p in FIG], color=col, max_n_ticks=4, labelpad=0.06,
                            plot_datapoints=False, plot_density=False, fill_contours=fill, no_fill_contours=not fill,
                            contour_kwargs=dict(linewidths=1.5), hist_kwargs=dict(alpha=0.0))
    axes = np.asarray(fig.axes).reshape(n, n)
    for i, p in enumerate(FIG):
        ax = axes[i, i]
        for art in list(ax.patches) + list(ax.lines):
            art.remove()
        grid = np.linspace(*rngs[i], 600); top = 0.0
        for label, s, col, fill in sets:
            y = kde1d(s[p], grid, *P[p][3])
            ax.plot(grid, y, color=col, lw=1.8)
            if fill:
                ax.fill_between(grid, 0, y, color=col, alpha=0.15, lw=0)
            top = max(top, y.max())
        ax.set_xlim(rngs[i]); ax.set_ylim(0, 1.08 * top); ax.set_yticks([])
        axes[n - 1, i].xaxis.set_major_locator(MaxNLocator(4, prune="both"))
        axes[n - 1, i].xaxis.set_major_formatter(FormatStrFormatter(P[p][2]))
        if i > 0:
            axes[i, 0].yaxis.set_major_locator(MaxNLocator(4, prune="both"))
            axes[i, 0].yaxis.set_major_formatter(FormatStrFormatter(P[p][2]))
    for ax in fig.axes:
        ax.tick_params(length=3.5, width=0.7, top=True, right=True)
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.985, top=0.985, wspace=0.05, hspace=0.05)
    handles = [mlines.Line2D([], [], color=col, lw=2.2, label=label) for label, _, col, _ in sets]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.975), fontsize=18, frameon=False)
    y0, dy = 0.875, 0.027
    fig.text(0.975, y0 + dy, "median and 90% credible interval;  JS divergence [bit]", ha="right", fontsize=13, color="0.3")
    for r, p in enumerate(FIG):
        f = P[p][1]
        fig.text(0.62, y0 - r * dy, P[p][0].split(r"\,[")[0] + ("$" if r"\,[" in P[p][0] else ""), ha="right", fontsize=14)
        for c, (label, s, col, fill) in enumerate(sets):
            lo, med, hi = np.quantile(s[p], [0.05, 0.5, 0.95])
            fig.text(0.65 + 0.115 * c, y0 - r * dy, f"${f.format(med)}^{{+{f.format(hi - med)}}}_{{-{f.format(med - lo)}}}$",
                     fontsize=13.5, color=col)
        fig.text(0.975, y0 - r * dy, f"{table[p]['jsd_median']:.4f}", ha="right", fontsize=13, color="0.25")
    for ext in ("pdf", "png"):
        fig.savefig(a.run_dir / f"{a.out}.{ext}", dpi=220 if ext == "png" else None)
    print("wrote", a.run_dir / f"{a.out}.pdf", "and .png")


if __name__ == "__main__":
    main()
