#!/usr/bin/env python
"""Publication-style corner plot of a gwjax GW170817 run (posterior_samples.npz in RUN_DIR).

    python plot_corner.py RUN_DIR [--params mc q chi_eff ...] [--out NAME]

Writes RUN_DIR/NAME.png and .pdf. 50% and 90% credible regions; titles give the median and
the 90% symmetric credible interval (LVK convention).
"""
import argparse
from pathlib import Path

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

LABELS = {
    "mc":           (r"$\mathcal{M}^{\rm det}\,[M_\odot]$", "{:.5f}", "%.4f"),
    "q":            (r"$q$", "{:.2f}", "%.2f"),
    "chi_eff":      (r"$\chi_{\rm eff}$", "{:.3f}", "%.2f"),
    "lambda_tilde": (r"$\tilde{\Lambda}$", "{:.0f}", "%.0f"),
    "lambda_1":     (r"$\Lambda_1$", "{:.0f}", "%.0f"),
    "lambda_2":     (r"$\Lambda_2$", "{:.0f}", "%.0f"),
    "distance":     (r"$d_L\,[\mathrm{Mpc}]$", "{:.1f}", "%.0f"),
    "inclination":  (r"$\theta_{JN}\,[\mathrm{rad}]$", "{:.2f}", "%.1f"),
    "ra":           (r"$\alpha\,[\mathrm{rad}]$", "{:.3f}", "%.2f"),
    "dec":          (r"$\delta\,[\mathrm{rad}]$", "{:.3f}", "%.2f"),
    "chi_1":        (r"$\chi_1$", "{:.3f}", "%.2f"),
    "chi_2":        (r"$\chi_2$", "{:.3f}", "%.2f"),
    "psi":          (r"$\psi\,[\mathrm{rad}]$", "{:.2f}", "%.1f"),
}
DEFAULT = ["mc", "q", "chi_eff", "lambda_tilde", "lambda_1", "lambda_2", "distance", "inclination", "ra", "dec"]
COLOR = "#1b4f8a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--params", nargs="+", default=DEFAULT)
    ap.add_argument("--out", default="corner_pretty")
    a = ap.parse_args()

    post = np.load(a.run_dir / "posterior_samples.npz")
    X = np.column_stack([post[p] for p in a.params])
    # plotting range: drop the extreme 0.1% tails (long Λ tails otherwise squash the panels)
    rng = [tuple(np.quantile(X[:, i], [0.001, 0.999])) for i in range(X.shape[1])]

    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 13,
        "axes.labelsize": 17, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    })
    fig = corner.corner(
        X, labels=[LABELS[p][0] for p in a.params], range=rng, bins=36, smooth=1.1, smooth1d=1.1,
        levels=(0.5, 0.9), fill_contours=True, plot_datapoints=False, plot_density=False,
        color=COLOR, max_n_ticks=4, labelpad=0.08,
        hist_kwargs=dict(lw=1.6),
        contour_kwargs=dict(linewidths=1.2),
    )
    n = len(a.params)
    axes = np.asarray(fig.axes).reshape(n, n)
    for i, p in enumerate(a.params):
        lo, med, hi = np.quantile(X[:, i], [0.05, 0.5, 0.95])
        f = LABELS[p][1]
        axes[i, i].set_title(f"{LABELS[p][0]} $= {f.format(med)}^{{+{f.format(hi - med)}}}_{{-{f.format(med - lo)}}}$",
                             fontsize=12.5, pad=8)
        for ln in axes[i, i].get_lines():      # light fill under the smoothed 1D marginal
            x, y = ln.get_data()
            axes[i, i].fill_between(x, 0, y, color=COLOR, alpha=0.18, lw=0)
        for q in (lo, med, hi):
            axes[i, i].axvline(q, color=COLOR, lw=1.0, ls="-" if q == med else "--", alpha=0.9)
        for j in range(n):          # no offset / scientific notation on any axis
            for ax in (axes[j, i], axes[i, j]):
                for axis in (ax.xaxis, ax.yaxis):
                    fmt = axis.get_major_formatter()
                    if hasattr(fmt, "set_useOffset"):
                        fmt.set_useOffset(False)
        axes[n - 1, i].xaxis.set_major_formatter(FormatStrFormatter(LABELS[p][2]))
        if i > 0:
            axes[i, 0].yaxis.set_major_formatter(FormatStrFormatter(LABELS[p][2]))
    for ax in fig.axes:
        ax.tick_params(length=3, width=0.7, top=True, right=True)
    fig.subplots_adjust(left=0.07, bottom=0.07, right=0.985, top=0.965, wspace=0.06, hspace=0.06)
    fig.text(0.97, 0.94, "GW170817 — mlgw_bns_jax (TEOBResumSPA surrogate)\nGWgpu-jax acceptance-walk NS",
             ha="right", va="top", fontsize=17, color=COLOR)
    fig.text(0.97, 0.885, "shaded: 50% and 90% credible regions\ntitles: median and 90% interval",
             ha="right", va="top", fontsize=13, color="0.35")
    for ext in ("png", "pdf"):
        fig.savefig(a.run_dir / f"{a.out}.{ext}", dpi=200 if ext == "png" else None)
    print("wrote", a.run_dir / f"{a.out}.png", "and .pdf")


if __name__ == "__main__":
    main()
