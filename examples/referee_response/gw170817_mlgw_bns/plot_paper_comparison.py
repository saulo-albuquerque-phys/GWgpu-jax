#!/usr/bin/env python
"""Paper figure: GW170817 posterior of the GPU run (mlgw_bns_jax + GWgpu-jax) overlaid on the
bilby/dynesty TEOBResumSPA reference (same data, priors, likelihood settings) and on the GWTC-1
IMRPhenomPv2_NRTidal low-spin samples.

    python plot_paper_comparison.py RUN_DIR [--bilby RESULT.json] [--gwtc1 GWTC-1.hdf5] [--out NAME]

Writes RUN_DIR/NAME.pdf and .png. Contours: 50% and 90% credible regions. 1D marginals are
Gaussian KDEs with reflection at the prior bounds; all three analyses get the same smoothing.
"""
import argparse
import json
from pathlib import Path

import corner
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
PI = np.pi
# name: (label, summary format, tick format, prior bounds used for KDE reflection)
PARAMS = {
    "mc":           (r"$\mathcal{M}^{\rm det}\,[M_\odot]$", "{:.4f}", "%.4f", (None, None)),
    "q":            (r"$q$", "{:.2f}", "%.2f", (None, 1.0)),
    "chi_eff":      (r"$\chi_{\rm eff}$", "{:.3f}", "%.2f", (None, None)),
    "lambda_tilde": (r"$\tilde{\Lambda}$", "{:.0f}", "%.0f", (0.0, None)),
    "distance":     (r"$d_L\,[\mathrm{Mpc}]$", "{:.1f}", "%.0f", (None, None)),
    "inclination":  (r"$\theta_{JN}\,[\mathrm{rad}]$", "{:.2f}", "%.1f", (None, PI)),
}
C_GWJAX, C_BILBY, C_GWTC = "#1b4f8a", "#d1495b", "0.45"


def lam_tilde(m1, m2, l1, l2):
    return 16 / 13 * ((m1 + 12 * m2) * m1**4 * l1 + (m2 + 12 * m1) * m2**4 * l2) / (m1 + m2) ** 5


def load_gwjax(run_dir):
    p = np.load(run_dir / "posterior_samples.npz")
    return {k: np.asarray(p[k]) for k in PARAMS}


def load_bilby(path):
    p = json.load(open(path))["posterior"]["content"]
    g = lambda k: np.asarray(p[k], dtype=float)
    m1, m2 = g("mass_1"), g("mass_2")
    return dict(mc=g("chirp_mass"), q=g("mass_ratio"), chi_eff=(m1 * g("chi_1") + m2 * g("chi_2")) / (m1 + m2),
                lambda_tilde=lam_tilde(m1, m2, g("lambda_1"), g("lambda_2")),
                distance=g("luminosity_distance"), inclination=g("theta_jn"))


def load_gwtc1(path):
    import h5py
    with h5py.File(path, "r") as f:
        g = f["IMRPhenomPv2NRT_lowSpin_posterior"][()]
    m1, m2 = g["m1_detector_frame_Msun"], g["m2_detector_frame_Msun"]
    return dict(mc=(m1 * m2) ** 0.6 / (m1 + m2) ** 0.2, q=m2 / m1,
                chi_eff=(m1 * g["spin1"] * g["costilt1"] + m2 * g["spin2"] * g["costilt2"]) / (m1 + m2),
                lambda_tilde=lam_tilde(m1, m2, g["lambda1"], g["lambda2"]),
                distance=g["luminosity_distance_Mpc"], inclination=np.arccos(g["costheta_jn"]))


def kde1d(x, grid, lo=None, hi=None):
    """Gaussian KDE with reflection at hard prior bounds (no spurious roll-off at q = 1, etc.)."""
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
    ap.add_argument("--bilby", type=Path,
                    default=HERE / "bilby_gw170817_TEOBResumSPA_gwtc1_lowspin/gw170817_TEOBResumSPA_result.json")
    ap.add_argument("--gwtc1", type=Path, default=HERE.parents[1] / "data_GW170817/GW170817_GWTC-1 (2).hdf5")
    ap.add_argument("--no-gwtc1", action="store_true")
    ap.add_argument("--smooth", type=float, default=1.6, help="2D Gaussian smoothing (bins)")
    ap.add_argument("--out", default="paper_gw170817_comparison")
    a = ap.parse_args()

    sets = [("GWgpu-jax + mlgw_bns_jax (GPU)", load_gwjax(a.run_dir), C_GWJAX, "-", True),
            ("bilby/dynesty + TEOBResumSPA (CPU)", load_bilby(a.bilby), C_BILBY, "-", False)]
    if not a.no_gwtc1:
        sets.append(("GWTC-1, IMRPhenomPv2_NRTidal", load_gwtc1(a.gwtc1), C_GWTC, "--", False))
    names = list(PARAMS)
    n = len(names)

    # axis range: wide enough to contain every analysis' 99.8% interval, clipped at the prior bounds
    rng = []
    for p in names:
        lo = min(np.quantile(s[1][p], 0.001) for s in sets)
        hi = max(np.quantile(s[1][p], 0.999) for s in sets)
        pad = 0.04 * (hi - lo)
        blo, bhi = PARAMS[p][3]
        rng.append((max(lo - pad, blo) if blo is not None else lo - pad,
                    min(hi + pad, bhi) if bhi is not None else hi + pad))

    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 13,
        "axes.labelsize": 19, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    })
    fig = None
    for label, s, col, ls, fill in sets:
        X = np.column_stack([s[p] for p in names])
        fig = corner.corner(
            X, fig=fig, range=rng, bins=40, smooth=a.smooth, levels=(0.5, 0.9),
            labels=[PARAMS[p][0] for p in names], color=col, max_n_ticks=4, labelpad=0.06,
            plot_datapoints=False, plot_density=False, fill_contours=fill, no_fill_contours=not fill,
            contour_kwargs=dict(linewidths=1.5 if ls == "-" else 1.3, linestyles=ls),
            hist_kwargs=dict(alpha=0.0),            # 1D panels are redrawn below with KDEs
        )
    axes = np.asarray(fig.axes).reshape(n, n)
    for i, p in enumerate(names):
        ax = axes[i, i]
        for art in list(ax.patches) + list(ax.lines):
            art.remove()
        grid = np.linspace(*rng[i], 600)
        top = 0.0
        for label, s, col, ls, fill in sets:
            y = kde1d(s[p], grid, *PARAMS[p][3])
            ax.plot(grid, y, color=col, lw=1.8 if ls == "-" else 1.5, ls=ls)
            if fill:
                ax.fill_between(grid, 0, y, color=col, alpha=0.15, lw=0)
            top = max(top, y.max())
        ax.set_xlim(rng[i]); ax.set_ylim(0, 1.08 * top); ax.set_yticks([])
        axes[n - 1, i].xaxis.set_major_locator(MaxNLocator(4, prune="both"))
        axes[n - 1, i].xaxis.set_major_formatter(FormatStrFormatter(PARAMS[p][2]))
        if i > 0:
            axes[i, 0].yaxis.set_major_locator(MaxNLocator(4, prune="both"))
            axes[i, 0].yaxis.set_major_formatter(FormatStrFormatter(PARAMS[p][2]))
    for ax in fig.axes:
        ax.tick_params(length=3.5, width=0.7, top=True, right=True)
    fig.subplots_adjust(left=0.085, bottom=0.085, right=0.985, top=0.985, wspace=0.05, hspace=0.05)

    # legend + median / 90% interval table in the empty upper-right triangle
    handles = [mlines.Line2D([], [], color=col, ls=ls, lw=2.2, label=label) for label, _, col, ls, _ in sets]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.975), fontsize=17, frameon=False)
    y0, dy = 0.835, 0.030
    fig.text(0.975, y0 + dy, "median and 90% credible interval", ha="right", fontsize=13, color="0.3")
    for r, p in enumerate(names):
        f = PARAMS[p][1]
        fig.text(0.60, y0 - r * dy, PARAMS[p][0].split(r"\,[")[0] + ("$" if r"\,[" in PARAMS[p][0] else ""),
                 ha="right", fontsize=14)
        for c, (label, s, col, ls, fill) in enumerate(sets):
            lo, med, hi = np.quantile(s[p], [0.05, 0.5, 0.95])
            fig.text(0.63 + 0.125 * c, y0 - r * dy,
                     f"${f.format(med)}^{{+{f.format(hi - med)}}}_{{-{f.format(med - lo)}}}$", fontsize=13.5, color=col)
            print(f"{p:13s} {label[:22]:22s} {f.format(med)} +{f.format(hi - med)} -{f.format(med - lo)}")
    for ext in ("pdf", "png"):
        fig.savefig(a.run_dir / f"{a.out}.{ext}", dpi=220 if ext == "png" else None)
    print("wrote", a.run_dir / f"{a.out}.pdf", "and .png")


if __name__ == "__main__":
    main()
