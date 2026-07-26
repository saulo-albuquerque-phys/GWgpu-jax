#!/usr/bin/env python
"""Red/blue comparison of the frozen 4s benchmark: gwjax vs bilby+dynesty.

Overlays the two posteriors in one corner plot (blue = bilby + dynesty,
red = gwjax + acceptance-walk kernel, black lines = injected values), mirroring
Fig. 1 of Prathaban et al. 2025, and prints the evidence comparison.

    python make_comparison_plot.py \
        --gwjax gwjax_frozen_4s \
        --bilby bilby_frozen_4s/frozen_4s_result.json \
        --out   frozen_4s_comparison.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gwjax", required=True, help="gwjax outdir (result.json + posterior_samples.npz)")
parser.add_argument("--bilby", required=True, help="bilby <label>_result.json")
parser.add_argument("--out", default="frozen_4s_comparison.png")
args = parser.parse_args()

# ── Load gwjax ────────────────────────────────────────────────────────────────
gdir = Path(args.gwjax)
gres = json.load(open(gdir / "result.json"))
gpost = dict(np.load(gdir / "posterior_samples.npz"))
start_time = gres["start_time"]

# ── Load bilby ────────────────────────────────────────────────────────────────
bres = json.load(open(args.bilby))
bpost_all = bres["posterior"]["content"]
def bcol(name):
    return np.asarray(bpost_all[name], dtype=float)

# Common parameter basis: gwjax name -> (bilby name, transform on bilby column)
MAP = {
    "Mc":          ("chirp_mass", None),
    "q":           ("mass_ratio", None),
    "chi_1":       ("chi_1", None),
    "chi_2":       ("chi_2", None),
    "distance":    ("luminosity_distance", None),
    "inclination": ("theta_jn", None),
    "ra":          ("ra", None),
    "dec":         ("dec", None),
    "psi":         ("psi", None),
    "phi_c":       ("phase", None),
    "tc":          ("geocent_time", lambda x: x - start_time),
}
LABELS = {
    "Mc": r"$M_c\,[M_\odot]$", "q": r"$q$", "chi_1": r"$\chi_1$",
    "chi_2": r"$\chi_2$", "distance": r"$d_L$ [Mpc]",
    "inclination": r"$\theta_{JN}$", "ra": r"$\alpha$", "dec": r"$\delta$",
    "psi": r"$\psi$", "phi_c": r"$\phi_c$", "tc": r"$t_c$ [s]",
}
inj = gres["injection"]
TRUTH = dict(Mc=inj["chirp_mass"], q=inj["mass_ratio"], chi_1=inj["chi_1"],
             chi_2=inj["chi_2"], distance=inj["luminosity_distance"],
             inclination=inj["theta_jn"], ra=inj["ra"], dec=inj["dec"],
             psi=inj["psi"], phi_c=inj["phase"],
             tc=inj["geocent_time"] - start_time)

names = list(MAP)
g_data = np.column_stack([np.asarray(gpost[n]) for n in names])
b_data = np.column_stack([
    MAP[n][1](bcol(MAP[n][0])) if MAP[n][1] else bcol(MAP[n][0]) for n in names])
print(f"gwjax samples: {g_data.shape[0]}   bilby samples: {b_data.shape[0]}")

# Shared plot ranges covering both posteriors
lo = np.minimum(g_data.min(axis=0), b_data.min(axis=0))
hi = np.maximum(g_data.max(axis=0), b_data.max(axis=0))
pad = 0.03 * (hi - lo)
ranges = list(zip(lo - pad, hi + pad))

ckw = dict(labels=[LABELS[n] for n in names], range=ranges,
           truths=[TRUTH[n] for n in names], truth_color="black",
           plot_datapoints=False, plot_density=False, levels=(0.393, 0.865, 0.989),
           smooth=0.9, bins=40)

fig = corner.corner(b_data, color="#3f4bd8",
                    hist_kwargs=dict(density=True, color="#3f4bd8"), **ckw)
corner.corner(g_data, fig=fig, color="#d8443f",
              hist_kwargs=dict(density=True, color="#d8443f"), **ckw)

import matplotlib.lines as mlines
fig.legend(handles=[
    mlines.Line2D([], [], color="#3f4bd8", label="bilby + dynesty"),
    mlines.Line2D([], [], color="#d8443f", label="gwjax + acceptance-walk (blackjax-ns)"),
], loc="upper right", fontsize=14, frameon=False)
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print("saved", args.out)

# ── Evidence comparison ───────────────────────────────────────────────────────
print("\n── log-evidence ─────────────────────────────────────────────")
print(f"bilby : logZ = {bres['log_evidence']:.4f} +/- {bres['log_evidence_err']:.4f}   "
      f"logBF = {bres['log_bayes_factor']:.4f}   (logZ_noise = {bres['log_noise_evidence']:.4f})")
print(f"gwjax : logZ = {gres['logZ']:.4f} +/- {gres['logZ_err']:.4f}   "
      f"logBF = {gres['log_bayes_factor']:.4f}   (logZ_noise = {gres['log_noise_evidence']:.4f})")
dbf = gres["log_bayes_factor"] - bres["log_bayes_factor"]
sig = np.hypot(gres["logZ_err"], bres["log_evidence_err"])
print(f"Delta logBF (gwjax - bilby) = {dbf:+.4f}  ({dbf/sig:+.1f} sigma of combined error)")

# ── One-dimensional agreement summary ────────────────────────────────────────
print("\n── per-parameter median [16th, 84th] ────────────────────────")
print(f"{'param':12s} {'bilby':>28s} {'gwjax':>28s}")
for j, n in enumerate(names):
    bq = np.percentile(b_data[:, j], [16, 50, 84])
    gq = np.percentile(g_data[:, j], [16, 50, 84])
    print(f"{n:12s} {bq[1]:10.4f} [{bq[0]:9.4f},{bq[2]:9.4f}]"
          f" {gq[1]:10.4f} [{gq[0]:9.4f},{gq[2]:9.4f}]")
