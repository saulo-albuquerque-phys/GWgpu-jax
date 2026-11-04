#!/usr/bin/env python
"""Corner-plot comparison of the mlgw_bbh case's posteriors.

Overlays whichever of the following are supplied (`--bilby` and
`--gwjax-ripple` are both optional, so this also works before the bilby
result is ready — e.g. to compare the two gwjax runs from
COLAB_run_gwjax_bbh.ipynb on their own):

* red   = gwjax + acceptance-walk + mlgw_bbh_jax [SEOBNRv5HM, (2,2) mode]  (--gwjax, required)
* blue  = bilby + dynesty + SEOBNRv5_ROM                                   (--bilby, optional)
* green = gwjax + acceptance-walk + ripplegw IMRPhenomXAS                  (--gwjax-ripple, optional)

black lines = injected values. Prints the evidence comparison for whichever
pairs are available. Adapted from frozen_benchmark/make_comparison_plot.py —
only the parameter map changed (mc/q/chi_1/chi_2 need no transform on either
side, since both gwjax's mlgw_bbh sampling basis and bilby's aligned-spin
BBH priors already use these exact names).

    python make_comparison_plot.py \
        --gwjax gwjax_bbh_case \
        --bilby bilby_bbh_case/bbh_case_result.json \
        --gwjax-ripple gwjax_bbh_case_ripple \
        --out   bbh_case_comparison.png

Just the two gwjax runs, no bilby yet:

    python make_comparison_plot.py \
        --gwjax gwjax_bbh_case \
        --gwjax-ripple gwjax_bbh_case_ripple \
        --out   bbh_case_gwjax_only.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import matplotlib.lines as mlines

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gwjax", required=True, help="gwjax outdir (result.json + posterior_samples.npz)")
parser.add_argument("--bilby", default=None, help="optional bilby <label>_result.json")
parser.add_argument("--gwjax-ripple", default=None,
                    help="optional gwjax-with-ripplegw outdir (same schema as --gwjax)")
parser.add_argument("--out", default="bbh_case_comparison.png")
args = parser.parse_args()

names = ["mc", "q", "distance", "dec", "ra", "inclination",
         "chi_1", "chi_2", "tc", "psi", "phi_c"]
LABELS = {
    "mc": r"$M_c$", "q": r"$q$", "chi_1": r"$\chi_1$",
    "chi_2": r"$\chi_2$", "distance": r"$d_L$",
    "inclination": r"$\theta_{JN}$", "ra": r"$\alpha$", "dec": r"$\delta$",
    "psi": r"$\psi$", "phi_c": r"$\phi_c$", "tc": r"$t_c$",
}

# ── Load gwjax (mlgw_bbh_jax) — always required ─────────────────────────────
gdir = Path(args.gwjax)
gres = json.load(open(gdir / "result.json"))
gpost = dict(np.load(gdir / "posterior_samples.npz"))
start_time = gres["start_time"]
inj = gres["injection"]

# mlgw_bbh_jax's TD->FD recipe (zero-crossing time grid + rFFT's implicit
# "index 0 = time 0" framing) reads out tc about MLGW_FINAL_TIME = 10 ms
# HIGHER than the standard tc=0-at-projection convention ripple/bilby use,
# for the identical physical merger time — characterized exactly by
# generate_injection.py's ripplegw triangulation check (best dtc = -10.00 ms
# needed on the ripple side to match mlgw's injected signal). Rigid
# correction (not a fit) so gwjax's tc panel compares like-for-like against
# both bilby and gwjax-ripple, both of which already use the standard
# convention.
# = the mlgw_final_time the run used (stored in result.json config since Nov 2026;
# older runs used 0.01 s).
TC_OFFSET_MLGW = float(gres.get("config", {}).get("mlgw_final_time", 0.01))
gpost = dict(gpost)
gpost["tc"] = gpost["tc"] - TC_OFFSET_MLGW

TRUTH = dict(mc=inj["chirp_mass"], q=inj["mass_ratio"], chi_1=inj["chi_1"],
             chi_2=inj["chi_2"], distance=inj["luminosity_distance"],
             inclination=inj["theta_jn"], ra=inj["ra"], dec=inj["dec"],
             psi=inj["psi"], phi_c=inj["phase"],
             # old injections (no mlgw_final_time in config) used mlgw tc = trigger, so their
             # physical merger was 10 ms early; new ones put the merger on the trigger.
             tc=inj["geocent_time"] - start_time
                - (0.0 if "mlgw_final_time" in gres.get("config", {}) else TC_OFFSET_MLGW))
g_data = np.column_stack([np.asarray(gpost[n]) for n in names])

# ── Load gwjax-ripple (optional) ─────────────────────────────────────────────
rdir = Path(args.gwjax_ripple) if args.gwjax_ripple else None
if rdir:
    rres = json.load(open(rdir / "result.json"))
    rpost = dict(np.load(rdir / "posterior_samples.npz"))
    r_data = np.column_stack([np.asarray(rpost[n]) for n in names])

# ── Load bilby (optional) ────────────────────────────────────────────────────
bilby_ok = args.bilby is not None
if bilby_ok:
    bres = json.load(open(args.bilby))
    bpost_all = bres["posterior"]["content"]
    def bcol(name):
        return np.asarray(bpost_all[name], dtype=float)
    # gwjax name -> (bilby name, transform on bilby column)
    MAP = {
        "mc": ("chirp_mass", None), "q": ("mass_ratio", None),
        "chi_1": ("chi_1", None), "chi_2": ("chi_2", None),
        "distance": ("luminosity_distance", None),
        "inclination": ("theta_jn", None), "ra": ("ra", None), "dec": ("dec", None),
        "psi": ("psi", None), "phi_c": ("phase", None),
        "tc": ("geocent_time", lambda x: x - start_time),
    }
    b_data = np.column_stack([
        MAP[n][1](bcol(MAP[n][0])) if MAP[n][1] else bcol(MAP[n][0]) for n in names])

sample_counts = [f"gwjax: {g_data.shape[0]}"]
if bilby_ok:
    sample_counts.append(f"bilby: {b_data.shape[0]}")
if rdir:
    sample_counts.append(f"gwjax-ripple: {r_data.shape[0]}")
print("samples —", "   ".join(sample_counts))

# ── Shared plot ranges covering every posterior present ─────────────────────
all_data = [g_data] + ([b_data] if bilby_ok else []) + ([r_data] if rdir else [])
lo = np.minimum.reduce([d.min(axis=0) for d in all_data])
hi = np.maximum.reduce([d.max(axis=0) for d in all_data])
pad = 0.03 * (hi - lo)
ranges = list(zip(lo - pad, hi + pad))

ckw = dict(labels=[LABELS[n] for n in names], range=ranges,
           truths=[TRUTH[n] for n in names], truth_color="black",
           plot_datapoints=False, plot_density=False, levels=(0.393, 0.865, 0.989),
           smooth=0.9, bins=40)

fig = None
legend_handles = []
if bilby_ok:
    fig = corner.corner(b_data, color="#3f4bd8", hist_kwargs=dict(density=True, color="#3f4bd8"), **ckw)
    legend_handles.append(mlines.Line2D([], [], color="#3f4bd8", label="bilby + dynesty (SEOBNRv5_ROM)"))
corner.corner(g_data, fig=fig, color="#d8443f", hist_kwargs=dict(density=True, color="#d8443f"), **ckw)
if fig is None:
    fig = plt.gcf()
legend_handles.append(mlines.Line2D([], [], color="#d8443f", label="gwjax + acceptance-walk (mlgw_bbh_jax)"))
if rdir:
    corner.corner(r_data, fig=fig, color="#3fa84b", hist_kwargs=dict(density=True, color="#3fa84b"), **ckw)
    legend_handles.append(
        mlines.Line2D([], [], color="#3fa84b", label="gwjax + acceptance-walk (ripplegw IMRPhenomXAS)"))

fig.legend(handles=legend_handles, loc="upper right", fontsize=14, frameon=False)
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print("saved", args.out)

# ── Evidence comparison ───────────────────────────────────────────────────────
print("\n── log-evidence ─────────────────────────────────────────────")
if bilby_ok:
    print(f"bilby       : logZ = {bres['log_evidence']:.4f} +/- {bres['log_evidence_err']:.4f}   "
          f"logBF = {bres['log_bayes_factor']:.4f}   (logZ_noise = {bres['log_noise_evidence']:.4f})")
print(f"gwjax       : logZ = {gres['logZ']:.4f} +/- {gres['logZ_err']:.4f}   "
      f"logBF = {gres['log_bayes_factor']:.4f}   (logZ_noise = {gres['log_noise_evidence']:.4f})")
if rdir:
    print(f"gwjax-ripple: logZ = {rres['logZ']:.4f} +/- {rres['logZ_err']:.4f}   "
          f"logBF = {rres['log_bayes_factor']:.4f}   (logZ_noise = {rres['log_noise_evidence']:.4f})")
if bilby_ok:
    dbf = gres["log_bayes_factor"] - bres["log_bayes_factor"]
    sig = np.hypot(gres["logZ_err"], bres["log_evidence_err"])
    print(f"Delta logBF (gwjax - bilby)        = {dbf:+.4f}  ({dbf/sig:+.1f} sigma of combined error)")
    if rdir:
        drb = rres["log_bayes_factor"] - bres["log_bayes_factor"]
        sig_rb = np.hypot(rres["logZ_err"], bres["log_evidence_err"])
        print(f"Delta logBF (gwjax-ripple - bilby)  = {drb:+.4f}  ({drb/sig_rb:+.1f} sigma) "
              "— pipeline+waveform-family systematic only, no surrogate")
if rdir:
    drg = rres["log_bayes_factor"] - gres["log_bayes_factor"]
    sig_rg = np.hypot(rres["logZ_err"], gres["logZ_err"])
    print(f"Delta logBF (gwjax-ripple - gwjax)  = {drg:+.4f}  ({drg/sig_rg:+.1f} sigma) "
          "— surrogate-vs-ripple only, same pipeline")
if not bilby_ok:
    print("(no --bilby given — bilby comparison skipped; this is a gwjax-only surrogate-vs-ripple check)")

# ── One-dimensional agreement summary ────────────────────────────────────────
print("\n── per-parameter median [16th, 84th] ────────────────────────")
header = f"{'param':12s} {'gwjax':>28s}"
if bilby_ok:
    header += f" {'bilby':>28s}"
if rdir:
    header += f" {'gwjax-ripple':>28s}"
print(header)
for j, n in enumerate(names):
    gq = np.percentile(g_data[:, j], [16, 50, 84])
    line = f"{n:12s} {gq[1]:10.4f} [{gq[0]:9.4f},{gq[2]:9.4f}]"
    if bilby_ok:
        bq = np.percentile(b_data[:, j], [16, 50, 84])
        line += f" {bq[1]:10.4f} [{bq[0]:9.4f},{bq[2]:9.4f}]"
    if rdir:
        rq = np.percentile(r_data[:, j], [16, 50, 84])
        line += f" {rq[1]:10.4f} [{rq[0]:9.4f},{rq[2]:9.4f}]"
    print(line)
