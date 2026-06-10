#!/usr/bin/env python3
"""Reproduce the original two-curve FD evaluation-time CDF.

Conditions that reproduce the old plot (mlgw_bns_jax faster than original):
- small frequency grid (default 64 points)
- JAX inputs prebuilt as device arrays (per-call jnp.array construction excluded
  from the timed region, matching the original benchmark)

Writes exec_time_cdf_repro.png.
"""
from __future__ import annotations

import argparse
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != BASE]

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mlgw_bns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-freqs", type=int, default=64)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--bins", type=int, default=50,
                    help="number of CDF steps (lower = chunkier)")
    ap.add_argument("--xmin", type=float, default=-2.8)
    ap.add_argument("--xmax", type=float, default=-2.0)
    ap.add_argument("--output", type=Path, default=BASE / "exec_time_cdf_repro.png")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("jip", BASE / "jax_import_n_predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    predict = jax.jit(mod.load_predict(str(BASE / "mlgw_bns_jax_model.h5")))
    model = mlgw_bns.Model.default()

    freqs = np.linspace(23.0, 1500.0, args.n_freqs)
    fj = jnp.asarray(freqs)
    rng = np.random.default_rng(args.seed)

    P, MT, D, INC, PE = [], [], [], [], []
    for _ in range(args.n):
        q, l1, l2 = rng.uniform(1, 2), rng.uniform(5, 1200), rng.uniform(5, 1200)
        c1, c2 = rng.uniform(-.1, .1), rng.uniform(-.1, .1)
        mt, d, inc = rng.uniform(2.4, 3.1), rng.uniform(20, 120), rng.uniform(0, np.pi)
        P.append(jnp.array([q, l1, l2, c1, c2], float))
        MT.append(jnp.array(mt)); D.append(jnp.array(d)); INC.append(jnp.array(inc))
        PE.append(mlgw_bns.ParametersWithExtrinsic(
            mass_ratio=q, lambda_1=l1, lambda_2=l2, chi_1=c1, chi_2=c2,
            distance_mpc=d, inclination=inc, total_mass=mt))

    jax.block_until_ready(predict(P[0], fj, total_mass=MT[0], distance_mpc=D[0], inclination=INC[0]))

    tj, tr = [], []
    for i in range(args.n):
        t0 = time.perf_counter()
        h = predict(P[i], fj, total_mass=MT[i], distance_mpc=D[i], inclination=INC[i])
        jax.block_until_ready(h)
        tj.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        model.predict(freqs, PE[i])
        tr.append(time.perf_counter() - t0)

    tj = np.array(tj); tr = np.array(tr)

    # Binned (stepped) CDF over the plotted window so the curve shows gentle
    # discrete steps rather than a smooth line.
    edges = np.linspace(args.xmin, args.xmax, args.bins + 1)

    def step_cdf(x):
        xs = np.sort(np.log10(x))
        return edges, np.searchsorted(xs, edges, side="right") / xs.size

    ej, yj = step_cdf(tj); er, yr = step_cdf(tr)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.step(ej, yj, where="post", lw=1.6, color="tab:green")
    ax.step(er, yr, where="post", lw=1.6, color="tab:blue")
    ax.axvline(np.log10(np.median(tj)), ls=":", color="tab:green", lw=1.3)
    ax.axvline(np.log10(np.median(tr)), ls=":", color="tab:blue", lw=1.3)
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_xlabel(r"$\log_{10}$[ Evaluation Time [s] ]")
    ax.set_ylabel("Cumulative distribution")

    # Legend: colour shown by hollow (unfilled) rectangles; package names in a
    # monospace ("code") font.
    handles = [
        mpatches.Patch(facecolor="none", edgecolor="tab:green", label="mlgw_bns_jax (FD)"),
        mpatches.Patch(facecolor="none", edgecolor="tab:blue", label="mlgw-bns original (FD)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              prop={"family": "monospace"})
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)

    print(f"n_freqs={args.n_freqs}  n={args.n}")
    print(f"  jax median = {1e3*np.median(tj):.3f} ms  (log10 {np.log10(np.median(tj)):+.2f})")
    print(f"  ref median = {1e3*np.median(tr):.3f} ms  (log10 {np.log10(np.median(tr)):+.2f})")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
