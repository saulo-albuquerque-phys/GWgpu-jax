#!/usr/bin/env python
"""Per-parameter Jensen-Shannon divergence between two posteriors (referee comment 1b).

Follows the LVK / pesummary convention: 1-D Gaussian KDEs evaluated on a shared
grid, JSD in **bits** (log base 2, range [0, 1]), bootstrap-resampled to give a
median and 90% interval. It also reports a **sampling-noise floor**: the JSD
between two random halves of the *same* run (A vs A), so a reader can tell
"different" from "finite-sample noise".

Reference scale: Romero-Shaw et al. 2020 (bilby validation, MNRAS 499, 3295) use
0.007 bits (two Gaussians whose means differ by ~20% of sigma) as the
"indistinguishable" threshold; GWTC analyses use ~0.05 bits to flag waveform
systematics.

Each posterior is given as SPEC = KIND:PATH, where KIND is
  gwjax   a gwjax outdir (posterior_samples.npz [+ result.json with start_time])
  npz     any .npz of 1-D arrays
  bilby   a bilby <label>_result.json
  txt     whitespace/tab table with a '#'-prefixed header (gwjax notebook export)
  h5      LVK PE release HDF5; use h5:PATH::GROUP (e.g. ::C01:IMRPhenomXPHM, or
          ::Overall_posterior for GWTC-1)

Parameter names default to the gwjax basis; translate the other side with
--map-b gwjax_name=other_name (repeatable). tc is shifted by --tc-offset-b.

Examples
  # frozen 4 s benchmark (gwjax vs bilby, identical data)
  python compute_jsd.py \
      --a gwjax:../frozen_benchmark/gwjax_frozen_4s --label-a gwjax \
      --b bilby:../frozen_benchmark/bilby_frozen_4s/frozen_4s_result.json --label-b bilby \
      --params Mc q chi_1 chi_2 distance inclination ra dec psi phi_c tc \
      --map-b Mc=chirp_mass q=mass_ratio distance=luminosity_distance \
              inclination=theta_jn phi_c=phase tc=geocent_time \
      --tc-offset-b auto --out frozen_4s_jsd.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------- io
def load(spec):
    kind, _, path = spec.partition(":")
    if kind in ("gwjax", "npz"):
        p = Path(path)
        f = p / "posterior_samples.npz" if p.is_dir() else p
        post = {k: np.asarray(v, float) for k, v in np.load(f).items()}
        meta = {}
        if p.is_dir() and (p / "result.json").exists():
            meta = json.load(open(p / "result.json"))
        return post, meta
    if kind == "bilby":
        r = json.load(open(path))
        c = r["posterior"]["content"]
        post = {}
        for k, v in c.items():
            try:
                post[k] = np.asarray(v, float)
            except (TypeError, ValueError):
                pass
        return post, r
    if kind == "txt":
        with open(path) as fh:
            header = fh.readline().lstrip("#").split()
        arr = np.loadtxt(path)
        return {n: arr[:, i] for i, n in enumerate(header)}, {}
    if kind == "h5":
        import h5py
        fpath, _, group = path.partition("::")
        with h5py.File(fpath, "r") as h:
            g = h[group] if group else h
            if isinstance(g, h5py.Group) and "posterior_samples" in g:
                g = g["posterior_samples"]
            data = g[()]
        return {n: np.asarray(data[n], float) for n in data.dtype.names}, {}
    raise ValueError(f"unknown kind {kind!r} in {spec!r}")


# -------------------------------------------------------------------------- jsd
def _kde_pdf(x, grid, bounds=None, period=None):
    """Gaussian KDE on `grid`. Hard prior bounds are handled by reflection (avoids
    the edge leakage that inflates JSD for railing params like q->1, chi, Lambda);
    periodic params (ra, phi_c, psi) are wrapped instead."""
    kde = gaussian_kde(x)
    pdf = kde(grid)
    if period is not None:
        pdf += kde(grid - period) + kde(grid + period)
    elif bounds is not None:
        lo, hi = bounds
        if lo is not None:
            pdf += kde(2 * lo - grid)
        if hi is not None:
            pdf += kde(2 * hi - grid)
    pdf = np.clip(pdf, 0, None)
    return pdf / np.trapz(pdf, grid)


def jsd_bits(x, y, bounds=None, n_grid=2000, period=None):
    # Grid spans the data (padded), clipped to the prior support: a grid set to the
    # full prior range under-resolves a narrow posterior and breaks normalisation.
    span = max(x.max(), y.max()) - min(x.min(), y.min())
    lo = min(x.min(), y.min()) - 0.1 * span
    hi = max(x.max(), y.max()) + 0.1 * span
    if bounds is not None:
        lo = max(lo, bounds[0]) if bounds[0] is not None else lo
        hi = min(hi, bounds[1]) if bounds[1] is not None else hi
    grid = np.linspace(lo, hi, n_grid)
    p = _kde_pdf(x, grid, bounds, period); q = _kde_pdf(y, grid, bounds, period)
    m = 0.5 * (p + q)
    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(a / b), 0.0)
        return np.trapz(t, grid)
    return float(np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0, 1))


def bootstrap_jsd(x, y, n_boot, n_sub, rng, bounds=None, period=None):
    n_sub = min(n_sub, len(x), len(y))
    out = [jsd_bits(rng.choice(x, n_sub, replace=False),
                    rng.choice(y, n_sub, replace=False), bounds, period=period)
           for _ in range(n_boot)]
    return np.percentile(out, [5, 50, 95])


def noise_floor(x, n_boot, n_sub, rng, bounds=None, period=None):
    n_sub = min(n_sub, len(x) // 2)
    out = []
    for _ in range(n_boot):
        perm = rng.permutation(len(x))
        out.append(jsd_bits(x[perm[:n_sub]], x[perm[n_sub:2 * n_sub]], bounds, period=period))
    return float(np.median(out))


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True); ap.add_argument("--label-a", default="A")
    ap.add_argument("--b", required=True); ap.add_argument("--label-b", default="B")
    ap.add_argument("--params", nargs="+", required=True)
    ap.add_argument("--map-a", nargs="*", default=[])
    ap.add_argument("--map-b", nargs="*", default=[])
    ap.add_argument("--bounds", nargs="*", default=[],
                    help="param=lo,hi prior bounds for KDE reflection (use '' for open)")
    ap.add_argument("--periodic", nargs="*", default=[],
                    help="param=period, wrapped instead of reflected (e.g. ra=6.283185307179586)")
    ap.add_argument("--tc-offset-a", type=float, default=0.0,
                    help="subtract from A's tc column (e.g. 0.01 for mlgw_bbh's 10 ms convention)")
    ap.add_argument("--tc-offset-b", default="0",
                    help="subtract from B's tc column; 'auto' = A's result.json start_time")
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--n-sub", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pa, ma = load(args.a); pb, mb = load(args.b)
    mapa = dict(s.split("=") for s in args.map_a)
    mapb = dict(s.split("=") for s in args.map_b)
    bounds = {}
    for s in args.bounds:
        k, v = s.split("="); lo, hi = v.split(",")
        bounds[k] = (float(lo) if lo else None, float(hi) if hi else None)
    periods = {k: float(v) for k, v in (s.split("=") for s in args.periodic)}
    tc_off = float(ma["start_time"]) if args.tc_offset_b == "auto" else float(args.tc_offset_b)

    print(f"{args.label_a}: {len(next(iter(pa.values())))} samples   "
          f"{args.label_b}: {len(next(iter(pb.values())))} samples")
    print(f"{'param':12s} {'JSD [bits]':>10s} {'90% CI':>19s}   "
          f"{'floor '+args.label_a:>12s} {'floor '+args.label_b:>12s} {'shift/σ':>8s}")
    rows = {}
    for n in args.params:
        x = pa[mapa.get(n, n)]
        y = pb[mapb.get(n, n)]
        if n == "tc":
            x = x - args.tc_offset_a
            y = y - tc_off
        b, per = bounds.get(n), periods.get(n)
        if per is not None:
            x, y, b = np.mod(x, per), np.mod(y, per), (0.0, per)
        lo, med, hi = bootstrap_jsd(x, y, args.n_boot, args.n_sub, rng, b, per)
        fa = noise_floor(x, max(10, args.n_boot // 5), args.n_sub, rng, b, per)
        fb = noise_floor(y, max(10, args.n_boot // 5), args.n_sub, rng, b, per)
        # physical size of the shift: median difference in units of the mean 1-sigma width
        sig = 0.5 * (np.subtract(*np.percentile(x, [84, 16])) + np.subtract(*np.percentile(y, [84, 16]))) / 2
        shift = float((np.median(x) - np.median(y)) / sig) if sig > 0 else float("nan")
        rows[n] = dict(jsd_median=med, jsd_5=lo, jsd_95=hi, floor_a=fa, floor_b=fb,
                       median_shift_sigma=shift)
        flag = "  <-- > 0.05" if med > 0.05 else ("  (> 0.007)" if med > 0.007 else "")
        print(f"{n:12s} {med:10.4f} [{lo:8.4f},{hi:8.4f}]   {fa:12.4f} {fb:12.4f} {shift:+8.2f}{flag}")
    if args.out:
        json.dump(dict(a=args.a, b=args.b, label_a=args.label_a, label_b=args.label_b,
                       n_sub=args.n_sub, n_boot=args.n_boot, jsd=rows),
                  open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
