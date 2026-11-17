#!/usr/bin/env python
"""Step 1 (BNS, teobresums environment): TEOBResumSPA (the model mlgw-bns-jax is trained on) at a few
GW170817-like reference points and at +-h_i steps in each intrinsic parameter, on the GW170817 analysis
frequencies (23-1500 Hz, df = 1/128 Hz), for central finite differences.

TEOBResumSPA finite differences are stable for relative mass steps of 1e-7 to 1e-5; much larger steps
leave the linear regime because a BNS accumulates thousands of cycles in band. Steps are chosen
accordingly (three sizes per parameter to check convergence).

    python gen_teob_fd_stencil.py --out teob_stencil.npz      (needs teobresumspa_source.py next to it)
"""
import argparse

import numpy as np
from teobresumspa_source import teobresumspa_polarizations

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="teob_stencil.npz")
ap.add_argument("--f-min", type=float, default=23.0)
ap.add_argument("--f-max", type=float, default=1500.0)
ap.add_argument("--duration", type=float, default=128.0)
a = ap.parse_args()

# (m1, m2, chi1, chi2, Lambda1, Lambda2), detector-frame masses, m1 >= m2
POINTS = np.array([[1.40, 1.18, 0.00, 0.00, 300.0, 300.0],
                   [1.50, 1.10, 0.02, -0.02, 600.0, 900.0],
                   [1.36, 1.30, 0.00, 0.00, 100.0, 150.0]])
NAMES = ["m1", "m2", "chi1", "chi2", "lambda1", "lambda2"]
STEPS = {"m1": [1e-6, 3e-6, 1e-5], "m2": [1e-6, 3e-6, 1e-5],            # relative
         "chi1": [1e-5, 3e-5, 1e-4], "chi2": [1e-5, 3e-5, 1e-4],        # absolute
         "lambda1": [1e-3, 3e-3, 1e-2], "lambda2": [1e-3, 3e-3, 1e-2]}  # relative
DIST, INCL, PHI = 40.0, 2.5, 0.0

f = np.arange(int(round(a.f_min * a.duration)), int(round(a.f_max * a.duration)) + 1) / a.duration


def teob(th):
    m1, m2, c1, c2, l1, l2 = th
    hp, _ = teobresumspa_polarizations(f, m1, m2, c1, c2, l1, l2, DIST, INCL, PHI, 0.9 * a.f_min)
    return hp


out = {"points": POINTS, "names": np.array(NAMES), "freqs": f, "dist": DIST, "incl": INCL, "model": "TEOBResumSPA"}
for k, th in enumerate(POINTS):
    out[f"p{k}_hp"] = teob(th)
    for j, nm in enumerate(NAMES):
        for s, rel in enumerate(STEPS[nm]):
            h = rel if nm.startswith("chi") else rel * th[j]
            for sign, tag in ((+1, "plus"), (-1, "minus")):
                tt = th.copy(); tt[j] += sign * h
                out[f"p{k}_{nm}_s{s}_{tag}_hp"] = teob(tt)
            out[f"p{k}_{nm}_s{s}_h"] = h
    print(f"point {k} {th} done", flush=True)
np.savez_compressed(a.out, **out)
print("wrote", a.out)
