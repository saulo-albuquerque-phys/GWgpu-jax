#!/usr/bin/env python
"""Step 1 of the gradient validation (pyseobnr environment): SEOBNRv5HM (2,2) waveforms at a few
reference points and at +-h_i steps in each intrinsic parameter, for central finite differences.

Each waveform is placed on mlgw-jax's time grid (4 s at 4096 Hz ending 0.12 s after merger) using
pyseobnr's own time array (t = 0 at merger, a smooth function of the parameters), then cubic-spline
interpolated onto the grid. Any constant offset between the two models' time origins is removed by
projecting out the t_c direction in compare_gradients_bbh.py.

    ~/.venvs/romjax-jim/bin/python gen_seob_fd_stencil.py --out seob_stencil.npz
"""
import argparse

import numpy as np
from pyseobnr.generate_waveform import GenerateWaveform
from scipy.interpolate import CubicSpline

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="seob_stencil.npz")
ap.add_argument("--f-start", type=float, default=8.0)
ap.add_argument("--srate", type=float, default=4096.0)
ap.add_argument("--duration", type=float, default=4.0)
ap.add_argument("--final-time", type=float, default=0.12)
a = ap.parse_args()

# reference points (detector-frame masses in Msun, aligned spins): GW150914-like and around it
POINTS = np.array([[39.0, 32.0, 0.0, 0.0],
                   [39.0, 32.0, 0.3, -0.3],
                   [45.0, 25.0, 0.2, 0.0],
                   [30.0, 25.0, -0.2, 0.2]])
NAMES = ["m1", "m2", "chi1", "chi2"]
# relative steps for the masses, absolute steps for the spins; three sizes to check FD convergence
STEPS = {"m1": [3e-4, 1e-3, 3e-3], "m2": [3e-4, 1e-3, 3e-3], "chi1": [3e-4, 1e-3, 3e-3], "chi2": [3e-4, 1e-3, 3e-3]}
DIST, INCL, PHI = 400.0, 0.6, 0.0

dt = 1.0 / a.srate
N = int(round(a.duration * a.srate))
t_grid = np.arange(N) * dt - a.duration + a.final_time


def seob(theta):
    m1, m2, c1, c2 = theta
    p = dict(mass1=m1, mass2=m2, spin1x=0., spin1y=0., spin1z=c1, spin2x=0., spin2y=0., spin2z=c2,
             distance=DIST, inclination=INCL, phi_ref=PHI, f22_start=a.f_start, f_ref=20.0,
             deltaT=dt, approximant="SEOBNRv5HM", mode_array=[(2, 2)])
    hp_ts, hc_ts = GenerateWaveform(p).generate_td_polarizations()
    hp, hc = np.asarray(hp_ts.data.data), np.asarray(hc_ts.data.data)
    # LAL/pyseobnr time convention: t = 0 at merger, set through the (continuous) epoch. Locating
    # the peak of h+^2 + hx^2 instead is NOT smooth: at nonzero inclination it oscillates at twice
    # the orbital frequency, and the argmax jumps between neighbouring parameter values.
    t = float(hp_ts.epoch) + np.arange(hp.size) * dt
    assert t[0] < t_grid[0], f"waveform too short for the grid at {theta}: starts at {t[0]:.2f} s"
    return CubicSpline(t, hp)(t_grid), CubicSpline(t, hc)(t_grid)


out = {"points": POINTS, "names": np.array(NAMES), "t_grid": t_grid, "dist": DIST, "incl": INCL}
for k, th in enumerate(POINTS):
    out[f"p{k}_hp"], out[f"p{k}_hc"] = seob(th)
    for j, nm in enumerate(NAMES):
        for s, rel in enumerate(STEPS[nm]):
            h = rel * th[j] if nm.startswith("m") else rel
            for sign, tag in ((+1, "plus"), (-1, "minus")):
                tt = th.copy(); tt[j] += sign * h
                out[f"p{k}_{nm}_s{s}_{tag}_hp"], out[f"p{k}_{nm}_s{s}_{tag}_hc"] = seob(tt)
            out[f"p{k}_{nm}_s{s}_h"] = h
    print(f"point {k} {th} done", flush=True)
out["steps"] = np.array([STEPS[n] for n in NAMES])
np.savez_compressed(a.out, **out)
print("wrote", a.out)
