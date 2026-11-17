#!/usr/bin/env python
"""Step 1b of the gradient validation (lalsuite environment with SEOBNRv5ROM_v1.0.hdf5 on
LAL_DATA_PATH): SEOBNRv5_ROM (the (2,2)-mode reduced-order model of SEOBNRv5) at the same reference
points and finite-difference steps as gen_seob_fd_stencil.py, in the same output format.

Why the ROM: SEOBNRv5HM (pyseobnr) integrates ODEs and attaches the ringdown on a discrete grid; its
waveforms are not smooth in the parameters at the 1e-5 relative level (finite differences with steps
< 1e-4 blow up, tighter ODE tolerances do not help), which is far too coarse once the time/phase
directions are projected out. The ROM is a smooth interpolant of SEOBNRv5 (2,2), so its finite
differences converge.

The ROM is evaluated in the frequency domain on a 32 s grid, transformed to the time domain (LAL
convention: merger near t = 0, negative times wrapped to the end), cubic-spline interpolated onto
mlgw-jax's 4 s grid; the comparison script then applies the identical taper and FFT to both models.

    LAL_DATA_PATH=~/lal_data python gen_rom_fd_stencil.py --out rom_stencil.npz
"""
import argparse

import lal
import lalsimulation as ls
import numpy as np
from scipy.interpolate import CubicSpline

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="rom_stencil.npz")
ap.add_argument("--srate", type=float, default=4096.0)
ap.add_argument("--duration", type=float, default=4.0)
ap.add_argument("--final-time", type=float, default=0.12)
ap.add_argument("--long", type=float, default=32.0, help="length of the FD generation segment [s]")
ap.add_argument("--f-start", type=float, default=8.0)
a = ap.parse_args()

POINTS = np.array([[39.0, 32.0, 0.0, 0.0],
                   [39.0, 32.0, 0.3, -0.3],
                   [45.0, 25.0, 0.2, 0.0],
                   [30.0, 25.0, -0.2, 0.2]])
NAMES = ["m1", "m2", "chi1", "chi2"]
STEPS = {"m1": [3e-4, 1e-3, 3e-3], "m2": [3e-4, 1e-3, 3e-3], "chi1": [3e-4, 1e-3, 3e-3], "chi2": [3e-4, 1e-3, 3e-3]}
DIST, INCL, PHI = 400.0, 0.6, 0.0

dt = 1.0 / a.srate
N = int(round(a.duration * a.srate))
t_grid = np.arange(N) * dt - a.duration + a.final_time
NL = int(round(a.long * a.srate))
dF = 1.0 / a.long
APPROX = ls.GetApproximantFromString("SEOBNRv5_ROM")
t_long = np.arange(NL) * dt
t_long = np.where(t_long > a.long / 2, t_long - a.long, t_long)      # wrap negative times
order = np.argsort(t_long)


def rom(theta):
    m1, m2, c1, c2 = theta
    hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, c1, 0, 0, c2,
                                            DIST * 1e6 * lal.PC_SI, INCL, PHI, 0, 0, 0, dF, a.f_start,
                                            a.srate / 2, 20.0, lal.CreateDict(), APPROX)
    out = []
    for h in (hp, hc):
        fd = np.zeros(NL // 2 + 1, complex)
        n = min(h.data.length, fd.size)
        fd[:n] = h.data.data[:n]
        td = np.fft.irfft(fd, n=NL) / dt                 # LAL FD convention: h(f) = dt * FFT
        out.append(CubicSpline(t_long[order], td[order])(t_grid))
    return out


out = {"points": POINTS, "names": np.array(NAMES), "t_grid": t_grid, "dist": DIST, "incl": INCL,
       "model": "SEOBNRv5_ROM"}
for k, th in enumerate(POINTS):
    out[f"p{k}_hp"], out[f"p{k}_hc"] = rom(th)
    for j, nm in enumerate(NAMES):
        for s, rel in enumerate(STEPS[nm]):
            h = rel * th[j] if nm.startswith("m") else rel
            for sign, tag in ((+1, "plus"), (-1, "minus")):
                tt = th.copy(); tt[j] += sign * h
                out[f"p{k}_{nm}_s{s}_{tag}_hp"], out[f"p{k}_{nm}_s{s}_{tag}_hc"] = rom(tt)
            out[f"p{k}_{nm}_s{s}_h"] = h
    print(f"point {k} {th} done", flush=True)
out["steps"] = np.array([STEPS[n] for n in NAMES])
np.savez_compressed(a.out, **out)
print("wrote", a.out)
