#!/usr/bin/env python
"""BNS tidal response check (gradient validation): for mlgw-bns-jax waveforms at Lambda1 = Lambda2 = L_sur,
find the TEOBResumSPA Lambda that matches best (overlap maximised over time and phase, GWTC-1 H1 PSD,
23-1500 Hz). Tests the sign and scale of the surrogate's tidal response over finite changes.

    teobresums env:  python lambda_match_scan.py generate --out teob_lambda_scan.npz
    gwjax env:       python lambda_match_scan.py compare teob_lambda_scan.npz
"""
import sys
import numpy as np

M1, M2, DIST, INCL = 1.40, 1.18, 40.0, 2.5
if sys.argv[1] == "generate":
    from teobresumspa_source import teobresumspa_polarizations as P
    f = np.arange(23 * 128, 1500 * 128 + 1) / 128.0
    L = np.arange(200.0, 525.0, 12.5)
    H = np.array([P(f, M1, M2, 0.0, 0.0, l, l, DIST, INCL, 0.0, 0.9 * 23)[0] for l in L])
    np.savez_compressed(sys.argv[sys.argv.index("--out") + 1], L=L, H=H, f=f)
    sys.exit()

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator
D = np.load(sys.argv[2]); f, L, H = D["f"], D["L"], D["H"]
psd = np.load("../gw170817_mlgw_bns/data_gwtc1/H1_psd.npy"); w = 1 / psd[np.searchsorted(np.arange(psd.size) / 128.0, f)]
pred = MLGWBNSGenerator().predict
sur = lambda l: np.asarray(pred(jnp.array([M1 / M2, l, l, 0., 0.]), jnp.asarray(f), total_mass=M1 + M2,
                                distance_mpc=DIST, inclination=INCL)[0])


def match(a, b):        # maximised over time (zero-padded inverse FFT of the cross-spectrum) and phase
    x = np.zeros(2**21, complex); x[:f.size] = a * np.conj(b) * w
    return np.max(np.abs(np.fft.ifft(x))) * x.size / np.sqrt(np.sum(np.abs(a)**2 * w) * np.sum(np.abs(b)**2 * w))


for ls in (250.0, 300.0, 400.0, 500.0):
    mm = np.array([1 - match(sur(ls), h) for h in H]); i = int(np.argmin(mm))
    j = min(max(i, 1), len(L) - 2); y = mm[j - 1:j + 2]
    best = L[j] + 0.5 * (L[1] - L[0]) * (y[0] - y[2]) / (y[0] - 2 * y[1] + y[2])
    print(f"surrogate Lambda {ls:5.0f}: best-matching TEOBResumSPA Lambda {best:6.1f}  (mismatch {mm[i]:.1e})")
