#!/usr/bin/env python
"""Step 2 (BNS, gwjax environment): JAX autodiff derivatives of mlgw-bns-jax vs central finite
differences of TEOBResumSPA (referee 1 comment 5, referee 2 major comment 5).

Same method as compare_gradients_bbh.py: h+ on the GW170817 analysis frequencies (23-1500 Hz,
df = 1/128 Hz), noise-weighted with the GWTC-1 H1 PSD; the surrogate is aligned to the target in time
and phase once per reference point (the models define t = 0 and the phase differently), and
derivatives are compared after projecting out span{h, i h, f h} (amplitude, phase, time directions):

  eps_i  = || P_perp dh_sur/dtheta_i - P_perp dh_tgt/dtheta_i || / || P_perp dh_tgt/dtheta_i ||
  cos_i  = cosine of the angle between the two projected derivatives
  checks = autodiff vs FD of the surrogate; FD convergence of the target (3 step sizes)
  cost   = gradient vs value of the production GW170817 likelihood (relative binning, tc/phi_c marg.)

    python compare_gradients_bns.py --stencil teob_stencil.npz --out gradients_bns.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--stencil", default=str(HERE / "teob_stencil.npz"))
ap.add_argument("--data", default=str(HERE.parent / "gw170817_mlgw_bns/data_gwtc1"))
ap.add_argument("--run", default=str(HERE.parent / "gwjax_runs/20261114_162230_gwtc1_lowspin"))
ap.add_argument("--out", default=str(HERE / "gradients_bns.json"))
a = ap.parse_args()

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import gwgpu_jax
from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator
from gwgpu_jax.gwgpu_jax_relative_binning import build_rb_likelihood_tc_phi_marg

S = np.load(a.stencil)
PTS, NAMES = S["points"], [str(n) for n in S["names"]]
fb = S["freqs"]
meta = json.load(open(Path(a.data) / "metadata.json"))
f_full = np.arange(int(meta["duration"] * meta["sampling_frequency"]) // 2 + 1) / meta["duration"]
idx = np.searchsorted(f_full, fb)
assert np.allclose(f_full[idx], fb)
w = 4.0 * (fb[1] - fb[0]) / np.load(Path(a.data) / "H1_psd.npy")[idx]
ip = lambda x, y: float(np.real(np.sum(x * np.conj(y) * w)))
nrm = lambda x: np.sqrt(ip(x, x))

predict = MLGWBNSGenerator().predict
DIST, INCL = float(S["dist"]), float(S["incl"])
fj = jnp.asarray(fb)


def sur(th6):
    m1, m2, c1, c2, l1, l2 = th6[0], th6[1], th6[2], th6[3], th6[4], th6[5]
    hp, _ = predict(jnp.stack([m1 / m2, l1, l2, c1, c2]), fj, total_mass=m1 + m2, distance_mpc=DIST, inclination=INCL)
    return hp


sur_j, sur_jac = jax.jit(sur), jax.jit(jax.jacfwd(sur))


def align(h_s, h_t):
    """Time shift and phase maximising the overlap of h_s with h_t (coarse then fine search)."""
    best = 0.0
    for span, n in ((3e-3, 3001), (4e-6, 801)):
        shifts = best + np.linspace(-span, span, n)
        z = np.array([np.sum(h_s * np.exp(-2j * np.pi * fb * d) * np.conj(h_t) * w) for d in shifts])
        best = shifts[np.argmax(np.abs(z))]
    zz = np.sum(h_s * np.exp(-2j * np.pi * fb * best) * np.conj(h_t) * w)
    return np.exp(-2j * np.pi * fb * best) * np.exp(-1j * np.angle(zz)), best


def perp_basis(h):
    B = []
    for v in (h, 1j * h, fb * h):
        for b in B:
            v = v - ip(v, b) * b
        B.append(v / nrm(v))
    return B


def project(v, B):
    for b in B:
        v = v - ip(v, b) * b
    return v


res = {"points": []}
for k, th in enumerate(PTS):
    h_t = S[f"p{k}_hp"]
    h_s = np.asarray(sur_j(jnp.asarray(th)))
    J_s = np.asarray(sur_jac(jnp.asarray(th)))
    rot, t_off = align(h_s, h_t)
    h_s, J_s = h_s * rot, J_s * rot[:, None]
    Bt, Bs = perp_basis(h_t), perp_basis(h_s)
    mm = 1 - ip(h_s, h_t) / (nrm(h_s) * nrm(h_t))
    row = dict(theta=dict(zip(NAMES, th.tolist())), mismatch_sur_vs_tgt=float(mm), time_offset_s=float(t_off), params={})
    print(f"point {k} {th}: mismatch {mm:.1e} (time offset {1e3 * t_off:+.3f} ms)")
    for j, nm in enumerate(NAMES):
        D_t = []
        for s_ in range(3):
            hs = float(S[f"p{k}_{nm}_s{s_}_h"])
            D_t.append((S[f"p{k}_{nm}_s{s_}_plus_hp"] - S[f"p{k}_{nm}_s{s_}_minus_hp"]) / (2 * hs))
        Pt = [project(d, Bt) for d in D_t]
        ps = project(J_s[:, j], Bs)
        hs = float(S[f"p{k}_{nm}_s1_h"]); e = np.zeros(6); e[j] = hs
        D_sfd = rot * (np.asarray(sur_j(jnp.asarray(th + e))) - np.asarray(sur_j(jnp.asarray(th - e)))) / (2 * hs)
        r = dict(eps_autodiff_vs_target_fd=float(nrm(ps - Pt[1]) / nrm(Pt[1])),
                 cos_autodiff_vs_target_fd=float(ip(ps, Pt[1]) / (nrm(ps) * nrm(Pt[1]))),
                 target_fd_convergence=float(max(nrm(Pt[0] - Pt[1]), nrm(Pt[2] - Pt[1])) / nrm(Pt[1])),
                 autodiff_vs_surrogate_fd=float(nrm(J_s[:, j] - D_sfd) / nrm(J_s[:, j])))
        row["params"][nm] = r
        print(f"   {nm:8s} eps {r['eps_autodiff_vs_target_fd']:.3f}  cos {r['cos_autodiff_vs_target_fd']:.4f}   "
              f"FD convergence {r['target_fd_convergence']:.1e}   AD vs FD(sur) {r['autodiff_vs_surrogate_fd']:.1e}")
    res["points"].append(row)

# ---- cost: production GW170817 likelihood, value vs gradient ------------------------------------
grid = gwgpu_jax.TimeFrequencyGrid(meta["duration"], meta["sampling_frequency"],
                                   f_min=meta["minimum_frequency"], f_max=meta["maximum_frequency"])
net = gwgpu_jax.Network.from_names(meta["detectors"], grid)
for ifo in net.interferometers:
    ifo.load_data(jnp.asarray(np.load(Path(a.data) / f"{ifo.name}_strain_fd.npy")), domain="fd")
    ifo.psd = jnp.asarray(np.load(Path(a.data) / f"{ifo.name}_psd.npy")); ifo._weight = None
net.gmst = meta["gmst"]


def wf(p, freqs):
    m1 = p["mc"] * (1.0 + p["q"]) ** 0.2 / p["q"] ** 0.6; m2 = p["q"] * m1
    hp, hc = predict(jnp.stack([m1 / m2, p["lambda_1"], p["lambda_2"], p["chi_1"], p["chi_2"]]), freqs,
                     total_mass=m1 + m2, distance_mpc=p["distance"], inclination=p["inclination"])
    ph = jnp.exp(-1j * p["phi_c"])
    return hp * ph, hc * ph


TC0 = meta["tc_in_segment"]
FID = {"mc": 1.1975, "q": 0.85, "chi_1": 0.0, "chi_2": 0.0, "lambda_1": 250.0, "lambda_2": 250.0,
       "distance": 37.0, "inclination": 2.545, "ra": 3.44616, "dec": -0.408084, "psi": 0.0, "tc": TC0, "phi_c": 0.0}
ll, _ = build_rb_likelihood_tc_phi_marg(network=net, waveform_fn=wf, fiducial_params=FID,
                                        tc_grid=TC0 + 1e-5 * np.arange(-1000, 1001), n_bins=1600,
                                        gmst=net.gmst, verbose=False)
post = np.load(Path(a.run) / "posterior_samples.npz")
keys = ["mc", "q", "chi_1", "chi_2", "lambda_1", "lambda_2", "distance", "inclination", "psi", "ra", "dec"]
P1 = {k_: jnp.asarray(float(np.median(post[k_]))) for k_ in keys}
PB = {k_: jnp.asarray(np.asarray(post[k_])[:100]) for k_ in keys}


def best(fn, x, n=10):
    jax.block_until_ready(fn(x)); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); jax.block_until_ready(fn(x)); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


g = jax.jit(jax.grad(ll))(P1)
assert all(np.isfinite(float(g[k_])) for k_ in keys)
cost = dict(lnL_value_s=best(jax.jit(ll), P1), lnL_grad_11params_s=best(jax.jit(jax.grad(ll)), P1),
            lnL_value_batch100_s=best(jax.jit(jax.vmap(ll)), PB, 5),
            lnL_grad_batch100_s=best(jax.jit(jax.vmap(jax.grad(ll))), PB, 5),
            device=str(jax.devices()[0]), jax=jax.__version__)
cost["grad_over_value_lnL"] = cost["lnL_grad_11params_s"] / cost["lnL_value_s"]
cost["grad_over_value_lnL_batch100"] = cost["lnL_grad_batch100_s"] / cost["lnL_value_batch100_s"]
res["cost"] = cost
print("cost:", {k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in cost.items()})
json.dump(res, open(a.out, "w"), indent=2)
print("wrote", a.out)
