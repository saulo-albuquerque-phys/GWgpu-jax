#!/usr/bin/env python
"""Step 2 of the gradient validation (gwjax environment): JAX autodiff derivatives of mlgw-jax
(SEOBNRv5HM surrogate, (2,2) mode) vs central finite differences of SEOBNRv5HM itself
(referee 1 comment 5, referee 2 major comment 5).

Waveforms: h+ on mlgw-jax's 4 s grid (t = 0 at the peak), tapered and Fourier transformed exactly as
in the GW150914 analysis; noise-weighted inner product with the GW150914 H1 PSD over 20-1024 Hz.

The two models fix the origin of time and phase, and the overall amplitude, by different
conventions. These conventions add to dh/dtheta components along the coalescence time, phase and
distance directions, which are sampled (or degenerate with sampled parameters) in PE. We therefore
compare the derivatives after projecting out span{h, i h, f h} of each model (P_perp):

  eps_i     = || P_perp dh_sur/dtheta_i - P_perp dh_tgt/dtheta_i || / || P_perp dh_tgt/dtheta_i ||
  sigma     = Fisher-matrix marginal uncertainties at network SNR 24 over (m1, m2, chi1, chi2,
              ln d, t_c, phi_c), transformed to (Mc, q, chi_eff); ratio surrogate/target
  checks    = autodiff vs FD of the surrogate itself; FD convergence of the target (3 step sizes)
  cost      = time of a gradient vs a value, for the waveform and for the full GW150914 likelihood

    python compare_gradients_bbh.py --stencil seob_stencil.npz --out gradients_bbh.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--stencil", default=str(HERE / "seob_stencil.npz"))
ap.add_argument("--psd", default=str(HERE.parent / "gw150914_mlgw_bbh/data_lvk_welch/H1_psd.npy"))
ap.add_argument("--data", default=str(HERE.parent / "gw150914_mlgw_bbh/data_lvk_welch"))
ap.add_argument("--run", default=str(HERE.parent / "gw150914_mlgw_bbh/20261117_135025_lvk_welch"))
ap.add_argument("--snr", type=float, default=24.0)
ap.add_argument("--out", default=str(HERE / "gradients_bbh.json"))
a = ap.parse_args()

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import gwgpu_jax
from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator

S = np.load(a.stencil)
PTS, NAMES = S["points"], [str(n) for n in S["names"]]
t_grid = S["t_grid"]
N = t_grid.size; dt = t_grid[1] - t_grid[0]
f = np.fft.rfftfreq(N, dt); df = f[1] - f[0]
band = (f >= 20.0) & (f <= 1024.0)
fb = f[band]
w = 4.0 * df / np.load(a.psd)[band]                     # inner-product weights


def tukey(n, al, ar):
    x = np.linspace(0, 1, n); win = np.ones(n)
    l = x < al / 2; r = x > 1 - ar / 2
    win[l] = 0.5 * (1 + np.cos(2 * np.pi * (x[l] / al - 0.5)))
    win[r] = 0.5 * (1 + np.cos(2 * np.pi * ((x[r] - 1) / ar + 0.5)))
    return win


W = tukey(N, 0.2, 0.001)
to_fd = lambda h_td: (np.fft.rfft(np.asarray(h_td) * W) * dt)[band]
ip = lambda x, y: float(np.real(np.sum(x * np.conj(y) * w)))
nrm = lambda x: np.sqrt(ip(x, x))

gen = MLGWBBHGenerator(model=4).generate
Wj, Tj, bj = jnp.asarray(W), jnp.asarray(t_grid), jnp.asarray(np.where(band)[0])
DIST, INCL = float(S["dist"]), float(S["incl"])


def sur_fd(th4):
    th = jnp.concatenate([th4, jnp.array([DIST, INCL, 0.0])])
    hp, _ = gen(th, Tj, modes=(2, 2))
    return (jnp.fft.rfft(hp * Wj) * dt)[bj]


sur_fd_j = jax.jit(sur_fd)
sur_jac = jax.jit(jax.jacfwd(sur_fd))


def perp_basis(h):
    """Orthonormal basis (Gram-Schmidt, weighted inner product) of span{h, i h, f h}."""
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


def align(h_s, h_t, dt_max=2e-3, n=4001):
    """Time shift and phase that maximise the overlap of h_s with h_t (models define t = 0 and the
    reference phase differently); returns the factor that multiplies h_s and its derivatives."""
    shifts = np.linspace(-dt_max, dt_max, n)
    z = np.array([np.sum(h_s * np.exp(-2j * np.pi * fb * d) * np.conj(h_t) * w) for d in shifts])
    d = shifts[np.argmax(np.abs(z))]
    zz = np.sum(h_s * np.exp(-2j * np.pi * fb * d) * np.conj(h_t) * w)
    return np.exp(-2j * np.pi * fb * d) * np.exp(-1j * np.angle(zz)), d


def fisher_sigmas(h, dh4):
    """Marginal sigmas over (m1, m2, chi1, chi2, ln d, t_c, phi_c) at network SNR a.snr, then
    transformed to (Mc, q, chi_eff) with the Jacobian at the reference point."""
    s = a.snr / nrm(h)
    D = [s * d for d in dh4] + [-s * h, -2j * np.pi * fb * s * h, 1j * s * h]
    G = np.array([[ip(x, y) for y in D] for x in D])
    C = np.linalg.inv(G)[:4, :4]
    return C


def derived_sigmas(C, th):
    m1, m2, c1, c2 = th
    M = m1 + m2; mc = (m1 * m2) ** 0.6 / M ** 0.2
    J = np.array([[mc * (0.6 / m1 - 0.2 / M), mc * (0.6 / m2 - 0.2 / M), 0, 0],          # dMc
                  [-m2 / m1**2, 1 / m1, 0, 0],                                          # q = m2/m1
                  [(c1 - c2) * m2 / M**2, (c2 - c1) * m1 / M**2, m1 / M, m2 / M]])      # chi_eff
    Cd = J @ C @ J.T
    return dict(zip(["mc", "q", "chi_eff"], np.sqrt(np.diag(Cd))))


res = {"points": []}
for k, th in enumerate(PTS):
    h_t = to_fd(S[f"p{k}_hp"])
    h_s = np.asarray(sur_fd_j(jnp.asarray(th)))
    J_s = np.asarray(sur_jac(jnp.asarray(th)))           # (n_f, 4)
    rot, t_off = align(h_s, h_t)                          # common time/phase alignment of the surrogate
    h_s, J_s = h_s * rot, J_s * rot[:, None]
    Bt, Bs = perp_basis(h_t), perp_basis(h_s)
    mm = 1 - ip(h_s, h_t) / (nrm(h_s) * nrm(h_t))        # after time and phase alignment
    row = dict(theta=dict(zip(NAMES, th.tolist())), mismatch_sur_vs_tgt=float(mm), time_offset_s=float(t_off),
               params={})
    dh_t_mid, dh_s = [], []
    for j, nm in enumerate(NAMES):
        D_t = []
        for s_ in range(3):
            hstep = float(S[f"p{k}_{nm}_s{s_}_h"])
            D_t.append((to_fd(S[f"p{k}_{nm}_s{s_}_plus_hp"]) - to_fd(S[f"p{k}_{nm}_s{s_}_minus_hp"])) / (2 * hstep))
        Pt = [project(d, Bt) for d in D_t]
        ps = project(J_s[:, j], Bs)
        # FD of the surrogate at the middle step (implementation check of autodiff)
        hstep = float(S[f"p{k}_{nm}_s1_h"]); e = np.zeros(4); e[j] = hstep
        D_sfd = rot * (np.asarray(sur_fd_j(jnp.asarray(th + e))) - np.asarray(sur_fd_j(jnp.asarray(th - e)))) / (2 * hstep)
        row["params"][nm] = dict(
            eps_autodiff_vs_target_fd=float(nrm(ps - Pt[1]) / nrm(Pt[1])),
            cos_autodiff_vs_target_fd=float(ip(ps, Pt[1]) / (nrm(ps) * nrm(Pt[1]))),
            target_fd_convergence=float(max(nrm(Pt[0] - Pt[1]), nrm(Pt[2] - Pt[1])) / nrm(Pt[1])),
            autodiff_vs_surrogate_fd=float(nrm(J_s[:, j] - D_sfd) / nrm(J_s[:, j])))
        dh_t_mid.append(D_t[1]); dh_s.append(J_s[:, j])
    Ct, Cs = fisher_sigmas(h_t, dh_t_mid), fisher_sigmas(h_s, dh_s)
    st = dict(zip(NAMES, np.sqrt(np.diag(Ct)))); ss = dict(zip(NAMES, np.sqrt(np.diag(Cs))))
    st.update(derived_sigmas(Ct, th)); ss.update(derived_sigmas(Cs, th))
    row["fisher_sigma_target"] = {k_: float(v) for k_, v in st.items()}
    row["fisher_sigma_surrogate"] = {k_: float(v) for k_, v in ss.items()}
    row["fisher_sigma_ratio"] = {k_: float(ss[k_] / st[k_]) for k_ in st}
    res["points"].append(row)
    p = row["params"]
    print(f"point {k} {th}: mismatch {mm:.1e} (time offset {1e3 * t_off:+.3f} ms)")
    for nm in NAMES:
        print(f"   {nm:5s} eps(AD sur vs FD target) {p[nm]['eps_autodiff_vs_target_fd']:.3f}  cos {p[nm]['cos_autodiff_vs_target_fd']:.4f}   "
              f"FD convergence {p[nm]['target_fd_convergence']:.1e}   AD vs FD(sur) {p[nm]['autodiff_vs_surrogate_fd']:.1e}")
    print("   Fisher sigma ratio sur/SEOB: " + ", ".join(f"{k_} {v:.3f}" for k_, v in row["fisher_sigma_ratio"].items()))

# ---- cost of gradients -----------------------------------------------------------------------
def best(fn, *x, n=20):
    jax.block_until_ready(fn(*x)); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); jax.block_until_ready(fn(*x)); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


th0 = jnp.asarray(PTS[0])
cost = dict(waveform_value_s=best(sur_fd_j, th0), waveform_jacfwd_4params_s=best(sur_jac, th0))

meta = json.load(open(Path(a.data) / "metadata.json"))
grid = gwgpu_jax.TimeFrequencyGrid(meta["duration"], meta["sampling_frequency"],
                                   f_min=meta["minimum_frequency"], f_max=meta["maximum_frequency"])
net = gwgpu_jax.Network.from_names(meta["detectors"], grid)
for ifo in net.interferometers:
    ifo.load_data(jnp.asarray(np.load(Path(a.data) / f"{ifo.name}_strain_fd.npy")), domain="fd")
    ifo.psd = jnp.asarray(np.load(Path(a.data) / f"{ifo.name}_psd.npy")); ifo._weight = None
net.gmst = meta["gmst"]
cfg = json.load(open(Path(a.run) / "result.json"))["config"]


def wf(p, freqs):
    m1 = p["mc"] * (1.0 + p["q"]) ** 0.2 / p["q"] ** 0.6; m2 = p["q"] * m1
    hp, hc = gen(jnp.stack([m1, m2, p["chi_1"], p["chi_2"], p["distance"], p["inclination"], p["phi_c"]]), Tj, modes=(2, 2))
    return jnp.fft.rfft(hp * Wj) * dt, -jnp.fft.rfft(hc * Wj) * dt


ll = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler(network=net, waveform_fn=wf, param_bounds=cfg["param_bounds"],
                                             gmst=None)._loglik_fn
post = np.load(Path(a.run) / "posterior_samples.npz")
keys = ["mc", "q", "chi_1", "chi_2", "distance", "inclination", "ra", "dec", "psi", "phi_c", "tc"]
P1 = {k_: jnp.asarray(float(np.median(post[k_]))) for k_ in keys}
PB = {k_: jnp.asarray(np.asarray(post[k_])[:100]) for k_ in keys}
v1, g1 = jax.jit(ll), jax.jit(jax.grad(ll))
vb, gb = jax.jit(jax.vmap(ll)), jax.jit(jax.vmap(jax.grad(ll)))
g = g1(P1)
assert all(np.isfinite(float(g[k_])) for k_ in keys)
cost.update(lnL_value_s=best(v1, P1), lnL_grad_11params_s=best(v1 if False else g1, P1),
            lnL_value_batch100_s=best(vb, PB, n=5), lnL_grad_batch100_s=best(gb, PB, n=5),
            device=str(jax.devices()[0]), jax=jax.__version__)
cost["grad_over_value_lnL"] = cost["lnL_grad_11params_s"] / cost["lnL_value_s"]
cost["grad_over_value_lnL_batch100"] = cost["lnL_grad_batch100_s"] / cost["lnL_value_batch100_s"]
cost["jacfwd_over_value_waveform"] = cost["waveform_jacfwd_4params_s"] / cost["waveform_value_s"]
res["cost"] = cost
print("cost:", {k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in cost.items()})
json.dump(res, open(a.out, "w"), indent=2)
print("wrote", a.out)
