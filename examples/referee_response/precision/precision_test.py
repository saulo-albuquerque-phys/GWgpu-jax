#!/usr/bin/env python
"""Single vs double precision (referee 2, major comment 4).

Evaluates, at posterior samples of the paper's runs, the waveforms and the likelihoods used in PE,
once with JAX in double precision and once in single precision (jax_enable_x64 off: every array,
including the data, PSD and the relative-binning summary data, becomes float32/complex64):

  gw150914  mlgw-jax (2,2), 4 s, full band: FD waveform on the 20-1024 Hz band; full-band lnL
  gw170817  mlgw-bns-jax, 128 s: FD waveform on the 23-1500 Hz band; production RB lnL (tc, phi_c marg.)

    python precision_test.py run --x64 1 --out fp64.npz [--n 200]
    python precision_test.py run --x64 0 --out fp32.npz [--n 200]
    python precision_test.py run --x64 0 --rescale 1e21 --out fp32_rescaled.npz
    python precision_test.py compare fp64.npz fp32.npz [--out precision_summary.json]

compare reports the waveform mismatch fp32 vs fp64 (noise-weighted with the event's H1 PSD, no
maximisation) and the lnL difference: its spread across the posterior (what changes the posterior)
and the reweighting efficiency ESS/N of fp32 samples to the fp64 likelihood.
Timings for the speed-up need a GPU (profile_pe.py with JAX_ENABLE_X64=0 on Colab).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run")
r.add_argument("--x64", type=int, required=True)
r.add_argument("--out", required=True)
r.add_argument("--n", type=int, default=200)
r.add_argument("--n-wf", type=int, default=50, help="samples for the waveform mismatch")
r.add_argument("--rescale", type=float, default=1.0,
               help="multiply strain and templates by this factor and the PSD by its square (lnL is invariant); "
                    "1e21 keeps all quantities inside the float32 range")
r.add_argument("--gw150914-run", default=str(REF / "gwjax_runs/20261114_183906_lvk_welch"))
r.add_argument("--gw150914-data", default=str(REF / "gw150914_mlgw_bbh/data_lvk_welch"))
r.add_argument("--gw170817-run", default=str(REF / "gwjax_runs/20261114_162230_gwtc1_lowspin"))
r.add_argument("--gw170817-data", default=str(REF / "gw170817_mlgw_bns/data_gwtc1"))
c = sub.add_parser("compare")
c.add_argument("fp64"); c.add_argument("fp32")
c.add_argument("--out", default=str(HERE / "precision_summary.json"))
a = ap.parse_args()

import numpy as np

if a.cmd == "compare":
    A, B = np.load(a.fp64), np.load(a.fp32)
    out = {}
    for ev in ("gw150914", "gw170817"):
        w = A[f"{ev}_psd_w"]                                   # 1/PSD on the band
        h64, h32 = A[f"{ev}_hp"], B[f"{ev}_hp"].astype(np.complex128)
        ip = lambda x, y: np.real(np.sum(np.conj(x) * y * w, axis=-1))
        mm = 1.0 - ip(h64, h32) / np.sqrt(ip(h64, h64) * ip(h32, h32))
        n_bad_wf = int(np.sum(~np.isfinite(mm))); mm = mm[np.isfinite(mm)]
        d = B[f"{ev}_lnL"].astype(np.float64) - A[f"{ev}_lnL"]
        n_bad_lnL = int(np.sum(~np.isfinite(d))); d = d[np.isfinite(d)]
        dc = d - np.median(d)
        lw = -d; wts = np.exp(lw - lw.max()); wts /= wts.sum()
        out[ev] = dict(mismatch_median=float(np.median(mm)), mismatch_max=float(np.max(mm)),
                       dlnL_median=float(np.median(d)), dlnL_std=float(np.std(d)),
                       dlnL_max_abs_dev=float(np.max(np.abs(dc))), ess_fraction=float(1 / np.sum(wts**2) / len(wts)),
                       lnL_std_fp64=float(np.std(A[f"{ev}_lnL"])), n_nan_waveforms=n_bad_wf, n_nan_lnL=n_bad_lnL,
                       n_waveforms=int(len(h64)), n_lnL=int(len(A[f"{ev}_lnL"])))
        o = out[ev]
        print(f"{ev}: mismatch fp32 vs fp64 median {o['mismatch_median']:.2e}, max {o['mismatch_max']:.2e};  "
              f"dlnL median {o['dlnL_median']:+.3f}, std {o['dlnL_std']:.4f}, max|dlnL-med| {o['dlnL_max_abs_dev']:.3f}; "
              f"ESS/N {o['ess_fraction']:.3f};  NaN: {n_bad_wf}/{len(h64)} waveforms, {n_bad_lnL}/{o['n_lnL']} lnL")
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)
    sys.exit()

import jax
jax.config.update("jax_enable_x64", bool(a.x64))
if not a.x64:
    # several mlgw_bns_jax modules call jax.config.update("jax_enable_x64", True) at import time;
    # block that here so that the single-precision run really is single precision.
    _update = jax.config.update
    jax.config.update = lambda name, val: None if name == "jax_enable_x64" else _update(name, val)
import jax.numpy as jnp
import gwgpu_jax
from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator
from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator
from gwgpu_jax.gwgpu_jax_relative_binning import build_rb_likelihood_tc_phi_marg

print("x64:", jax.config.read("jax_enable_x64"), "jax", jax.__version__, flush=True)
rng = np.random.default_rng(3)
SC = a.rescale
RES = {}


def load_network(meta, data_dir):
    grid = gwgpu_jax.TimeFrequencyGrid(meta["duration"], meta["sampling_frequency"],
                                       f_min=meta["minimum_frequency"], f_max=meta["maximum_frequency"])
    net = gwgpu_jax.Network.from_names(meta["detectors"], grid)
    for ifo in net.interferometers:
        ifo.load_data(jnp.asarray(np.load(Path(data_dir) / f"{ifo.name}_strain_fd.npy") * SC), domain="fd")
        ifo.psd = jnp.asarray(np.load(Path(data_dir) / f"{ifo.name}_psd.npy") * SC**2); ifo._weight = None
    net.gmst = meta["gmst"]
    return grid, net


def band_weight(grid, data_dir):
    f = np.asarray(grid.frequency_domain_array); m = np.asarray(grid.frequency_mask)
    return m, 1.0 / np.load(Path(data_dir) / "H1_psd.npy")[m]


# ---- GW150914 (2,2) -------------------------------------------------------------------------
meta = json.load(open(Path(a.gw150914_data) / "metadata.json"))
grid, net = load_network(meta, a.gw150914_data)
gen = MLGWBBHGenerator(model=4).generate
N, dt = grid.n_samples, grid.dt
FT = json.load(open(Path(a.gw150914_run) / "result.json"))["config"]["mlgw_final_time"]
T = jnp.arange(N) * dt - grid.duration + FT
x = jnp.linspace(0.0, 1.0, N)
W = jnp.where(x < 0.1, 0.5 * (1 + jnp.cos(2 * jnp.pi * (x / 0.2 - 0.5))), 1.0)
W = jnp.where(x > 1 - 0.0005, 0.5 * (1 + jnp.cos(2 * jnp.pi * ((x - 1) / 0.001 + 0.5))), W)


def wf150914(p, freqs):
    m1 = p["mc"] * (1.0 + p["q"]) ** 0.2 / p["q"] ** 0.6; m2 = p["q"] * m1
    hp, hc = gen(jnp.stack([m1, m2, p["chi_1"], p["chi_2"], p["distance"], p["inclination"], p["phi_c"]]), T, modes=(2, 2))
    return jnp.fft.rfft(hp * W) * (dt * SC), -jnp.fft.rfft(hc * W) * (dt * SC)


cfg = json.load(open(Path(a.gw150914_run) / "result.json"))["config"]
ll = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler(network=net, waveform_fn=wf150914, param_bounds=cfg["param_bounds"],
                                             gmst=None)._loglik_fn
post = np.load(Path(a.gw150914_run) / "posterior_samples.npz")
keys = ["mc", "q", "chi_1", "chi_2", "distance", "inclination", "ra", "dec", "psi", "phi_c", "tc"]
idx = rng.choice(len(post["mc"]), a.n, replace=False)
P = {k: jnp.asarray(np.asarray(post[k])[idx]) for k in keys}
RES["gw150914_lnL"] = np.asarray(jax.jit(jax.vmap(ll))(P), dtype=np.float64)
m, w = band_weight(grid, a.gw150914_data)
freqs = grid.frequency_domain_array
hp = jax.jit(jax.vmap(lambda p: wf150914(p, freqs)[0]))({k: v[: a.n_wf] for k, v in P.items()})
RES["gw150914_hp"] = np.asarray(hp)[:, m].astype(np.complex128)
RES["gw150914_psd_w"] = w
print("gw150914 done", RES["gw150914_lnL"][:3], flush=True)

# ---- GW170817 ----------------------------------------------------------------------------------
meta = json.load(open(Path(a.gw170817_data) / "metadata.json"))
grid, net = load_network(meta, a.gw170817_data)
predict = MLGWBNSGenerator().predict


def wf170817(p, freqs):
    m1 = p["mc"] * (1.0 + p["q"]) ** 0.2 / p["q"] ** 0.6; m2 = p["q"] * m1
    hp, hc = predict(jnp.stack([m1 / m2, p["lambda_1"], p["lambda_2"], p["chi_1"], p["chi_2"]]), freqs,
                     total_mass=m1 + m2, distance_mpc=p["distance"], inclination=p["inclination"])
    ph = jnp.exp(-1j * p["phi_c"]) * SC
    return hp * ph, hc * ph


TC0 = meta["tc_in_segment"]
FID = {"mc": 1.1975, "q": 0.85, "chi_1": 0.0, "chi_2": 0.0, "lambda_1": 250.0, "lambda_2": 250.0,
       "distance": 37.0, "inclination": 2.545, "ra": 3.44616, "dec": -0.408084, "psi": 0.0, "tc": TC0, "phi_c": 0.0}
ll_rb, rb = build_rb_likelihood_tc_phi_marg(network=net, waveform_fn=wf170817, fiducial_params=FID,
                                            tc_grid=TC0 + 1e-5 * np.arange(-1000, 1001), n_bins=1600,
                                            gmst=net.gmst, verbose=False)
post = np.load(Path(a.gw170817_run) / "posterior_samples.npz")
keys = ["mc", "q", "chi_1", "chi_2", "lambda_1", "lambda_2", "distance", "inclination", "psi", "ra", "dec"]
idx = rng.choice(len(post["mc"]), a.n, replace=False)
P = {k: jnp.asarray(np.asarray(post[k])[idx]) for k in keys}
RES["gw170817_lnL"] = np.asarray(jax.jit(jax.vmap(ll_rb))(P), dtype=np.float64)
m, w = band_weight(grid, a.gw170817_data)
fb = jnp.asarray(np.asarray(grid.frequency_domain_array)[m])
Pw = {k: v[: a.n_wf] for k, v in P.items()}; Pw["phi_c"] = jnp.zeros(a.n_wf)
hp = jnp.stack([wf170817({k: v[i] for k, v in Pw.items()}, fb)[0] for i in range(a.n_wf)])
RES["gw170817_hp"] = np.asarray(hp).astype(np.complex128)
RES["gw170817_psd_w"] = w
print("gw170817 done", RES["gw170817_lnL"][:3], flush=True)

np.savez_compressed(a.out, **RES)
print("wrote", a.out)
