#!/usr/bin/env python
"""Run-time profile of the paper's likelihoods (referee comments 4 and 6).

For each analysis, times every stage of the likelihood the sampler calls, on whatever device JAX
uses (GPU, or CPU restricted with `taskset` to N cores), per evaluation inside a vmapped batch of
B evaluations, plus the JIT compile time of each function:

  gw170817     mlgw_bns_jax + relative binning, the production tc/phi_c-marginalised likelihood:
                 waveform at the RB bin frequencies | RB at fixed (tc, phi_c) | RB + tc marg. |
                 RB + tc + phi_c marg. (production) | full band, fixed (tc, phi_c) |
                 full band + tc + phi_c marg. (on every 4th tc point, 501 of 2001)
  gw150914_22  mlgw_bbh_jax (2,2), 4 s, full-band likelihood (the paper's run): waveform | likelihood
  gw150914_hm  mlgw_bbh_jax, 7 modes, 8 s, full-band likelihood: waveform | likelihood

    python profile_pe.py --what gw170817 gw150914_22 gw150914_hm --batches 1 10 100 1000 \
        --gw170817-data DIR --gw150914-data DIR --gw150914-hm-data FILE --out profile_gpu.json

Parameter batches are random draws around each event's posterior (the cost does not depend on the
values). Timings are the median of --repeats calls after compilation, divided by the batch size.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--what", nargs="+", default=["gw170817", "gw150914_22", "gw150914_hm"])
ap.add_argument("--batches", nargs="+", type=int, default=[1, 10, 100, 1000])
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--max-seconds", type=float, default=60.0, help="skip larger batches once one call exceeds this")
ap.add_argument("--gw170817-data", default=None, help="data_gwtc1/ directory (freeze_gw170817_data.py output)")
ap.add_argument("--gw150914-data", default=None, help="data_lvk_welch/ directory (freeze_gw150914_data.py output)")
ap.add_argument("--gw150914-hm-data", default=None, help="gw150914_data.npz of the 8 s bilby 7-mode run")
ap.add_argument("--skip-full-marg", action="store_true", help="skip the full-band tc/phi-marg. GW170817 likelihood")
ap.add_argument("--sampler-compile", action="store_true",
                help="GW170817: also time the production acceptance-walk sampler's compile + first iteration")
ap.add_argument("--sampler-nlive", type=int, default=1000)
ap.add_argument("--label", default="")
ap.add_argument("--out", required=True)
args, _ = ap.parse_known_args()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import gwgpu_jax

DEV = jax.devices()[0]
try:
    N_CORES = len(os.sched_getaffinity(0))
except AttributeError:
    N_CORES = os.cpu_count()
INFO = dict(label=args.label, device=str(DEV), platform=DEV.platform, device_kind=getattr(DEV, "device_kind", ""),
            cpu_cores_available=N_CORES, cpu_model=platform.processor() or platform.machine(),
            jax=jax.__version__, gwgpu_jax=getattr(gwgpu_jax, "__version__", "?"))
try:
    INFO["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
except (OSError, StopIteration):
    pass
print(INFO, flush=True)
RES = dict(info=INFO, results={})


def _sync(x):
    return jax.block_until_ready(x)


def bench(name, fn, make_batch, batches):
    """fn: single-particle function. Returns {batch: {compile_s, per_eval_s}}."""
    out = {}
    vf = jax.jit(jax.vmap(fn))
    for b in batches:
        P = make_batch(b)
        t0 = time.perf_counter(); _sync(vf(P)); t_first = time.perf_counter() - t0
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter(); _sync(vf(P)); ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        out[b] = dict(first_call_s=t_first, compile_s=max(t_first - t, 0.0), call_s=t, per_eval_s=t / b)
        print(f"  {name:34s} batch {b:5d}: {1e6 * t / b:12.2f} us/eval   (call {t:.4f} s, first call incl. compile {t_first:.1f} s)", flush=True)
        if t > args.max_seconds:
            print(f"  {name}: call above {args.max_seconds} s, larger batches skipped", flush=True)
            break
    return out


def jitter(rng, b, centre, scale, lo=None, hi=None):
    x = centre + scale * rng.standard_normal(b)
    if lo is not None or hi is not None:
        x = np.clip(x, lo, hi)
    return x


# ── GW170817 ───────────────────────────────────────────────────────────────────────────────────
def profile_gw170817():
    from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator
    from gwgpu_jax.gwgpu_jax_relative_binning import (
        build_rb_likelihood, build_rb_likelihood_time_marg, build_rb_likelihood_tc_phi_marg,
        build_full_likelihood_tc_phi_marg)
    DATA = Path(args.gw170817_data)
    meta = json.load(open(DATA / "metadata.json"))
    grid = gwgpu_jax.TimeFrequencyGrid(meta["duration"], meta["sampling_frequency"],
                                       f_min=meta["minimum_frequency"], f_max=meta["maximum_frequency"])
    network = gwgpu_jax.Network.from_names(meta["detectors"], grid)
    for ifo in network.interferometers:
        ifo.load_data(jnp.asarray(np.load(DATA / f"{ifo.name}_strain_fd.npy")), domain="fd")
        ifo.psd = jnp.asarray(np.load(DATA / f"{ifo.name}_psd.npy")); ifo._weight = None
    network.gmst = meta["gmst"]
    predict = MLGWBNSGenerator().predict

    def waveform_fn(p, freqs):                       # identical to the GW170817 notebook
        m1 = p["mc"] * (1.0 + p["q"]) ** 0.2 / p["q"] ** 0.6; m2 = p["q"] * m1
        hp, hc = predict(jnp.stack([m1 / m2, p["lambda_1"], p["lambda_2"], p["chi_1"], p["chi_2"]]), freqs,
                         total_mass=m1 + m2, distance_mpc=p["distance"], inclination=p["inclination"])
        ph = jnp.exp(-1j * p["phi_c"])
        return hp * ph, hc * ph

    TC0 = meta["tc_in_segment"]
    FID = {"mc": 1.1975, "q": 0.85, "chi_1": 0.0, "chi_2": 0.0, "lambda_1": 250.0, "lambda_2": 250.0,
           "distance": 37.0, "inclination": 2.545, "ra": 3.44616, "dec": -0.408084, "psi": 0.0, "tc": TC0, "phi_c": 0.0}
    TC_GRID = TC0 + 1e-5 * np.arange(-1000, 1001)    # +-10 ms, 10 us: the production grid (2001 points)
    N_BINS = 1600
    t0 = time.perf_counter()
    ll_tcphi, rb = build_rb_likelihood_tc_phi_marg(network=network, waveform_fn=waveform_fn, fiducial_params=FID,
                                                   tc_grid=TC_GRID, n_bins=N_BINS, gmst=network.gmst, verbose=False)
    t_setup = time.perf_counter() - t0
    ll_tc, _ = build_rb_likelihood_time_marg(network=network, waveform_fn=waveform_fn, fiducial_params=FID,
                                             tc_grid=TC_GRID, n_bins=N_BINS, gmst=network.gmst, verbose=False)
    ll_plain, _ = build_rb_likelihood(network=network, waveform_fn=waveform_fn, fiducial_params=FID,
                                      n_bins=N_BINS, gmst=network.gmst, verbose=False)
    PI = np.pi
    bounds = {"mc": (1.18, 1.21), "q": (0.34, 1.0), "chi_1": (-0.05, 0.05), "chi_2": (-0.05, 0.05),
              "lambda_1": (5.0, 5000.0), "lambda_2": (5.0, 5000.0), "distance": (1.0, 75.0), "inclination": (0.0, PI),
              "psi": (0.0, PI), "ra": (0.0, 2 * PI), "dec": (-PI / 2, PI / 2), "phi_c": (0.0, 2 * PI),
              "tc": (TC0 - 0.01, TC0 + 0.01)}
    full_plain = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler(network=network, waveform_fn=waveform_fn, param_bounds=bounds,
                                                          gmst=None)._loglik_fn
    f_bins = rb.f_bins
    f_band = jnp.asarray(np.asarray(grid.frequency_domain_array)[np.asarray(grid.frequency_mask)])
    rng = np.random.default_rng(0)

    def batch(b, keys):
        P = dict(mc=jitter(rng, b, 1.19755, 2e-5), q=rng.uniform(0.7, 1.0, b), chi_1=rng.uniform(-0.02, 0.02, b),
                 chi_2=rng.uniform(-0.02, 0.02, b), lambda_1=rng.uniform(50, 1000, b), lambda_2=rng.uniform(50, 1500, b),
                 distance=rng.uniform(25, 45, b), inclination=rng.uniform(2.2, 2.9, b), psi=rng.uniform(0, PI, b),
                 ra=jitter(rng, b, 3.44616, 0.01), dec=jitter(rng, b, -0.408, 0.01), phi_c=rng.uniform(0, 2 * PI, b),
                 tc=jitter(rng, b, TC0, 1e-3))
        return {k: jnp.asarray(P[k]) for k in keys}

    k11 = ["mc", "q", "chi_1", "chi_2", "lambda_1", "lambda_2", "distance", "inclination", "psi", "ra", "dec"]
    r = dict(n_rb_bins=int(len(f_bins)), n_band_bins=int(f_band.size), n_tc=len(TC_GRID), rb_setup_s=t_setup, stages={})
    print(f"GW170817: {len(f_bins)} RB bins, {f_band.size} band bins, {len(TC_GRID)} tc points; RB setup {t_setup:.1f} s", flush=True)
    S = r["stages"]
    S["waveform_rb_bins"] = bench("waveform @ RB bins", lambda p: waveform_fn(p, f_bins)[0], lambda b: batch(b, k11 + ["phi_c"]), args.batches)
    S["rb_fixed_tc_phi"] = bench("RB, fixed tc & phi_c", ll_plain, lambda b: batch(b, k11 + ["phi_c", "tc"]), args.batches)
    S["rb_tc_marg"] = bench("RB + tc marg.", ll_tc, lambda b: batch(b, k11 + ["phi_c"]), args.batches)
    S["rb_tc_phi_marg"] = bench("RB + tc + phi_c marg. (production)", ll_tcphi, lambda b: batch(b, k11), args.batches)
    S["waveform_full_band"] = bench("waveform, full band", lambda p: waveform_fn(p, f_band)[0], lambda b: batch(b, k11 + ["phi_c"]),
                                    [b for b in args.batches if b <= 100])
    S["full_fixed_tc_phi"] = bench("full band, fixed tc & phi_c", full_plain, lambda b: batch(b, k11 + ["phi_c", "tc"]),
                                   [b for b in args.batches if b <= 100])
    if not args.skip_full_marg:
        TC_CHECK = TC_GRID[::4]
        full_tcphi = build_full_likelihood_tc_phi_marg(network=network, waveform_fn=waveform_fn, fiducial_params=FID,
                                                       tc_grid=TC_CHECK, gmst=network.gmst)
        # one particle per call (the N_tc x N_band phase matrix is 1.5 GB); not vmapped
        P1 = batch(1, k11); p1 = {k: v[0] for k, v in P1.items()}
        f = jax.jit(full_tcphi)
        t0 = time.perf_counter(); _sync(f(p1)); tf = time.perf_counter() - t0
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter(); _sync(f(p1)); ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        S["full_tc_phi_marg_501tc"] = {1: dict(first_call_s=tf, compile_s=max(tf - t, 0), call_s=t, per_eval_s=t)}
        print(f"  {'full band + tc + phi_c marg. (501 tc)':34s} batch     1: {1e6 * t:12.2f} us/eval   (first call {tf:.1f} s)", flush=True)
    if args.sampler_compile:
        from gwgpu_jax.gwgpu_jax_acceptance_walk import WALK_RULE
        priors = {"mc": "uniform_in_components_chirp_mass", "q": "uniform_in_components_mass_ratio",
                  "chi_1": "aligned_spin", "chi_2": "aligned_spin", "distance": "volumetric", "inclination": "sin", "dec": "cos"}
        b11 = {k: bounds[k] for k in k11}
        smp = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler.from_loglikelihood(
            loglikelihood_fn=ll_tcphi, param_bounds=b11, priors=priors, periodic_params=("ra", "psi"))
        out_runs = []
        for n_it in (1, 3):     # 1 iteration = compile + first step; the difference to 3 gives one iteration
            t0 = time.perf_counter()
            smp.run(rng_key=jax.random.PRNGKey(0), num_live=args.sampler_nlive, n_target=60, max_mcmc=5000,
                    num_delete=max(args.sampler_nlive // 10, 1), max_iterations=n_it,
                    log_dlogz_target=float(np.log(np.expm1(0.1))), num_posterior_samples=100, verbose=False)
            out_runs.append(time.perf_counter() - t0)
        per_it = (out_runs[1] - out_runs[0]) / 2
        r["sampler"] = dict(walk_rule=WALK_RULE, num_live=args.sampler_nlive, wall_1_iteration_s=out_runs[0],
                            wall_3_iterations_s=out_runs[1], est_per_iteration_s=per_it,
                            est_compile_and_setup_s=out_runs[0] - per_it)
        print(f"  sampler (nlive {args.sampler_nlive}): 1 iteration {out_runs[0]:.1f} s, 3 iterations {out_runs[1]:.1f} s "
              f"-> compile + setup ~{out_runs[0] - per_it:.1f} s, ~{per_it:.1f} s per iteration "
              "(early iterations; later ones need longer walks)", flush=True)
    return r


# ── GW150914 ───────────────────────────────────────────────────────────────────────────────────
def _tukey(n, a_l, a_r):
    x = jnp.linspace(0.0, 1.0, n); w = jnp.ones_like(x)
    left = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * (x / a_l - 0.5)))
    right = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * ((x - 1.0) / a_r + 0.5)))
    w = jnp.where(x < a_l / 2.0, left, w)
    return jnp.where(x > 1.0 - a_r / 2.0, right, w)


def profile_gw150914(hm):
    from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator
    PI = np.pi
    if hm:
        from gwgpu_jax.gwgpu_jax_compatibility import gmst_from_gps
        D = np.load(args.gw150914_hm_data)
        dt, n = float(D["dt"]), len(D["times"])
        grid = gwgpu_jax.TimeFrequencyGrid(duration=n * dt, sampling_rate=1 / dt, f_min=float(D["f_low"]), f_max=float(D["f_high"]))
        network = gwgpu_jax.Network.from_names([str(x) for x in D["names"]], grid)
        for k, ifo in enumerate(network.interferometers):
            ifo.load_data(jnp.asarray(D["data_fd"][k]), domain="fd"); ifo.psd = jnp.asarray(D["psd"][k]); ifo._weight = None
        network.gmst = float(gmst_from_gps(float(D["gps"])))
        tref, modes = float(D["t_ref"]), None
    else:
        DATA = Path(args.gw150914_data)
        meta = json.load(open(DATA / "metadata.json"))
        grid = gwgpu_jax.TimeFrequencyGrid(duration=meta["duration"], sampling_rate=meta["sampling_frequency"],
                                           f_min=meta["minimum_frequency"], f_max=meta["maximum_frequency"])
        network = gwgpu_jax.Network.from_names(meta["detectors"], grid)
        for ifo in network.interferometers:
            ifo.load_data(jnp.asarray(np.load(DATA / f"{ifo.name}_strain_fd.npy")), domain="fd")
            ifo.psd = jnp.asarray(np.load(DATA / f"{ifo.name}_psd.npy")); ifo._weight = None
        network.gmst = meta["gmst"]
        tref, modes = 0.875 * grid.duration + 0.12, (2, 2)
    gen = MLGWBBHGenerator(model=4).generate
    N, dt = grid.n_samples, grid.dt
    T = jnp.arange(N) * dt - grid.duration + 0.12
    W = _tukey(N, 0.2, 0.001)

    def waveform_fn(p, freqs):                       # the notebooks' TD -> FD recipe
        m1 = p["mc"] * (1.0 + p["q"]) ** 0.2 / p["q"] ** 0.6; m2 = p["q"] * m1
        hp, hc = gen(jnp.stack([m1, m2, p["chi_1"], p["chi_2"], p["distance"], p["inclination"], p["phi_c"]]), T, modes=modes)
        return jnp.fft.rfft(hp * W) * dt, -jnp.fft.rfft(hc * W) * dt

    bounds = {"mc": (15.0, 50.0), "q": (0.1, 1.0), "chi_1": (-0.9, 0.9), "chi_2": (-0.9, 0.9), "distance": (50.0, 3000.0),
              "inclination": (0.0, PI), "ra": (0.0, 2 * PI), "dec": (-PI / 2, PI / 2), "psi": (0.0, PI),
              "phi_c": (0.0, 2 * PI), "tc": (tref - 0.1, tref + 0.1)}
    ll = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler(network=network, waveform_fn=waveform_fn, param_bounds=bounds, gmst=None)._loglik_fn
    rng = np.random.default_rng(0)

    def batch(b):
        P = dict(mc=jitter(rng, b, 30.9, 1.0), q=rng.uniform(0.7, 1.0, b), chi_1=rng.uniform(-0.5, 0.5, b),
                 chi_2=rng.uniform(-0.5, 0.5, b), distance=rng.uniform(350, 650, b), inclination=rng.uniform(2.2, 3.0, b),
                 ra=rng.uniform(1.2, 2.6, b), dec=jitter(rng, b, -1.22, 0.05), psi=rng.uniform(0, PI, b),
                 phi_c=rng.uniform(0, 2 * PI, b), tc=jitter(rng, b, tref, 1e-3))
        return {k: jnp.asarray(v) for k, v in P.items()}

    name = "GW150914 7 modes (8 s)" if hm else "GW150914 (2,2) (4 s)"
    print(f"{name}: {grid.n_samples} samples, {int(np.sum(np.asarray(grid.frequency_mask)))} band bins", flush=True)
    freqs = grid.frequency_domain_array
    return dict(n_samples=int(N), duration=float(grid.duration), modes="all 7" if hm else "(2,2)", stages=dict(
        waveform_td_fft=bench("waveform (TD + taper + rFFT)", lambda p: waveform_fn(p, freqs)[0], batch, args.batches),
        likelihood=bench("likelihood (full band)", ll, batch, args.batches)))


for w in args.what:
    t0 = time.perf_counter()
    if w == "gw170817":
        RES["results"][w] = profile_gw170817()
    elif w in ("gw150914_22", "gw150914_hm"):
        RES["results"][w] = profile_gw150914(hm=w == "gw150914_hm")
    else:
        raise SystemExit(f"unknown --what {w}")
    RES["results"][w]["profile_wall_s"] = time.perf_counter() - t0
    json.dump(RES, open(args.out, "w"), indent=2)
print("wrote", args.out)
