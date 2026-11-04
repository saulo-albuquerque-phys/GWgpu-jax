#!/usr/bin/env python
"""P-P injection campaign for the mlgw_bbh_jax (SEOBNRv5HM) GWgpu-jax pipeline.

Referee comment 1(a): tests the statistical calibration of the *full* pipeline
used for GW150914 in the paper — mlgw_bbh_jax TD->FD waveform, gwgpu_jax
likelihood/network projection, acceptance-walk nested sampler — on injections
drawn from the sampling prior.

Design (each choice is what makes the P-P test strictly valid):
  * injections are drawn FROM THE SAME PRIOR the sampler uses, no SNR cut
    (an SNR selection on theta while sampling the unconditioned prior biases
    the P-P curves, mostly in distance/inclination);
  * injection and recovery use the same model (mlgw_bbh_jax, (2,2) by default,
    identical to the paper's GW150914 run), so the test isolates the
    pipeline/sampler, not waveform systematics;
  * Gaussian noise coloured by the design PSD, a fresh seed per injection;
  * the signal is injected with the sampler's own tc + dt_ifo convention
    (network.project_waveform alone would leave tc unrecoverable);
  * sampler settings follow the bilby-validated acceptance-walk injection study
    (Prathaban et al. 2025): nlive 1400, num_delete 700, n_target 60,
    maxmcmc 5000, stop at logZ_live - logZ < ln(0.001).

Priors are WIDE, set by the surrogate's training range (model_4 README:
q in [1,10], |chi| <= 0.9, t_min = 2 s/M_sun) rather than by GW150914 — this also
answers referee comment 1(c). Mc >= 15 M_sun keeps the 20 Hz inspiral inside the
4 s segment (tau_N(20 Hz; Mc=15) ~= 2.4 s, + 0.4 s Tukey taper).
The volumetric distance prior is capped at 500 Mpc (brackets GW150914's 440 Mpc):
with the design PSD this gives median network SNR ~7 with a tail to strong
signals. A 2000 Mpc cap gives median SNR 1.7 (98% of injections essentially
noise) — still a valid P-P test, but one that barely exercises the masses.

Stop-and-continue: every injection writes <run-dir>/inj_NNN/{result.json,
posterior_samples.npz}; a result.json with completed=true is skipped on re-run.
Use --start/--stop to split the campaign across Colab sessions / GPUs.

    # full campaign (GPU)
    python run_pp_mlgw_bbh.py --run-dir /content/drive/MyDrive/gwjax_pp/mlgw_bbh_pp

    # half of it on a second GPU
    python run_pp_mlgw_bbh.py --run-dir ... --start 50 --stop 100

    # local smoke test (pipeline only, NOT calibrated)
    python run_pp_mlgw_bbh.py --run-dir /tmp/pp_smoke --n-inj 2 --nlive 60 \
        --num-delete 30 --n-target 5 --max-mcmc 200 --max-iters 40 --num-posterior 100

Afterwards:  python make_pp_plot.py --run-dir <run-dir>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run-dir", default="pp_mlgw_bbh_run")
ap.add_argument("--n-inj", type=int, default=100)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--stop", type=int, default=None)
# data segment — same as the paper's GW150914 analysis
ap.add_argument("--detectors", nargs="+", default=["H1", "L1"])
ap.add_argument("--duration", type=float, default=4.0)
ap.add_argument("--sampling-rate", type=float, default=4096.0)
ap.add_argument("--f-min", type=float, default=20.0)
ap.add_argument("--f-max", type=float, default=1024.0)
ap.add_argument("--mlgw-final-time", type=float, default=0.12,
                help="post-peak time kept on the TD grid [s]. Must cover mlgw's native end "
                     "(0.00052 s/M_sun * M_tot, i.e. 0.104 s at the prior's max M_tot~200): "
                     "the paper's 0.01 s truncates the ringdown (MM~3e-3 at M=65, near-peak at M>150)")
ap.add_argument("--modes", default="22",
                help="comma-separated mlgw modes, e.g. '22' (paper) or '22,21,33,44'")
# priors
ap.add_argument("--mc", type=float, nargs=2, default=(15.0, 50.0))
ap.add_argument("--q", type=float, nargs=2, default=(0.125, 1.0))
ap.add_argument("--chi", type=float, nargs=2, default=(-0.9, 0.9))
ap.add_argument("--distance", type=float, nargs=2, default=(50.0, 500.0))
ap.add_argument("--tc-halfwidth", type=float, default=0.1)
# sampler (Prathaban et al. 2025 injection-study settings)
ap.add_argument("--nlive", type=int, default=1400)
ap.add_argument("--num-delete", type=int, default=700)
ap.add_argument("--n-target", type=int, default=60)
ap.add_argument("--max-mcmc", type=int, default=5000)
ap.add_argument("--max-iters", type=int, default=20000)
ap.add_argument("--log-dlogz", type=float, default=float(np.log(1e-3)))
ap.add_argument("--num-posterior", type=int, default=4000)
# seeds
ap.add_argument("--inj-seed", type=int, default=20261104)
ap.add_argument("--noise-seed0", type=int, default=1000)
ap.add_argument("--sampler-seed0", type=int, default=7000)
ap.add_argument("--force", action="store_true", help="re-run completed injections")
args, _ = ap.parse_known_args()   # tolerate Jupyter's -f flag

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import gwgpu_jax
from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator

RUN_DIR = Path(args.run_dir)
RUN_DIR.mkdir(parents=True, exist_ok=True)
MODES = [tuple(int(c) for c in m.strip()) for m in args.modes.split(",")]
MODES_ARG = MODES[0] if len(MODES) == 1 else MODES

# ── grid / waveform ──────────────────────────────────────────────────────────
grid = gwgpu_jax.TimeFrequencyGrid(duration=args.duration, sampling_rate=args.sampling_rate,
                                   f_min=args.f_min, f_max=args.f_max)
FREQS = grid.frequency_domain_array

# TD->FD recipe of the paper notebook / bbh_case runner (zero-crossing time grid +
# asymmetric Tukey taper), but with a post-peak span long enough for the whole
# ringdown. mlgw returns exact zeros past its native end, so a long span is safe.
# NB: the model's merger sits at (tc - MLGW_FINAL_TIME) mod duration.
MLGW_FINAL_TIME = args.mlgw_final_time
TUKEY_ALPHA_LEFT, TUKEY_ALPHA_RIGHT = 0.2, 0.001


def _tukey_asymmetric(n, a_l, a_r):
    x = jnp.linspace(0.0, 1.0, n)
    w = jnp.ones_like(x)
    left = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * (x / a_l - 0.5)))
    right = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * ((x - 1.0) / a_r + 0.5)))
    w = jnp.where(x < a_l / 2.0, left, w)
    return jnp.where(x > 1.0 - a_r / 2.0, right, w)


_bbh = MLGWBBHGenerator(model=4)
_gen = _bbh.generate
_dt, _N = grid.dt, grid.n_samples
_T_GRID = jnp.arange(_N) * _dt - grid.duration + MLGW_FINAL_TIME
_WINDOW = _tukey_asymmetric(_N, TUKEY_ALPHA_LEFT, TUKEY_ALPHA_RIGHT)


def mc_q_to_m1_m2(mc, q):
    m1 = mc * (1.0 + q) ** 0.2 / q ** 0.6
    return m1, q * m1


def waveform_fn(params, freqs):
    m1, m2 = mc_q_to_m1_m2(params["mc"], params["q"])
    theta = jnp.stack([m1, m2, params["chi_1"], params["chi_2"],
                       params["distance"], params["inclination"], params["phi_c"]])
    hp_td, hc_td = _gen(theta, _T_GRID, modes=MODES_ARG)
    return (jnp.fft.rfft(hp_td * _WINDOW) * _dt,
            -jnp.fft.rfft(hc_td * _WINDOW) * _dt)


# ── priors (used to draw injections AND inside the sampler) ─────────────────
TC_CENTER = 0.875 * args.duration       # paper convention (3.5 s of a 4 s segment)
PI = float(np.pi)
PARAM_BOUNDS = {
    "mc":          tuple(args.mc),
    "q":           tuple(args.q),
    "chi_1":       tuple(args.chi),
    "chi_2":       tuple(args.chi),
    "distance":    tuple(args.distance),
    "inclination": (0.0, PI),
    "ra":          (0.0, 2 * PI),
    "dec":         (-PI / 2, PI / 2),
    "psi":         (0.0, PI),
    "phi_c":       (0.0, 2 * PI),
    "tc":          (TC_CENTER - args.tc_halfwidth, TC_CENTER + args.tc_halfwidth),
}
PRIORS = {"inclination": "sin", "dec": "cos", "distance": "volumetric"}
NAMES = list(PARAM_BOUNDS)

cfg = dict(vars(args), modes=args.modes, param_bounds=PARAM_BOUNDS, priors=PRIORS,
           waveform="mlgw_bbh_jax model_4 (SEOBNRv5HM)", tc_center=TC_CENTER,
           mlgw_final_time=MLGW_FINAL_TIME,
           tukey=(TUKEY_ALPHA_LEFT, TUKEY_ALPHA_RIGHT),
           gwgpu_jax_version=getattr(gwgpu_jax, "__version__", "?"))
cfg_file = RUN_DIR / "config.json"
if cfg_file.exists():
    old = json.load(open(cfg_file))
    for k in ("param_bounds", "priors", "inj_seed", "modes", "duration", "f_min", "f_max",
              "mlgw_final_time"):
        if json.dumps(old.get(k)) != json.dumps(cfg.get(k)):
            raise SystemExit(f"config mismatch on {k!r} vs existing {cfg_file}: "
                             f"{old.get(k)} != {cfg.get(k)} — use a new --run-dir")
else:
    json.dump(cfg, open(cfg_file, "w"), indent=2, default=list)

# ── injections (cached so truths persist across restarts) ───────────────────
INJ_FILE = RUN_DIR / "injections.json"
if INJ_FILE.exists():
    injections = json.load(open(INJ_FILE))
else:
    specs = gwgpu_jax.resolve_priors(PARAM_BOUNDS, PRIORS)
    draw = gwgpu_jax.sample_prior(jax.random.PRNGKey(args.inj_seed), args.n_inj, specs)[0]
    injections = [{n: float(np.asarray(draw[n])[i]) for n in NAMES} for i in range(args.n_inj)]
    json.dump(injections, open(INJ_FILE, "w"), indent=2)
print(f"{len(injections)} injections in {INJ_FILE}")


def run_one(i, truth):
    inj_dir = RUN_DIR / f"inj_{i:03d}"
    res_file = inj_dir / "result.json"
    if res_file.exists() and not args.force and json.load(open(res_file)).get("completed"):
        return None
    inj_dir.mkdir(exist_ok=True)

    network = gwgpu_jax.Network.from_names(args.detectors, grid)
    network.generate_noise(seed=args.noise_seed0 + i)
    hp, hc = waveform_fn({k: jnp.asarray(v) for k, v in truth.items()}, FREQS)
    gmst = float(network.gmst)
    h = {}
    for ifo in network.interferometers:
        Fp, Fc = ifo.antenna_pattern(truth["ra"], truth["dec"], truth["psi"], gmst)
        dt = ifo.time_delay_from_geocenter(truth["ra"], truth["dec"], gmst)
        h[ifo.name] = gwgpu_jax.waveform_projection_fd(hp, hc, Fp, Fc, FREQS, truth["tc"] + dt)
    snr = float(network.network_optimal_snr(h))
    network.inject_signal(h, domain="fd")

    sampler = gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler(
        network=network, waveform_fn=waveform_fn, param_bounds=PARAM_BOUNDS,
        priors=PRIORS, gmst=None, periodic_params=("phi_c", "ra", "psi"))
    t0 = time.perf_counter()
    res = sampler.run(
        rng_key=jax.random.PRNGKey(args.sampler_seed0 + i),
        num_live=args.nlive, n_target=args.n_target, max_mcmc=args.max_mcmc,
        num_delete=args.num_delete, max_iterations=args.max_iters,
        log_dlogz_target=args.log_dlogz, num_posterior_samples=args.num_posterior,
        verbose=False)
    elapsed = time.perf_counter() - t0

    post = {n: np.asarray(res.posterior_samples[n]) for n in NAMES}
    np.savez_compressed(inj_dir / "posterior_samples.npz", **post)
    rec = dict(index=i, completed=True, truth=truth, network_snr=snr,
               p_values={n: float(np.mean(post[n] < truth[n])) for n in NAMES},
               logZ=float(res.logZ), logZ_err=float(res.logZ_err), ess=float(res.ess),
               n_iterations=int(res.n_iterations), elapsed_s=elapsed)
    json.dump(rec, open(res_file, "w"), indent=2)
    return rec


stop = len(injections) if args.stop is None else min(args.stop, len(injections))
for i in range(args.start, stop):
    rec = run_one(i, injections[i])
    if rec is None:
        print(f"[{i:03d}] already complete — skipped")
        continue
    print(f"[{i:03d}] SNR={rec['network_snr']:5.1f}  ESS={rec['ess']:6.0f}  "
          f"iters={rec['n_iterations']:4d}  logZ={rec['logZ']:+9.2f}  "
          f"{rec['elapsed_s']:6.0f} s", flush=True)
print("done. Plot with:  python make_pp_plot.py --run-dir", RUN_DIR)
