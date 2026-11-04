#!/usr/bin/env python
"""Generate the frozen BBH injection for the mlgw_bbh vs bilby benchmark.

A "golden" aligned-spin BBH injection (SEOBNRv5HM, (2,2) mode only — the
model `mlgw_bbh_jax` is a surrogate of), consistent with the trained
parameter box of the existing real-GW150914 notebooks
(`examples/NEWParameter_Estimation_mlgw_bbh_GW150914_acceptance_walk_VSCode.ipynb`).

Unlike `frozen_benchmark` (which used the reference paper's exact ripple
IMRPhenomD injection), this is a fresh injection generated here, through the
*production* mlgw_bbh waveform model, so that the downstream comparison
(`run_gwjax_bbh.py` vs `run_bilby_bbh.py` recovering with LAL `SEOBNRv5_ROM`
— the official LAL ROM of the same SEOBNRv5 family) exercises the mlgw NN
surrogate + the acceptance-walk pipeline, not just the sampler/likelihood
plumbing.

Before freezing, this script also cross-checks the mlgw_bbh injection against
an independent ripplegw `IMRPhenomXAS` waveform at the same parameters
(mismatch + logL delta), triangulating: mlgw surrogate vs ripple analytic vs
(downstream) LAL SEOBNRv5_ROM. ripple is independently validated to agree
with LAL to ~1e-9 (see `frozen_benchmark/README.md`), so a small mlgw-vs-ripple
difference consistent with the expected SEOBNRv5-vs-IMRPhenomX model
systematic is the expected, healthy outcome here; a gross mismatch would flag
an mlgw-side problem.

Output: `data/` (frequency_array.npy, PSD + strain per IFO) + `metadata.json`
(injection params, GPS times, sha256 checksums, SNRs, triangulation numbers)
— consumed by both `run_gwjax_bbh.py` and `run_bilby_bbh.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--outdir", default=None, help="output dir (default: ./data next to this script)")
parser.add_argument("--distance", type=float, default=127.0, help="luminosity distance [Mpc]")
parser.add_argument("--seed", type=int, default=1234, help="noise realisation seed")
args = parser.parse_args()

OUT = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import gwgpu_jax
from gwgpu_jax.gwgpu_jax_compatibility import gmst_from_gps
from gwgpu_jax.gwgpu_jax_likelihood_utils import optimal_snr_squared

# ── Detector network: GW150914-standard 4 s @ 4096 Hz, 20-1024 Hz, H1+L1 ─────
DURATION, SAMPLING_RATE = 4.0, 4096.0
F_MIN, F_MAX = 20.0, 1024.0
TC_CENTER = 0.875 * DURATION           # 3.5 s — PHYSICAL merger time in the segment (= trigger)
TRIGGER_TIME = 1126259642.413          # arbitrary but realistic GPS epoch (GW150914-era, reused
                                        # for continuity with frozen_benchmark's precedent of
                                        # reusing this epoch for unrelated synthetic injections)
START_TIME = TRIGGER_TIME - TC_CENTER

grid = gwgpu_jax.TimeFrequencyGrid(
    duration=DURATION, sampling_rate=SAMPLING_RATE, f_min=F_MIN, f_max=F_MAX,
)
net = gwgpu_jax.Network.from_names(["H1", "L1"], grid)   # default aLIGO design PSD
net.gmst = gmst_from_gps(TRIGGER_TIME)

# ── Injection parameters ("golden": inside the trained mc/q box, away from
#    prior edges and orientation degeneracies) ───────────────────────────────
INJECTION = dict(
    mc=28.0, q=0.85, chi_1=0.10, chi_2=-0.05,
    distance=args.distance, inclination=1.0,
    ra=1.375, dec=-1.2108, psi=0.8, phi_c=1.3,
)


def mc_q_to_m1_m2(mc, q):
    m1 = mc * (1.0 + q) ** (1.0 / 5.0) / q ** (3.0 / 5.0)
    return m1, q * m1


m1, m2 = mc_q_to_m1_m2(INJECTION["mc"], INJECTION["q"])
print(f"m1={float(m1):.3f} Msun  m2={float(m2):.3f} Msun  (from mc={INJECTION['mc']}, q={INJECTION['q']})")

# ── mlgw_bbh (SEOBNRv5HM, (2,2) mode) TD -> FD waveform, self-contained ─────
# Recipe copied from NEWParameter_Estimation_mlgw_bbh_GW150914_acceptance_walk_VSCode.ipynb
# (cell 12) — see that notebook for the full rationale (zero-crossing time
# grid + asymmetric Tukey taper, both essential for real-data-quality PE).
from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator

# Post-merger span kept on the TD grid. 0.1 s covers the whole ringdown (mlgw returns
# exact zeros past its native end, 0.00052 s/M_sun x M_tot = 34 ms here). The previous
# 0.01 s truncated the ringdown at ~15% of peak amplitude (mismatch ~3e-3 vs SEOBNRv5HM;
# see examples/referee_response/waveform_check/).
MLGW_FINAL_TIME = 0.1
# mlgw's template merger sits at (tc - MLGW_FINAL_TIME), so the mlgw `tc` parameter
# that puts the merger at the trigger is:
INJECTION["tc"] = TC_CENTER + MLGW_FINAL_TIME
TUKEY_ALPHA_LEFT, TUKEY_ALPHA_RIGHT = 0.2, 0.001


def _tukey_asymmetric(n, alpha_left, alpha_right):
    x = jnp.linspace(0.0, 1.0, n)
    w = jnp.ones_like(x)
    left = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * (x / alpha_left - 0.5)))
    right = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * ((x - 1.0) / alpha_right + 0.5)))
    w = jnp.where(x < alpha_left / 2.0, left, w)
    w = jnp.where(x > 1.0 - alpha_right / 2.0, right, w)
    return w


def build_mlgw_bbh_waveform_fn(grid, modes=(2, 2),
                                final_time=MLGW_FINAL_TIME,
                                alpha_left=TUKEY_ALPHA_LEFT,
                                alpha_right=TUKEY_ALPHA_RIGHT):
    bbh = MLGWBBHGenerator(model=4)   # model=4 (int) resolves the SEOBNRv5HM weights
    dt, N = grid.dt, grid.n_samples
    t_grid = jnp.arange(N) * dt - grid.duration + final_time
    window = _tukey_asymmetric(N, alpha_left, alpha_right)
    gen_wf = bbh.generate

    def waveform_fn(params, freqs):
        mc, q = params["mc"], params["q"]
        m1 = mc * (1.0 + q) ** (1.0 / 5.0) / q ** (3.0 / 5.0)
        m2 = q * m1
        theta = jnp.stack([
            m1, m2, params["chi_1"], params["chi_2"],
            params["distance"], params["inclination"], params["phi_c"],
        ])
        hp_td, hc_td = gen_wf(theta, t_grid, modes=modes)
        hp_fd = jnp.fft.rfft(hp_td * window) * dt
        hc_fd = jnp.fft.rfft(hc_td * window) * dt
        return hp_fd, -hc_fd

    return waveform_fn


t0 = time.perf_counter()
mlgw_waveform_fn = build_mlgw_bbh_waveform_fn(grid, modes=(2, 2))
print(f"mlgw_bbh_jax (SEOBNRv5HM, model_4, (2,2) mode) loaded in {time.perf_counter()-t0:.1f} s")

mlgw_params = {k: jnp.asarray(v) for k, v in INJECTION.items() if k != "ra" and k != "dec" and k != "psi"}
hp_mlgw, hc_mlgw = mlgw_waveform_fn(mlgw_params, grid.frequency_domain_array)

# ── Project onto H1/L1 and inject ────────────────────────────────────────────
freqs = grid.frequency_domain_array
gmst = float(net.gmst)
ra, dec, psi = INJECTION["ra"], INJECTION["dec"], INJECTION["psi"]
h_dict = {}
for ifo in net.interferometers:
    Fp, Fc = ifo.antenna_pattern(ra, dec, psi, gmst)
    dt_ifo = ifo.time_delay_from_geocenter(ra, dec, gmst)
    h_dict[ifo.name] = gwgpu_jax.waveform_projection_fd(
        hp_mlgw, hc_mlgw, Fp, Fc, freqs, INJECTION["tc"] + dt_ifo)

snrs = net.optimal_snr(h_dict)
net_snr = float(net.network_optimal_snr(h_dict))
print("mlgw_bbh injection:")
for name, snr in snrs.items():
    print(f"    {name}: SNR = {float(snr):.2f}")
print(f"    network SNR = {net_snr:.2f}")

net.generate_noise(seed=args.seed)
net.inject_signal(h_dict, domain="fd")

# ── Ripplegw triangulation check (IMRPhenomXAS, closest aligned-spin ripple
#    family to SEOBNRv5) — mismatch + logL delta at the injection point ─────
ripple_waveform_fn = gwgpu_jax.build_ripplegw_waveform_fn("IMRPhenomXAS", f_ref=20.0)
ripple_params = {
    "m1": m1, "m2": m2, "chi_1": INJECTION["chi_1"], "chi_2": INJECTION["chi_2"],
    "distance": INJECTION["distance"], "inclination": INJECTION["inclination"],
    "phi_c": INJECTION["phi_c"],
}
hp_ripple, hc_ripple = ripple_waveform_fn(ripple_params, freqs)

print("\nripplegw (IMRPhenomXAS) triangulation check:")
# ripplegw's analytic model returns a NaN DC (f=0) bin; that bin is already
# excluded from the analysis band by grid.frequency_mask, but nan * 0 is
# still nan (not 0) in IEEE arithmetic, so multiplying by the mask would not
# clear it — use jnp.where to actually zero it out.
band_mask = grid.frequency_mask
hp_ripple = jnp.where(band_mask, hp_ripple, 0.0)
hc_ripple = jnp.where(band_mask, hc_ripple, 0.0)

# mlgw's zero-crossing TD grid places the template merger at tc - MLGW_FINAL_TIME,
# while ripple (and bilby/LAL) put it at tc. The mlgw injection used
# tc = TC_CENTER + MLGW_FINAL_TIME, so ripple is projected at the physical merger
# TC_CENTER; the residual best dtc should then be ~0. A small tc scan still gives
# the standard time-and-phase-maximised overlap.
TC_SCAN = np.linspace(-0.05, 0.05, 201)

for ifo in net.interferometers:
    Fp, Fc = ifo.antenna_pattern(ra, dec, psi, gmst)
    dt_ifo = ifo.time_delay_from_geocenter(ra, dec, gmst)
    h_mlgw_ifo = h_dict[ifo.name]
    w = ifo.weight
    norm_mlgw = jnp.sqrt(jnp.sum(jnp.abs(h_mlgw_ifo) ** 2 * w))

    best_overlap, best_dtc, best_h_ripple = -1.0, 0.0, None
    for dtc in TC_SCAN:
        h_r = gwgpu_jax.waveform_projection_fd(
            hp_ripple, hc_ripple, Fp, Fc, freqs, TC_CENTER + dt_ifo + dtc)
        num = jnp.sum(jnp.conj(h_mlgw_ifo) * h_r * w)
        den = norm_mlgw * jnp.sqrt(jnp.sum(jnp.abs(h_r) ** 2 * w))
        ov = float(jnp.abs(num) / den)
        if ov > best_overlap:
            best_overlap, best_dtc, best_h_ripple = ov, float(dtc), h_r

    logl_mlgw = float(ifo.log_likelihood_data(h_mlgw_ifo))
    logl_ripple = float(ifo.log_likelihood_data(best_h_ripple))
    print(f"    {ifo.name}: time-and-phase-maximized overlap={best_overlap:.6f}  "
          f"mismatch={1.0-best_overlap:.2e}  (best dtc={best_dtc*1e3:+.2f} ms)  "
          f"logL(mlgw)={logl_mlgw:+.3f}  logL(ripple)={logl_ripple:+.3f}  "
          f"delta={logl_mlgw-logl_ripple:+.4f}")

# ── Freeze to data/ + metadata.json ──────────────────────────────────────────
freqs_np = np.asarray(freqs)
np.save(OUT / "frequency_array.npy", freqs_np)
for ifo in net.interferometers:
    np.save(OUT / f"{ifo.name}_strain_fd.npy", np.asarray(ifo.strain_data_fd))
    np.save(OUT / f"{ifo.name}_psd.npy", np.asarray(ifo.psd))


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


sha = {"frequency_array": _sha(freqs_np)}
for ifo in net.interferometers:
    sha[f"{ifo.name}_strain_fd"] = _sha(np.asarray(ifo.strain_data_fd))
    sha[f"{ifo.name}_psd"] = _sha(np.asarray(ifo.psd))

meta = dict(
    source="generated by generate_injection.py — mlgw_bbh_jax (SEOBNRv5HM, model_4, (2,2) mode)",
    event="golden synthetic BBH injection for the mlgw_bbh vs bilby benchmark",
    injection=dict(
        chirp_mass=INJECTION["mc"], mass_ratio=INJECTION["q"],
        chi_1=INJECTION["chi_1"], chi_2=INJECTION["chi_2"],
        luminosity_distance=INJECTION["distance"], theta_jn=INJECTION["inclination"],
        psi=INJECTION["psi"], phase=INJECTION["phi_c"],
        geocent_time=TRIGGER_TIME, ra=INJECTION["ra"], dec=INJECTION["dec"],
        mass_1=float(m1), mass_2=float(m2),
    ),
    duration=DURATION, sampling_frequency=SAMPLING_RATE,
    minimum_frequency=F_MIN, maximum_frequency=F_MAX,
    start_time=START_TIME, trigger_time=TRIGGER_TIME,
    tc_in_segment=TC_CENTER, reference_frequency=20.0,
    # mlgw template convention: merger at (tc - mlgw_final_time). Recover with the
    # same value, tc prior centred on tc_in_segment + mlgw_final_time.
    mlgw_final_time=MLGW_FINAL_TIME, mlgw_tc_param=INJECTION["tc"],
    detectors=net.names, noise_seed=args.seed,
    network_optimal_snr_of_injected_template=net_snr,
    per_ifo_optimal_snr={k: float(v) for k, v in snrs.items()},
    sha256_16=sha,
)
json.dump(meta, open(OUT / "metadata.json", "w"), indent=2)
print(f"\nfrozen BBH injection written to {OUT}")
