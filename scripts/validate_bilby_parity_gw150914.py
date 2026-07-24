"""Cross-validate gwgpu_jax against bilby on real GW150914 GWOSC data.

Sections
A. GWOSC import  — same TD arrays as bilby's set_from_gwpy_timeseries
B. Window        — gwjax Tukey vs bilby's time_domain_window
C. FD strain     — gwjax rfft*dt vs bilby frequency_domain_strain
D. PSD           — gwjax psd_from_data (median Welch) vs gwpy/bilby-pipe median
E. Geometry      — antenna patterns, time delays, GMST vs bilby/LAL
F. Likelihood    — identical data+PSD+ripple waveform into both pipelines
"""
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import gwgpu_jax
from gwgpu_jax.gwgpu_jax_compatibility import (
    get_event_info, timeseries_to_ifo, gmst_from_gps)
from gwgpu_jax.gwgpu_jax_psd_utils import psd_from_data

import bilby
import lal
from gwpy.timeseries import TimeSeries

EVENT     = "GW150914"
DETS      = ["H1", "L1"]
DURATION  = 4.0
FS        = 4096.0
F_MIN, F_MAX = 20.0, 1024.0
ROLL_OFF  = 0.2           # bilby InterferometerStrainData default
PSD_SEG   = 32.0          # off-source stretch for PSD
PSD_OFF   = 8.0

info   = get_event_info(EVENT)
gps    = info.gps_time
pre    = 0.875 * DURATION

def sec(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)

# ── Fetch one long stretch per detector (covers on-source + PSD window) ──────
fetch_start = gps - pre - PSD_SEG - PSD_OFF
fetch_end   = gps + (DURATION - pre) + 2.0
raw = {d: TimeSeries.fetch_open_data(d, fetch_start, fetch_end,
                                     sample_rate=int(FS), cache=True)
       for d in DETS}
print("fetched:", {d: (float(ts.t0.value), ts.size) for d, ts in raw.items()})

# ── gwjax network ────────────────────────────────────────────────────────────
grid = gwgpu_jax.TimeFrequencyGrid(DURATION, FS, f_min=F_MIN, f_max=F_MAX)
net  = gwgpu_jax.Network.from_names(DETS, grid)

# Install with NO window / NO detrend first (raw-array comparison), then redo
# with the Tukey window for the FD comparison.
for ifo in net.interferometers:
    timeseries_to_ifo(raw[ifo.name], ifo, event_gps=gps, pre_merger=pre,
                      bandpass=None, detrend=False, window=None)
epoch = float(net.interferometers[0].data_epoch)
td_raw = {i.name: np.asarray(i.strain_data_td) for i in net.interferometers}

# ── bilby interferometers from the SAME gwpy segments ────────────────────────
b_ifos = {}
for d in DETS:
    cropped = raw[d].crop(epoch, epoch + DURATION)
    bi = bilby.gw.detector.get_empty_interferometer(d)
    bi.minimum_frequency = F_MIN
    bi.maximum_frequency = F_MAX
    bi.strain_data.roll_off = ROLL_OFF
    bi.strain_data.set_from_gwpy_timeseries(cropped)
    b_ifos[d] = bi

# ═════ A. TD arrays ══════════════════════════════════════════════════════════
sec("A. GWOSC time-domain import (raw segment, no window/detrend)")
for d in DETS:
    b_td = b_ifos[d].strain_data.time_domain_strain[:grid.n_samples]
    g_td = td_raw[d]
    denom = np.max(np.abs(b_td))
    print(f"{d}: n_bilby={b_ifos[d].strain_data.time_domain_strain.size} "
          f"n_gwjax={g_td.size} start_bilby={b_ifos[d].strain_data.start_time:.6f} "
          f"epoch_gwjax={epoch:.6f}")
    print(f"   max |Δtd| / max|td| = {np.max(np.abs(g_td - b_td)) / denom:.3e}")

# ═════ B. Window function ════════════════════════════════════════════════════
sec("B. Tukey window")
from scipy.signal.windows import tukey
w_gw = tukey(grid.n_samples, 2.0 * ROLL_OFF / DURATION)
d0 = DETS[0]
w_b  = b_ifos[d0].strain_data.time_domain_window()   # bilby stores/returns window
w_b  = np.asarray(w_b)[:grid.n_samples] if np.ndim(w_b) else None
if w_b is not None and w_b.size == w_gw.size:
    print(f"max |Δwindow| = {np.max(np.abs(w_gw - w_b)):.3e}")
    print(f"bilby window_factor = {b_ifos[d0].strain_data.window_factor:.12f}, "
          f"mean(w^2) gwjax = {np.mean(w_gw**2):.12f}")
else:
    print("bilby time_domain_window returned scalar/none; window_factor =",
          b_ifos[d0].strain_data.window_factor)

# ═════ C. FD strain ══════════════════════════════════════════════════════════
sec("C. Frequency-domain strain (Tukey-windowed, detrend OFF)")
for ifo in net.interferometers:
    timeseries_to_ifo(raw[ifo.name], ifo, event_gps=gps, pre_merger=pre,
                      bandpass=None, detrend=False, window="tukey",
                      roll_off=ROLL_OFF)
mask = np.asarray(grid.frequency_mask)
for d in DETS:
    g_fd = np.asarray(dict((i.name, i) for i in net.interferometers)[d].strain_data_fd)
    b_fd = np.asarray(b_ifos[d].frequency_domain_strain)
    fa_b = np.asarray(b_ifos[d].frequency_array)
    fa_g = np.asarray(grid.frequency_domain_array)
    assert fa_b.size == fa_g.size and np.max(np.abs(fa_b - fa_g)) == 0.0
    scale = np.median(np.abs(b_fd[mask]))
    print(f"{d}: in-band max|Δfd|/median|fd| = "
          f"{np.max(np.abs(g_fd[mask] - b_fd[mask])) / scale:.3e}")
    # bilby frequency mask vs gwjax mask
    bm = np.asarray(b_ifos[d].strain_data.frequency_mask)
    print(f"   frequency-mask identical: {bool(np.array_equal(bm, mask))} "
          f"(n_band bilby={bm.sum()}, gwjax={mask.sum()})")

# detrend effect (gwjax default detrend=True vs bilby's no-detrend)
ifo0 = net.interferometers[0]
fd_nodetrend = np.asarray(ifo0.strain_data_fd).copy()
timeseries_to_ifo(raw[d0], ifo0, event_gps=gps, pre_merger=pre,
                  bandpass=None, detrend=True, window="tukey", roll_off=ROLL_OFF)
fd_detrend = np.asarray(ifo0.strain_data_fd)
rel = np.max(np.abs(fd_detrend[mask] - fd_nodetrend[mask])) / np.median(np.abs(fd_nodetrend[mask]))
print(f"\ndetrend=True vs False ({d0}): in-band max relative change = {rel:.3e}")
# restore detrend=False data for the likelihood section (matches bilby exactly)
timeseries_to_ifo(raw[d0], ifo0, event_gps=gps, pre_merger=pre,
                  bandpass=None, detrend=False, window="tukey", roll_off=ROLL_OFF)

# ═════ D. PSD ════════════════════════════════════════════════════════════════
sec("D. PSD: gwjax median-Welch vs gwpy/bilby-pipe median")
freqs = np.asarray(grid.frequency_domain_array)
psd_gw, psd_gwpy, psd_matched = {}, {}, {}
for d in DETS:
    off = raw[d].crop(gps - PSD_OFF - PSD_SEG, gps - PSD_OFF)
    # gwjax default path: welch, hann, nperseg = N//8 (= 4 s here), 50% overlap, median
    _, p = psd_from_data(np.asarray(off.value, dtype=np.float64), FS,
                         target_freqs=grid.frequency_domain_array)
    psd_gw[d] = np.asarray(p)
    # bilby_pipe standard: gwpy .psd, fftlength=duration, overlap=0,
    # tukey window with the analysis roll-off, method='median'
    alpha = 2.0 * ROLL_OFF / DURATION
    pg = off.psd(fftlength=DURATION, overlap=0, window=("tukey", alpha),
                 method="median")
    psd_gwpy[d] = np.interp(freqs, np.asarray(pg.frequencies.value),
                            np.asarray(pg.value))
    # matched-settings check: gwpy with hann + 50% overlap (gwjax's settings)
    pm = off.psd(fftlength=DURATION, overlap=DURATION / 2, window="hann",
                 method="median")
    psd_matched[d] = np.interp(freqs, np.asarray(pm.frequencies.value),
                               np.asarray(pm.value))
    r_pipe    = psd_gw[d][mask] / psd_gwpy[d][mask]
    r_matched = psd_gw[d][mask] / psd_matched[d][mask]
    print(f"{d}: gwjax/gwpy(bilby-pipe cfg) ratio: median={np.median(r_pipe):.4f} "
          f"[16,84]%=[{np.percentile(r_pipe,16):.3f},{np.percentile(r_pipe,84):.3f}]")
    print(f"    gwjax/gwpy(matched hann+50%) ratio: median={np.median(r_matched):.4f} "
          f"max|1-r|={np.max(np.abs(1 - r_matched)):.3e}")

# ═════ E. Geometry: antenna patterns, delays, GMST ═══════════════════════════
sec("E. Detector geometry vs bilby/LAL")
g_lal  = lal.GreenwichMeanSiderealTime(gps) % (2 * np.pi)
g_gwx  = gmst_from_gps(gps)
print(f"GMST(trigger): gwjax={g_gwx:.12f}  lal={g_lal:.12f}  "
      f"|Δ|={abs(g_gwx - g_lal):.3e} rad")

rng = np.random.default_rng(0)
n_sky = 200
ras  = rng.uniform(0, 2 * np.pi, n_sky)
decs = np.arcsin(rng.uniform(-1, 1, n_sky))
psis = rng.uniform(0, np.pi, n_sky)
maxF, maxT = 0.0, 0.0
for d in DETS:
    gi = dict((i.name, i) for i in net.interferometers)[d]
    bi = b_ifos[d]
    for ra, de, ps in zip(ras, decs, psis):
        Fp_g, Fc_g = gi.antenna_pattern(ra, de, ps, g_gwx)
        Fp_b = bi.antenna_response(ra, de, gps, ps, "plus")
        Fc_b = bi.antenna_response(ra, de, gps, ps, "cross")
        dt_g = gi.time_delay_from_geocenter(ra, de, g_gwx)
        dt_b = bi.time_delay_from_geocenter(ra, de, gps)
        maxF = max(maxF, abs(float(Fp_g) - Fp_b), abs(float(Fc_g) - Fc_b))
        maxT = max(maxT, abs(float(dt_g) - dt_b))
print(f"max |ΔF+,x| over {n_sky} sky points x {len(DETS)} dets = {maxF:.3e}")
print(f"max |Δ time-delay| = {maxT:.3e} s  (light-time scale ~1e-2 s)")

# ═════ F. Likelihood on identical data + PSD + waveform ══════════════════════
sec("F. Gaussian likelihood: gwjax vs bilby on identical inputs")
# Install the SAME PSD (gwjax median-Welch) into both pipelines.
for d in DETS:
    gi = dict((i.name, i) for i in net.interferometers)[d]
    gi.psd = jnp.asarray(psd_gw[d])
    gi._weight = None
    psd_b = psd_gw[d].copy()
    psd_b[~np.isfinite(psd_b)] = 1.0        # only out-of-band bins; masked anyway
    b_ifos[d].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freqs, psd_array=psd_b)
    # push the gwjax FD data (tukey, no detrend) into bilby verbatim
    b_ifos[d].set_strain_data_from_frequency_domain_strain(
        np.asarray(gi.strain_data_fd), sampling_frequency=FS,
        duration=DURATION, start_time=epoch)
    b_ifos[d].minimum_frequency = F_MIN
    b_ifos[d].maximum_frequency = F_MAX

net.gmst = gmst_from_gps(gps)

# ripple IMRPhenomD, shared by both sides
wf_fn = gwgpu_jax.build_ripplegw_waveform_fn("IMRPhenomD", f_ref=20.0)

PARAM_BOUNDS = {
    "m1": (20.0, 60.0), "m2": (15.0, 50.0),
    "chi_1": (-0.9, 0.9), "chi_2": (-0.9, 0.9),
    "distance": (100.0, 2000.0), "inclination": (0.0, np.pi),
    "ra": (0.0, 2 * np.pi), "dec": (-np.pi / 2, np.pi / 2),
    "psi": (0.0, np.pi), "phi_c": (0.0, 2 * np.pi),
    "tc": (3.0, 4.0),
}
gw_sampler = gwgpu_jax.GWgpu_jaxNestedSampler(
    network=net, waveform_fn=wf_fn, param_bounds=PARAM_BOUNDS,
    priors={"inclination": "sin", "dec": "cos", "distance": "volumetric"},
)
gw_loglik = gw_sampler.loglikelihood_fn

def ripple_source_model(frequency_array, mass_1, mass_2, chi_1, chi_2,
                        luminosity_distance, theta_jn, phase, **kwargs):
    p = {"m1": mass_1, "m2": mass_2, "chi_1": chi_1, "chi_2": chi_2,
         "distance": luminosity_distance, "inclination": theta_jn,
         "tc": 0.0, "phi_c": phase}
    hp, hc = wf_fn(p, jnp.asarray(frequency_array))
    hp = np.nan_to_num(np.asarray(hp))
    hc = np.nan_to_num(np.asarray(hc))
    return {"plus": hp, "cross": hc}

wg = bilby.gw.WaveformGenerator(
    duration=DURATION, sampling_frequency=FS,
    frequency_domain_source_model=ripple_source_model,
    start_time=epoch)
b_like = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=bilby.gw.detector.InterferometerList([b_ifos[d] for d in DETS]),
    waveform_generator=wg)

tc_star = gps - epoch   # tc that makes geocent_time == trigger (gmst-exact)
draws = []
for k in range(15):
    r = np.random.default_rng(100 + k)
    m1 = r.uniform(30, 45); m2 = r.uniform(22, m1)
    draws.append(dict(
        m1=m1, m2=m2,
        chi_1=r.uniform(-0.5, 0.5), chi_2=r.uniform(-0.5, 0.5),
        distance=r.uniform(300, 900), inclination=r.uniform(0.1, np.pi - 0.1),
        ra=r.uniform(0, 2 * np.pi), dec=np.arcsin(r.uniform(-1, 1)),
        psi=r.uniform(0, np.pi), phi_c=r.uniform(0, 2 * np.pi),
        tc=tc_star + r.uniform(-0.02, 0.02)))
draws.append(dict(m1=38.0, m2=32.0, chi_1=0.1, chi_2=-0.2, distance=420.0,
                  inclination=2.7, ra=1.95, dec=-1.27, psi=0.82, phi_c=1.3,
                  tc=tc_star))   # gmst-exact point, near-GW150914 params

rows = []
for p in draws:
    lg = float(gw_loglik({k: jnp.asarray(v) for k, v in p.items()}))
    b_like.parameters = dict(
        mass_1=p["m1"], mass_2=p["m2"], chi_1=p["chi_1"], chi_2=p["chi_2"],
        luminosity_distance=p["distance"], theta_jn=p["inclination"],
        phase=p["phi_c"], ra=p["ra"], dec=p["dec"], psi=p["psi"],
        geocent_time=epoch + p["tc"])
    lb = float(b_like.log_likelihood())
    rows.append((lg, lb, p["tc"] - tc_star))

lgs  = np.array([r[0] for r in rows])
lbs  = np.array([r[1] for r in rows])
dtc  = np.array([r[2] for r in rows])
diff = lgs - lbs
print(f"noise logL: bilby={float(b_like.noise_log_likelihood()):.6f}  "
      f"gwjax={float(sum(i.log_likelihood_data(jnp.zeros_like(i.strain_data_fd)) for i in net.interferometers)):.6f}")
print(f"{'tc-tc*':>9s} {'logL gwjax':>16s} {'logL bilby':>16s} {'Δ':>12s}")
for (lg, lb, dt) in rows:
    print(f"{dt:9.4f} {lg:16.6f} {lb:16.6f} {lg - lb:12.3e}")
print(f"\nmax|Δ| (all draws)          = {np.max(np.abs(diff)):.3e}")
print(f"|Δ| at tc=tc* (gmst-exact)  = {abs(diff[-1]):.3e}")
print(f"corr(|Δ|, |tc-tc*|)         = "
      f"{np.corrcoef(np.abs(diff[:-1]), np.abs(dtc[:-1]))[0,1]:.2f} "
      f"(bilby re-evaluates GMST at geocent_time; gwjax fixes GMST at trigger)")
print("\nDONE")
