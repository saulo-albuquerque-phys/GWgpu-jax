#!/usr/bin/env python
"""Step 2 (gwjax env): where does the mlgw_bbh <-> SEOBNRv5 difference come from?

Noise-weighted mismatch (design H1 PSD, 20-1024 Hz), maximised over time and phase,
at the bbh_case injection, between
  (a) mlgw TD  vs SEOBNRv5HM TD, BOTH through the paper's TD->FD recipe
      (4 s grid ending at +10 ms, asymmetric Tukey 0.2/0.001, rFFT)   -> surrogate (NN+PCA) error
  (b) SEOBNRv5HM through the recipe vs full-length SEOBNRv5HM         -> recipe (truncation/taper) error
  (c) mlgw through the recipe vs full-length SEOBNRv5HM               -> total, what bilby+ROM "sees"
and converts each to the expected log-likelihood loss at the injection SNR,
delta lnL ~= rho^2 * MM.

    python compare_mlgw_vs_seobnrv5hm.py --seob seob_td.npz
"""
import argparse, json
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import gwgpu_jax
from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator

ap = argparse.ArgumentParser()
ap.add_argument("--seob", default="seob_td.npz")
ap.add_argument("--meta", default="../../../drive_data/bbh_case/data/metadata.json")
ap.add_argument("--snr", type=float, default=None, help="default: metadata network SNR")
ap.add_argument("--f-min", type=float, default=20.0)
ap.add_argument("--f-max", type=float, default=1024.0)
ap.add_argument("--final-time", type=float, default=0.01, help="post-peak time kept [s] (paper: 0.01)")
ap.add_argument("--alpha-left", type=float, default=0.2, help="inspiral-side Tukey alpha (paper: 0.2)")
ap.add_argument("--alpha-right", type=float, default=0.001, help="ringdown-side Tukey alpha (paper: 0.001)")
ap.add_argument("--duration", type=float, default=4.0)
ap.add_argument("--out", default="mismatch_bbh_case.json")
a = ap.parse_args()

meta = json.load(open(a.meta)); inj = meta["injection"]
s = np.load(a.seob)
SR, DUR, FINAL = 4096.0, a.duration, a.final_time
dt = 1.0 / SR
N = int(DUR * SR)
t_grid = np.arange(N) * dt - DUR + FINAL

def tukey_asym(n, al, ar):
    x = np.linspace(0, 1, n); w = np.ones(n)
    l = x < al / 2; r = x > 1 - ar / 2
    w[l] = 0.5 * (1 + np.cos(2 * np.pi * (x[l] / al - 0.5)))
    w[r] = 0.5 * (1 + np.cos(2 * np.pi * ((x[r] - 1) / ar + 0.5)))
    return w
W = tukey_asym(N, a.alpha_left, a.alpha_right)

# mlgw on the recipe grid, (2,2) mode — exactly the paper's waveform_fn before the rFFT
gen = MLGWBBHGenerator(model=4).generate
th = jnp.array([inj["mass_1"], inj["mass_2"], inj["chi_1"], inj["chi_2"],
                inj["luminosity_distance"], inj["theta_jn"], inj["phase"]])
hp_ml, _ = gen(th, jnp.asarray(t_grid), modes=(2, 2))
hp_ml = np.asarray(hp_ml)

# SEOBNRv5HM interpolated onto the same grid (zero outside its support)
from scipy.interpolate import CubicSpline
cs = CubicSpline(s["t"], s["hp"])
inside = (t_grid >= s["t"][0]) & (t_grid <= s["t"][-1])
hp_se_grid = np.where(inside, cs(np.clip(t_grid, s["t"][0], s["t"][-1])), 0.0)

# common long buffer (16 s) so windowed-4s and full-length signals share one df
NB = int(16 * SR)
def place(h_td, t_td, taper_start=0):
    buf = np.zeros(NB)
    i0 = int(round((t_td[0] + 12.0) / dt))       # put t=0 at 12 s into the buffer
    x = h_td.copy()
    if taper_start:
        x[:taper_start] *= 0.5 * (1 - np.cos(np.pi * np.arange(taper_start) / taper_start))
    buf[i0:i0 + x.size] = x
    return np.fft.rfft(buf) * dt

f = np.fft.rfftfreq(NB, dt)
net = gwgpu_jax.Network.from_names(["H1", "L1"], gwgpu_jax.TimeFrequencyGrid(
    duration=16.0, sampling_rate=SR, f_min=a.f_min, f_max=a.f_max))
psd_full = np.asarray(net.interferometers[0].psd)
f_net = np.asarray(net.interferometers[0].grid.frequency_domain_array) \
    if hasattr(net.interferometers[0], "grid") else np.asarray(
        gwgpu_jax.TimeFrequencyGrid(16.0, SR, f_min=a.f_min, f_max=a.f_max).frequency_domain_array)
psd = np.interp(f, f_net, psd_full) if psd_full.size != f.size else psd_full
band = (f >= a.f_min) & (f <= a.f_max) & np.isfinite(psd) & (psd > 0)

def match(h1, h2):
    """max over time & phase of the normalised noise-weighted overlap."""
    w = np.zeros_like(f); w[band] = 1.0 / psd[band]
    n1 = np.sqrt(4 * np.sum(np.abs(h1) ** 2 * w))
    n2 = np.sqrt(4 * np.sum(np.abs(h2) ** 2 * w))
    z = np.zeros(2 * (f.size - 1), complex)
    z[:f.size] = np.conj(h1) * h2 * w
    c = np.fft.ifft(z) * z.size * 4                  # complex overlap vs time lag
    return float(np.max(np.abs(c)) / (n1 * n2))

H_ml_recipe = place(hp_ml * W, t_grid)
H_se_recipe = place(hp_se_grid * W, t_grid)
H_se_full = place(s["hp"], s["t"], taper_start=int(0.2 * SR))   # 10 Hz start, tapered well below band

rho = a.snr or meta.get("network_snr") or meta.get("snr", {}).get("network")
res = {}
for key, (h1, h2) in {
        "a_surrogate_mlgw_vs_seob_same_recipe": (H_ml_recipe, H_se_recipe),
        "b_recipe_seob_recipe_vs_seob_full":   (H_se_recipe, H_se_full),
        "c_total_mlgw_recipe_vs_seob_full":    (H_ml_recipe, H_se_full)}.items():
    mm = 1 - match(h1, h2)
    res[key] = dict(mismatch=mm, delta_lnL_at_snr=(rho ** 2 * mm) if rho else None)
    print(f"{key:40s} MM = {mm:.2e}" + (f"   rho^2*MM = {rho**2*mm:5.2f} nats (rho={rho:.1f})" if rho else ""))
json.dump(dict(recipe=dict(duration=DUR, final_time=FINAL, alpha_left=a.alpha_left, alpha_right=a.alpha_right), injection=inj, snr=rho, f_band=[a.f_min, a.f_max], results=res),
          open(a.out, "w"), indent=2)
print("saved", a.out)
