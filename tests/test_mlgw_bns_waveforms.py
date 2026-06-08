"""
test_mlgw_bns_waveforms.py
==========================
Generate and plot BNS waveforms from the mlgw_bns JAX model.

Waveform: mlgw_bns (ML surrogate trained on TEOBResumS)
Model   : gwgpu_jax/mlgw_jax/mlgw_bns_jax/mlgw_bns_jax_model.h5
Output  : mlgw_bns_waveforms.png

Parameter convention:
  q        : mass ratio m1/m2 >= 1
  lambda_1 : tidal deformability of heavier star   [0, 5000]
  lambda_2 : tidal deformability of lighter star   [0, 5000]
  chi_1    : aligned spin of heavier star          [-0.3, 0.3]
  chi_2    : aligned spin of lighter star          [-0.3, 0.3]
  total_mass  : total mass  [M_sun]
  distance    : luminosity distance  [Mpc]
  inclination : inclination angle  [rad]

Usage:
  python test_mlgw_bns_waveforms.py
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator

# ── Frequency grid ────────────────────────────────────────────────────────────
F_MIN   = 20.0     # Hz
F_MAX   = 1024.0   # Hz
N_FREQS = 4096
freqs   = jnp.linspace(F_MIN, F_MAX, N_FREQS)
df      = float(freqs[1] - freqs[0])

# ── Load generator (JIT compiles on first call) ───────────────────────────────
print("Loading mlgw_bns JAX model...")
gen = MLGWBNSGenerator()

# ── Reference BNS parameters ─────────────────────────────────────────────────
REF = dict(
    q          = 1.2,
    lambda_1   = 400.0,
    lambda_2   = 300.0,
    chi_1      = 0.02,
    chi_2      = -0.01,
    total_mass = 2.8,
    distance   = 100.0,
    inclination= 0.0,      # face-on → maximum signal
)

# ── Single waveform ───────────────────────────────────────────────────────────
print("Generating reference waveform (first call includes JIT compile)...")
t0 = time.perf_counter()
hp_fd, hc_fd = gen.get_hphc(freqs, **REF)
hp_fd.block_until_ready()
print(f"  Done in {time.perf_counter()-t0:.2f} s  (includes JIT compile)")

# warm-up done — time a second call
t0 = time.perf_counter()
hp_fd, hc_fd = gen.get_hphc(freqs, **REF)
hp_fd.block_until_ready()
print(f"  Second call: {(time.perf_counter()-t0)*1e3:.1f} ms")

# ── Batch: vary tidal deformability ─────────────────────────────────────────
lambdas = jnp.array([100., 400., 800., 1500., 3000.])
batch_params = jnp.stack([
    jnp.ones(len(lambdas)) * REF["q"],
    lambdas,
    lambdas,
    jnp.ones(len(lambdas)) * REF["chi_1"],
    jnp.ones(len(lambdas)) * REF["chi_2"],
], axis=1)

print(f"\nGenerating batch of {len(lambdas)} waveforms via vmap...")
t0 = time.perf_counter()
hp_batch, hc_batch = gen.get_hphc_batch(
    freqs, batch_params,
    total_mass = REF["total_mass"],
    distance   = REF["distance"],
    inclination= REF["inclination"],
)
hp_batch.block_until_ready()
print(f"  Done in {(time.perf_counter()-t0)*1e3:.1f} ms  "
      f"({len(lambdas)} waveforms × {N_FREQS} freq bins)")

# ── Time domain (iFFT) ────────────────────────────────────────────────────────
n_zero  = round(F_MIN / df)
hp_ext  = jnp.concatenate([jnp.zeros(n_zero, dtype=hp_fd.dtype), hp_fd])
hc_ext  = jnp.concatenate([jnp.zeros(n_zero, dtype=hc_fd.dtype), hc_fd])
hp_td   = jnp.fft.irfft(hp_ext)
hc_td   = jnp.fft.irfft(hc_ext)
n_td    = len(hp_td)
dt      = 1.0 / (2.0 * (len(hp_ext) - 1) * df)
times   = jnp.arange(n_td) * dt - n_td * dt
mask_td = (times >= -8.0) & (times <= 0.1)

# ── Plot ──────────────────────────────────────────────────────────────────────
cmap   = plt.cm.viridis
colors = [cmap(i / (len(lambdas) - 1)) for i in range(len(lambdas))]

fig = plt.figure(figsize=(13, 14))
fig.suptitle(
    f"mlgw_bns JAX  |  $q={REF['q']}$,  "
    f"$M_{{\\rm tot}}={REF['total_mass']}\\,M_\\odot$,  "
    f"$D_L={REF['distance']}\\,{{\\rm Mpc}}$,  "
    f"$\\iota={REF['inclination']:.0f}$",
    fontsize=11,
)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

# ── Time domain ──────────────────────────────────────────────────────────────
ax_td = fig.add_subplot(gs[0, :])
ax_td.plot(times[mask_td], hp_td[mask_td], color="C0", lw=1.0, label=r"$h_+$")
ax_td.plot(times[mask_td], hc_td[mask_td], color="C1", lw=1.0, ls="--", label=r"$h_\times$")
ax_td.set_xlabel("Time to coalescence [s]")
ax_td.set_ylabel("Strain")
ax_td.set_title("Time-domain polarizations (reference BNS)")
ax_td.legend(frameon=False)
ax_td.grid(alpha=0.3)

# ── FD waveform: Re + Im ─────────────────────────────────────────────────────
ax_re = fig.add_subplot(gs[1, 0])
ax_re.plot(freqs, jnp.real(hp_fd), color="C0", lw=0.7,
           label=r"$\mathrm{Re}[\tilde{h}_+]$")
ax_re.plot(freqs, jnp.imag(hp_fd), color="C0", lw=0.7, ls="--", alpha=0.6,
           label=r"$\mathrm{Im}[\tilde{h}_+]$")
ax_re.set_xlabel("Frequency [Hz]")
ax_re.set_ylabel(r"$\tilde{h}_+(f)$")
ax_re.set_title(r"FD waveform  $h_+$")
ax_re.legend(frameon=False, fontsize=8)
ax_re.grid(alpha=0.3)

ax_im = fig.add_subplot(gs[1, 1])
ax_im.plot(freqs, jnp.real(hc_fd), color="C1", lw=0.7,
           label=r"$\mathrm{Re}[\tilde{h}_\times]$")
ax_im.plot(freqs, jnp.imag(hc_fd), color="C1", lw=0.7, ls="--", alpha=0.6,
           label=r"$\mathrm{Im}[\tilde{h}_\times]$")
ax_im.set_xlabel("Frequency [Hz]")
ax_im.set_ylabel(r"$\tilde{h}_\times(f)$")
ax_im.set_title(r"FD waveform  $h_\times$")
ax_im.legend(frameon=False, fontsize=8)
ax_im.grid(alpha=0.3)

# ── FD amplitude ─────────────────────────────────────────────────────────────
ax_amp = fig.add_subplot(gs[2, 0])
ax_amp.loglog(freqs, jnp.abs(hp_fd), color="C0", label=r"$h_+$")
ax_amp.loglog(freqs, jnp.abs(hc_fd), color="C1", ls="--", label=r"$h_\times$")
ax_amp.set_xlabel("Frequency [Hz]")
ax_amp.set_ylabel(r"$|\tilde{h}(f)|$")
ax_amp.set_title("FD amplitude")
ax_amp.legend(frameon=False)
ax_amp.grid(alpha=0.3, which="both")

# ── FD phase ─────────────────────────────────────────────────────────────────
ax_ph = fig.add_subplot(gs[2, 1])
ax_ph.semilogx(freqs, jnp.unwrap(jnp.angle(hp_fd)), color="C0", label=r"$h_+$")
ax_ph.semilogx(freqs, jnp.unwrap(jnp.angle(hc_fd)), color="C1", ls="--", label=r"$h_\times$")
ax_ph.set_xlabel("Frequency [Hz]")
ax_ph.set_ylabel("Unwrapped phase [rad]")
ax_ph.set_title("FD phase")
ax_ph.legend(frameon=False)
ax_ph.grid(alpha=0.3, which="both")

# ── Batch: amplitude vs tidal deformability ───────────────────────────────────
ax_lam = fig.add_subplot(gs[3, :])
for i, lam in enumerate(lambdas):
    ax_lam.loglog(freqs, jnp.abs(hp_batch[i]),
                  color=colors[i], lw=1.1,
                  label=rf"$\Lambda={float(lam):.0f}$")
ax_lam.set_xlabel("Frequency [Hz]")
ax_lam.set_ylabel(r"$|\tilde{h}_+(f)|$")
ax_lam.set_title(r"Effect of tidal deformability $\Lambda$ on FD amplitude")
ax_lam.legend(frameon=False, fontsize=8, ncol=len(lambdas))
ax_lam.grid(alpha=0.3, which="both")

out = "mlgw_bns_waveforms.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.show()
