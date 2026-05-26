"""
test_waveforms.py
=================
Generate and plot GW polarizations (h+, hx) using ripplegw.

Waveform: IMRPhenomD (aligned-spin BBH, frequency domain)
Output  : waveform_polarizations.png

Parameter convention (IMRPhenomD):
  theta = [Mchirp, eta, chi1, chi2, D_L, t_c, phi_c, iota]
  - Mchirp : chirp mass          [M_sun]
  - eta    : symmetric mass ratio [0, 0.25]
  - chi1   : aligned spin of primary   [-1, 1]
  - chi2   : aligned spin of secondary [-1, 1]
  - D_L    : luminosity distance  [Mpc]
  - t_c    : time of coalescence  [s]  (sets phase shift only)
  - phi_c  : phase at coalescence [rad]
  - iota   : inclination angle    [rad]

Usage:
  python test_waveforms.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ripplegw.waveforms import IMRPhenomD

# ── Parameters ────────────────────────────────────────────────────────────────
# GW150914-like binary black hole
MCHIRP   = 28.3              # chirp mass [M_sun]
ETA      = 0.247             # symmetric mass ratio  (q ~ 0.82)
CHI1     = 0.32              # primary aligned spin
CHI2     = -0.44             # secondary aligned spin
D_L      = 410.0             # luminosity distance [Mpc]
T_C      = 0.0               # coalescence time [s]
PHI_C    = 0.0               # coalescence phase [rad]
IOTA     = jnp.pi / 3        # inclination [rad]  (60 degrees)

F_LOW    = 20.0        # starting frequency [Hz]
F_HIGH   = 1024.0      # maximum frequency [Hz]
F_REF    = 20.0        # reference frequency [Hz]
N_FREQS  = 4096        # number of frequency points

# ── Frequency grid ────────────────────────────────────────────────────────────
freqs = jnp.linspace(F_LOW, F_HIGH, N_FREQS)
df    = float(freqs[1] - freqs[0])

theta = jnp.array([MCHIRP, ETA, CHI1, CHI2, D_L, T_C, PHI_C, IOTA])

# ── Generate polarizations ────────────────────────────────────────────────────
hp_fd, hc_fd = IMRPhenomD.gen_IMRPhenomD_hphc(freqs, theta, F_REF)

# ── Time-domain via iFFT ──────────────────────────────────────────────────────
# ripplegw returns a one-sided spectrum from F_LOW to F_HIGH.
# We zero-pad from DC (0 Hz) to F_LOW so that jnp.fft.irfft sees a complete
# one-sided spectrum starting at 0 Hz.
n_zero = round(F_LOW / df)                          # bins from DC to F_LOW
hp_ext = jnp.concatenate([jnp.zeros(n_zero, dtype=hp_fd.dtype), hp_fd])
hc_ext = jnp.concatenate([jnp.zeros(n_zero, dtype=hc_fd.dtype), hc_fd])

hp_td  = jnp.fft.irfft(hp_ext)                     # real time series
hc_td  = jnp.fft.irfft(hc_ext)

n_td   = len(hp_td)
dt     = 1.0 / (2.0 * (len(hp_ext) - 1) * df)     # sample spacing [s]
times  = jnp.arange(n_td) * dt - n_td * dt          # shifted so t=0 at end

# Keep only the final 4 s leading up to coalescence
mask = (times >= -4.0) & (times <= 0.1)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 14))
fig.suptitle(
    f"IMRPhenomD  |  $\\mathcal{{M}}={MCHIRP}\\,M_\\odot$,  "
    f"$\\eta={ETA}$,  $D_L={D_L}\\,\\mathrm{{Mpc}}$,  "
    f"$\\chi_1={CHI1}$,  $\\chi_2={CHI2}$,  $\\iota={float(jnp.degrees(IOTA)):.0f}°$",
    fontsize=11,
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

# ── Time-domain strain (top row, full width) ──────────────────────────────────
ax_td = fig.add_subplot(gs[0, :])
ax_td.plot(times[mask], hp_td[mask], color="C0", lw=1.0, label=r"$h_+$")
ax_td.plot(times[mask], hc_td[mask], color="C1", lw=1.0, ls="--", label=r"$h_\times$")
ax_td.set_xlabel("Time to coalescence [s]")
ax_td.set_ylabel("Strain $h(t)$")
ax_td.set_title("Time-domain polarizations")
ax_td.legend(frameon=False)
ax_td.grid(alpha=0.3)

# ── Frequency-domain waveform: Re and Im ─────────────────────────────────────
ax_re = fig.add_subplot(gs[1, 0])
ax_re.plot(freqs, jnp.real(hp_fd), color="C0",      lw=0.7, label=r"$\mathrm{Re}[\tilde{h}_+]$")
ax_re.plot(freqs, jnp.imag(hp_fd), color="C0",      lw=0.7, ls="--", alpha=0.6, label=r"$\mathrm{Im}[\tilde{h}_+]$")
ax_re.set_xlabel("Frequency [Hz]")
ax_re.set_ylabel(r"$\tilde{h}_+(f)$")
ax_re.set_title(r"FD waveform  $h_+$")
ax_re.legend(frameon=False, fontsize=8)
ax_re.grid(alpha=0.3)

ax_im = fig.add_subplot(gs[1, 1])
ax_im.plot(freqs, jnp.real(hc_fd), color="C1",      lw=0.7, label=r"$\mathrm{Re}[\tilde{h}_\times]$")
ax_im.plot(freqs, jnp.imag(hc_fd), color="C1",      lw=0.7, ls="--", alpha=0.6, label=r"$\mathrm{Im}[\tilde{h}_\times]$")
ax_im.set_xlabel("Frequency [Hz]")
ax_im.set_ylabel(r"$\tilde{h}_\times(f)$")
ax_im.set_title(r"FD waveform  $h_\times$")
ax_im.legend(frameon=False, fontsize=8)
ax_im.grid(alpha=0.3)

# ── Frequency-domain amplitude ────────────────────────────────────────────────
ax_amp = fig.add_subplot(gs[2, 0])
ax_amp.loglog(freqs, jnp.abs(hp_fd), color="C0", label=r"$h_+$")
ax_amp.loglog(freqs, jnp.abs(hc_fd), color="C1", ls="--", label=r"$h_\times$")
ax_amp.set_xlabel("Frequency [Hz]")
ax_amp.set_ylabel(r"$|\tilde{h}(f)|$")
ax_amp.set_title("FD amplitude")
ax_amp.legend(frameon=False)
ax_amp.grid(alpha=0.3, which="both")

# ── Frequency-domain phase ────────────────────────────────────────────────────
ax_ph = fig.add_subplot(gs[2, 1])
ax_ph.semilogx(freqs, jnp.unwrap(jnp.angle(hp_fd)), color="C0", label=r"$h_+$")
ax_ph.semilogx(freqs, jnp.unwrap(jnp.angle(hc_fd)), color="C1", ls="--", label=r"$h_\times$")
ax_ph.set_xlabel("Frequency [Hz]")
ax_ph.set_ylabel("Unwrapped phase [rad]")
ax_ph.set_title("FD phase")
ax_ph.legend(frameon=False)
ax_ph.grid(alpha=0.3, which="both")

# ── Phase difference h+ − hx (should be π/2 for a circular orbit) ────────────
ax_diff = fig.add_subplot(gs[3, :])
phase_diff = jnp.unwrap(jnp.angle(hp_fd)) - jnp.unwrap(jnp.angle(hc_fd))
ax_diff.semilogx(freqs, phase_diff, color="C3", lw=1.0)
ax_diff.axhline(-jnp.pi / 2, color="k", ls=":", lw=0.8, label=r"$-\pi/2$ (face-on)")
ax_diff.set_xlabel("Frequency [Hz]")
ax_diff.set_ylabel(r"$\Phi_+ - \Phi_\times$ [rad]")
ax_diff.set_title("Phase difference between polarizations")
ax_diff.legend(frameon=False)
ax_diff.grid(alpha=0.3, which="both")

out = "waveform_polarizations.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
