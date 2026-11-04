#!/usr/bin/env python
"""Freeze the GW170817 analysis inputs (FD strain + PSD) for the gwjax / bilby revision runs.

Builds the H1+L1+V1 network exactly as the published notebook
(examples/public_Parameter_Estimation_MLGW_BNS_JAX_GW170817_tc95_acceptance_walk.ipynb):
BayesWave-deglitched 4 kHz strain (BWCLEANED, 1024 s files), 128 s analysis segment with the
merger at 0.95*T, Tukey roll-off 0.2 s, f_min = 23 Hz -- except that NO time-domain filter is
applied by default (bilby/LVK convention; the published run's gwpy highpass rippled in-band). Writes

    data_<psd_mode>/{frequency_array.npy, <IFO>_strain_fd.npy, <IFO>_psd.npy, metadata.json}

with sha256 checksums; COLAB_GW170817_mlgw_bns_LVKpriors.ipynb and run_bilby_gw170817.py both read
these arrays verbatim and verify the checksums, so the two codes analyse byte-identical data.

PSD modes (referee comment 1d):
    gwtc1  (default) the GWTC-1 GW170817 PSDs released by the LVK (DCC LIGO-P1900011,
           GWTC1_GW170817_PSDs.dat): BayesWave on-source PSDs, 23-2048 Hz on a 1/128 Hz grid,
           i.e. exactly our analysis grid. This is the PSD of the GWTC-1 analysis; it whitens
           this data to 0.96-1.05 in every band up to 1600 Hz.
    welch  the published run: median Welch over the full 1024 s BWCLEANED file, 128 s Hann
           segments, 50% overlap (15 periodograms; the file contains the event).

    python freeze_gw170817_data.py                       # gwtc1 PSD, 23-1500 Hz
    python freeze_gw170817_data.py --psd-mode welch      # the paper's PSD

The frozen arrays are ~19 MB per mode and are not committed (metadata.json with the checksums
is). Needs gwgpu_jax[data] (CPU is fine), the three BWCLEANED .txt files (default
examples/data_GW170817/) and network access to dcc.ligo.org for the GWTC-1 PSD file (cached).
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import gwgpu_jax
from gwgpu_jax.gwgpu_jax_compatibility import get_event_info

HERE = Path(__file__).resolve().parent
GWTC1_PSD_URL = "https://dcc.ligo.org/public/0158/P1900011/001/GWTC1_GW170817_PSDs.dat"
GWTC1_PSD_SHA256 = "93460c6113ac594e43cd01ce91c695108aaf0031bc1e05648cc090214b23578a"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--psd-mode", choices=["gwtc1", "welch"], default="gwtc1")
ap.add_argument("--f-max", type=float, default=1500.0,
                help="likelihood upper cutoff [Hz]. 1500 (published run, default): the GWOSC 4 kHz "
                     "files carry the anti-alias low-pass of the 16->4 kHz decimation, which starts "
                     "at ~1650 Hz (whitened power 0.79 at 1650-1700 Hz, 0.36 at 1700-1750, ~0 above "
                     "1800, identical in H1/L1/V1) and acts on the signal too; GWTC-1's 2048 Hz "
                     "assumed the full-band 16 kHz data")
ap.add_argument("--bw-dir", default=str(HERE.parents[1] / "data_GW170817"),
                help="directory with the three *_BWCLEANED_4KHZ-1187008114-1024.txt files")
ap.add_argument("--gwtc1-psd", default=str(HERE / "GWTC1_GW170817_PSDs.dat"),
                help="GWTC-1 PSD file (downloaded here from the DCC if missing)")
ap.add_argument("--bandpass", choices=["none", "flat"], default="none",
                help="time-domain band limit before cropping: none (bilby/LVK convention, default) or the flat "
                     "Butterworth (23, 2048) Hz. The published run used gwpy's Chebyshev highpass, which "
                     "attenuated the in-band data by up to 37%% (fixed in gwgpu_jax Nov 2026)")
ap.add_argument("--outdir", default=None, help="default: ./data_<psd-mode> next to this script")
args = ap.parse_args()

DURATION, FS, F_MIN, F_MAX = 128.0, 4096.0, 23.0, args.f_max
BW_GPS, BW_DUR = 1187008114, 1024
DETS = ("H1", "L1", "V1")


def sha16(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ── strain: identical to the published notebook's BWCLEANED path ────────────────────────────
from gwpy.timeseries import TimeSeries

bw = Path(args.bw_dir)
ts = {}
for det in DETS:
    fp = bw / f"{det[0]}-{det}_BWCLEANED_4KHZ-{BW_GPS}-{BW_DUR}.txt"
    if not fp.exists():
        raise SystemExit(f"missing {fp} (BayesWave-cleaned GW170817 strain, see the notebook's cell 3)")
    ts[det] = TimeSeries(np.loadtxt(fp, comments="#"), sample_rate=FS, t0=BW_GPS,
                         channel=f"{det}:BWCLEANED_4KHZ", name=f"{det}_BWCLEANED")

grid = gwgpu_jax.TimeFrequencyGrid(duration=DURATION, sampling_rate=FS, f_min=F_MIN, f_max=F_MAX)
network = gwgpu_jax.Network.from_names(list(DETS), grid)
info = get_event_info("GW170817")
gwgpu_jax.compat.attach_data_to_network(
    network, ts,
    event_gps=info.gps_time, pre_merger=0.95 * DURATION,
    bandpass=None if args.bandpass == "none" else info.bandpass,
    estimate_psd=True, psd_full_data=True, psd_nperseg=int(DURATION * FS))

freqs = np.asarray(grid.frequency_domain_array)
psds = {i.name: np.asarray(i.psd) for i in network.interferometers}

# ── PSD: optionally replace by the GWTC-1 release ─────────────────────────────────────────
psd_info = dict(psd_method="median Welch over the full 1024 s BWCLEANED file, 128 s Hann segments, "
                           "50% overlap (the published run)")
if args.psd_mode == "gwtc1":
    fp = Path(args.gwtc1_psd)
    if not fp.exists():
        print("downloading", GWTC1_PSD_URL)
        urllib.request.urlretrieve(GWTC1_PSD_URL, fp)
    h = hashlib.sha256(fp.read_bytes()).hexdigest()
    if h != GWTC1_PSD_SHA256:
        raise SystemExit(f"{fp}: sha256 {h} != expected {GWTC1_PSD_SHA256}")
    tab = np.loadtxt(fp)
    f_tab = tab[:, 0]
    k = np.round(f_tab * DURATION).astype(int)
    # the file prints frequencies to 1e-6 Hz (23.007812 for 23.0078125): match bins, not floats
    assert np.allclose(freqs[k], f_tab, rtol=0, atol=1e-5) and np.all(np.diff(k) == 1), \
        "GWTC-1 PSD grid != analysis grid"
    for j, det in enumerate(DETS):
        p = np.full_like(freqs, np.inf)          # out of the GWTC-1 band: zero weight
        p[k] = tab[:, 1 + j]
        psds[det] = p
    psd_info = dict(psd_method="GWTC-1 GW170817 PSDs (LVK release, DCC LIGO-P1900011, BayesWave on-source)",
                    psd_file=fp.name, psd_file_sha256=GWTC1_PSD_SHA256, psd_url=GWTC1_PSD_URL,
                    psd_band_hz=[float(f_tab[0]), float(f_tab[-1])])

# ── write ──────────────────────────────────────────────────────────────────────────────────
epochs = {i.name: float(i.data_epoch) for i in network.interferometers}
start = epochs["H1"]
assert all(abs(e - start) < 1e-6 for e in epochs.values()), epochs
out = Path(args.outdir) if args.outdir else HERE / f"data_{args.psd_mode}"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "frequency_array.npy", freqs)
checks = {"frequency_array": sha16(freqs)}
for ifo in network.interferometers:
    d = np.asarray(ifo.strain_data_fd)
    np.save(out / f"{ifo.name}_strain_fd.npy", d)
    np.save(out / f"{ifo.name}_psd.npy", psds[ifo.name])
    checks[f"{ifo.name}_strain_fd"], checks[f"{ifo.name}_psd"] = sha16(d), sha16(psds[ifo.name])

# ── where mlgw_bns_jax puts the likelihood peak in tc (template time-origin convention) ─────
# The gwjax runs marginalise tc over TC0 +- 1 ms (TC0 = trigger position in the segment); the
# bilby run samples geocent_time over a window of the same width. TEOBResumSPA's time origin
# differs from mlgw_bns's by a fraction of a ms, so run_bilby_gw170817.py centres its window at
# (TEOB peak) - mlgw_tc_peak_offset_s, i.e. at the same place relative to the likelihood peak.
from gwgpu_jax.gwgpu_jax_relative_binning import _project_at_freqs
from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator
import jax.numpy as jnp

_gen = MLGWBNSGenerator()


def _mlgw(p, f):
    m1 = p["mc"] * (1 + p["q"]) ** 0.2 / p["q"] ** 0.6
    m2 = p["q"] * m1
    hp, hc = _gen.predict(jnp.stack([m1 / m2, p["lambda_1"], p["lambda_2"], p["chi_1"], p["chi_2"]]), f,
                          total_mass=m1 + m2, distance_mpc=p["distance"], inclination=p["inclination"])
    return hp, hc


TC0 = float(info.gps_time) - start
FID = dict(mc=1.1975, q=0.85, chi_1=0.0, chi_2=0.0, lambda_1=250.0, lambda_2=250.0, distance=37.0,
           inclination=2.545, ra=3.44616, dec=-0.408084, psi=0.0, tc=TC0, phi_c=0.0)
dts = np.linspace(-1.5e-3, 1.5e-3, 601)
z = np.zeros(len(dts), complex)
hh = 0.0
inband = np.asarray(grid.frequency_mask)
for ifo in network.interferometers:
    S = psds[ifo.name]
    w = np.where(inband & np.isfinite(S), 4.0 / DURATION / np.where(np.isfinite(S), S, 1.0), 0.0)
    h = np.asarray(_project_at_freqs(_mlgw, FID, grid.frequency_domain_array, ifo, gmst=float(network.gmst)))
    hh += float(np.sum(np.abs(h) ** 2 * w))
    a = np.conj(np.asarray(ifo.strain_data_fd)) * h * w
    sel = w > 0
    z += np.exp(-2j * np.pi * np.outer(dts, freqs[sel])) @ a[sel]
logl_tc = np.abs(z) - 0.5 * hh                       # phase-maximised log-likelihood ratio
k = int(np.argmax(logl_tc))
MLGW_TC_PEAK = float(dts[k])
print(f"mlgw_bns_jax tc peak at TC0 {1e3 * MLGW_TC_PEAK:+.3f} ms (phase-max logLR {logl_tc[k]:.2f} at the "
      f"paper's fiducial point; {logl_tc[k] - logl_tc[100]:.0f} / {logl_tc[k] - logl_tc[500]:.0f} above the "
      f"-1 / +1 ms window edges)")
if abs(MLGW_TC_PEAK) > 0.6e-3:
    print("WARNING: the mlgw tc peak is not well inside TC0 +- 1 ms; widen TC_GRID in the notebook")

meta = dict(
    event="GW170817", source="BayesWave-deglitched BWCLEANED 4 kHz strain via gwgpu_jax.compat.attach_data_to_network",
    detectors=list(DETS), duration=DURATION, sampling_frequency=FS,
    minimum_frequency=F_MIN, maximum_frequency=F_MAX, reference_frequency=F_MIN,
    trigger_time=float(info.gps_time), start_time=start, tc_in_segment=float(info.gps_time) - start,
    gmst=float(network.gmst), tukey_roll_off=0.2,
    bandpass=None if args.bandpass == "none" else [*info.bandpass, "flat Butterworth"],
    psd_mode=args.psd_mode, **psd_info,
    mlgw_tc_peak_offset_s=MLGW_TC_PEAK, sha256_16=checks)

band = np.asarray(grid.frequency_mask)
# whitening check: 2|d|^2/(T S) ~ Exp(1) for noise, so median/ln2 should be ~1 (the published
# run's filtered data gave 0.5-0.6 between 40 and 200 Hz)
WB = [(23, 40), (40, 80), (80, 200), (200, 500), (500, F_MAX)]
meta["whitened_power_median"] = {"bands_hz": WB}
for det in DETS:
    d = np.load(out / f"{det}_strain_fd.npy")
    row = []
    for lo, hi in WB:
        b = (freqs >= lo) & (freqs < hi) & np.isfinite(psds[det])
        row.append(float(np.median(2 * np.abs(d[b]) ** 2 / (DURATION * psds[det][b])) / np.log(2)))
    meta["whitened_power_median"][det] = row
    print(f"  {det} whitened power (expect ~1) in 23-40/40-80/80-200/200-500/500+ Hz: "
          + " ".join(f"{x:.2f}" for x in row))
json.dump(meta, open(out / "metadata.json", "w"), indent=2)
print(f"trigger {info.gps_time:.4f}  segment start {start:.4f}  (merger at {info.gps_time - start:.3f} s)")
for det in DETS:
    p = psds[det][band]
    print(f"  {det}: median in-band ASD = {np.sqrt(np.median(p[np.isfinite(p)])):.3e}")
print("wrote", out)
