#!/usr/bin/env python
"""Freeze the GW150914 analysis inputs (FD strain + PSD) exactly as the gwjax notebook builds them.

Writes data_<psd_mode>/{frequency_array.npy, <IFO>_strain_fd.npy, <IFO>_psd.npy, metadata.json}
with sha256 checksums. run_bilby_gw150914.py reads these arrays verbatim, so bilby and
gwjax (COLAB_GW150914_mlgw_bbh_LVKpriors.ipynb, same settings) analyse identical data:
GWOSC 4 s @ 4096 Hz, 20–1024 Hz, H1+L1, Tukey roll-off 0.2 s, off-source median Welch PSD.

    python freeze_gw150914_data.py --psd-mode lvk_welch     # 128 s, 4 s segments (default)
    python freeze_gw150914_data.py --psd-mode paper         # 32 s, the published run

Needs gwgpu_jax[data] (CPU is fine) and network access to GWOSC.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import gwgpu_jax
from gwgpu_jax.gwgpu_jax_compatibility import get_event_info

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--psd-mode", choices=["lvk_welch", "paper"], default="lvk_welch")
ap.add_argument("--outdir", default=None, help="default: ./data_<psd-mode> next to this script")
args = ap.parse_args()

DURATION, FS, F_MIN, F_MAX = 4.0, 4096.0, 20.0, 1024.0
PSD_OFFSET = 8.0
PSD_SEGMENT_DURATION = {"lvk_welch": 128.0, "paper": 32.0}[args.psd_mode]

grid = gwgpu_jax.TimeFrequencyGrid(duration=DURATION, sampling_rate=FS, f_min=F_MIN, f_max=F_MAX)
PSD_NPERSEG = {"lvk_welch": int(grid.n_samples), "paper": None}[args.psd_mode]
network = gwgpu_jax.Network.from_names(["H1", "L1"], grid)
gwgpu_jax.compat.attach_event_to_network(
    network, "GW150914", estimate_psd=True,
    psd_segment_duration=PSD_SEGMENT_DURATION, psd_offset=PSD_OFFSET,
    psd_nperseg=PSD_NPERSEG, verbose=False)

trigger = float(get_event_info("GW150914").gps_time)
epochs = {i.name: float(i.data_epoch) for i in network.interferometers}
start = epochs["H1"]
assert all(abs(e - start) < 1e-6 for e in epochs.values()), epochs

out = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent / f"data_{args.psd_mode}"
out.mkdir(parents=True, exist_ok=True)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


freqs = np.asarray(grid.frequency_domain_array)
np.save(out / "frequency_array.npy", freqs)
checks = {"frequency_array": sha(freqs)}
for ifo in network.interferometers:
    d, p = np.asarray(ifo.strain_data_fd), np.asarray(ifo.psd)
    np.save(out / f"{ifo.name}_strain_fd.npy", d)
    np.save(out / f"{ifo.name}_psd.npy", p)
    checks[f"{ifo.name}_strain_fd"], checks[f"{ifo.name}_psd"] = sha(d), sha(p)

json.dump(dict(
    event="GW150914", source="GWOSC via gwgpu_jax.compat.attach_event_to_network",
    detectors=[i.name for i in network.interferometers],
    duration=DURATION, sampling_frequency=FS, minimum_frequency=F_MIN, maximum_frequency=F_MAX,
    reference_frequency=20.0, trigger_time=trigger, start_time=start,
    trigger_in_segment=trigger - start, gmst=float(network.gmst),
    psd_mode=args.psd_mode, psd_segment_duration=PSD_SEGMENT_DURATION, psd_offset=PSD_OFFSET,
    psd_nperseg=PSD_NPERSEG, psd_method="scipy.signal.welch, hann, 50% overlap, median",
    tukey_roll_off=0.2, sha256_16=checks,
), open(out / "metadata.json", "w"), indent=2)
print(f"trigger {trigger:.4f}  segment start {start:.4f}  (trigger at {trigger - start:.3f} s)")
for ifo in network.interferometers:
    band = np.asarray(grid.frequency_mask)
    print(f"  {ifo.name}: median in-band ASD = {np.sqrt(np.median(np.asarray(ifo.psd)[band])):.3e}")
print("wrote", out)
