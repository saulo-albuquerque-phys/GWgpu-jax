#!/usr/bin/env python
"""Step 1 (pyseobnr env): SEOBNRv5HM (2,2)-only TD polarizations at the bbh_case
injection, saved with t=0 at the peak of |h22| (mlgw's time convention).

    python gen_seobnrv5hm_td.py --out seob_td.npz [--f-start 10]
"""
import argparse, json
import numpy as np
from pyseobnr.generate_waveform import GenerateWaveform

ap = argparse.ArgumentParser()
ap.add_argument("--meta", default="../../../drive_data/bbh_case/data/metadata.json")
ap.add_argument("--out", default="seob_td.npz")
ap.add_argument("--f-start", type=float, default=10.0)
ap.add_argument("--srate", type=float, default=4096.0)
a = ap.parse_args()
inj = json.load(open(a.meta))["injection"]
p = dict(mass1=inj["mass_1"], mass2=inj["mass_2"], spin1x=0., spin1y=0., spin1z=inj["chi_1"],
         spin2x=0., spin2y=0., spin2z=inj["chi_2"], distance=inj["luminosity_distance"],
         inclination=inj["theta_jn"], phi_ref=inj["phase"], f22_start=a.f_start, f_ref=20.0,
         deltaT=1.0 / a.srate, approximant="SEOBNRv5HM", mode_array=[(2, 2)])
hp, hc = GenerateWaveform(p).generate_td_polarizations()
hp_d, hc_d = np.asarray(hp.data.data), np.asarray(hc.data.data)
t = float(hp.epoch) + np.arange(hp_d.size) * hp.deltaT
ipk = np.argmax(hp_d**2 + hc_d**2)
t = t - t[ipk]
np.savez(a.out, t=t, hp=hp_d, hc=hc_d, **{k: v for k, v in p.items() if not isinstance(v, list)})
print(f"SEOBNRv5HM (2,2): {hp_d.size} samples, t in [{t[0]:.2f}, {t[-1]:.3f}] s, saved {a.out}")
