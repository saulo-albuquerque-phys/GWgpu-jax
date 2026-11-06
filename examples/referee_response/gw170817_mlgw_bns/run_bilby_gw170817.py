#!/usr/bin/env python
"""bilby + dynesty (acceptance-walk) PE of GW170817 with TEOBResumSPA -- the CPU reference for
COLAB_GW170817_mlgw_bns_LVKpriors.ipynb (referee comments 1b, 1c, 6).

Same frozen inputs as gwjax (data_<psd_mode>/ from freeze_gw170817_data.py, checksum-verified),
same likelihood and the same priors:

  waveform    TEOBResumSPA (2,2), via the `teobresums` package (teobresumspa_source.py) -- the model
              mlgw_bns_jax's surrogate is trained on (mlgw_bns_jax reproduces it to overlap 0.999999)
  likelihood  relative binning (bilby RelativeBinningGravitationalWaveTransient, epsilon 0.01, ~3000
              bins: |dlnL| <~ 0.005 vs the full likelihood in the posterior region), phase
              marginalised analytically; geocent_time SAMPLED over a 20 ms window (gwjax marginalises
              tc numerically over the same width -- identical target; bilby's RB time
              marginalisation would need the full-grid waveform at every call). Fiducial point = the
              gwjax notebook's. No distance or calibration marginalisation (as gwjax).
  priors      chirp_mass   UniformInComponentsChirpMass(1.18, 1.21)       (detector frame)
              mass_ratio   UniformInComponentsMassRatio(0.34, 1)          (surrogate: m1/m2 <= 3)
              chi_1, chi_2 AlignedSpin(|a| <= 0.05) low-spin [--spin-prior high: |a| <= 0.5, the
                           surrogate limit]
              lambda_1,2   Uniform(5, 5000)                               (GWTC-1: 0-5000)
              luminosity_distance PowerLaw(2, 1, 75)
              theta_jn / dec / ra / psi   Sine / Cosine / U(0,2pi) / U(0,pi); phase marginalised
              geocent_time U(c - 10 ms, c + 10 ms), c = TEOB likelihood peak - mlgw_tc_peak_offset_s,
                           i.e. the window sits where gwjax's TC0 +- 10 ms sits relative to its peak.
                           (A +-1 ms window truncated the posterior: tc is correlated with lambda_tilde.)
  sampler     dynesty acceptance-walk, nlive 1000, naccept 60, maxmcmc 5000, dlogz 0.1 (= gwjax)

    python run_bilby_gw170817.py --npool 32                # LDG (run_bilby_gw170817.sub)
    python run_bilby_gw170817.py --benchmark 200           # timing only: RB and full-likelihood cost
    python run_bilby_gw170817.py --nlive 50 --naccept 5 --outdir smoke --npool 4   # smoke test

Checkpoints every 30 min: rerun the same command to resume.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--psd-mode", choices=["gwtc1", "welch"], default="gwtc1")
ap.add_argument("--data", default=None, help="default: ./data_<psd-mode> next to this script")
ap.add_argument("--spin-prior", choices=["low", "high"], default="low")
ap.add_argument("--outdir", default=None, help="default: bilby_gw170817_TEOBResumSPA_<psd-mode>_<spin>spin")
ap.add_argument("--label", default="gw170817_TEOBResumSPA")
ap.add_argument("--npool", type=int, default=16)
ap.add_argument("--nlive", type=int, default=1000)
ap.add_argument("--naccept", type=int, default=60)
ap.add_argument("--maxmcmc", type=int, default=5000)
ap.add_argument("--epsilon", type=float, default=0.01, help="RB bin tolerance (bilby default 0.5 is too coarse here)")
ap.add_argument("--seed", type=int, default=170817)
ap.add_argument("--tc-half-width", type=float, default=0.01,
                help="geocent_time window half-width [s] (= gwjax TC_HALF_WIDTH). 10 ms: the first run with 1 ms "
                     "truncated the tc posterior, which is correlated with lambda_tilde (r = -0.6)")
ap.add_argument("--benchmark", type=int, default=0, metavar="N",
                help="time N RB and N/10 full-likelihood evaluations at prior draws, write timing json, exit")
args = ap.parse_args()

TC_HALF_WIDTH = args.tc_half_width    # = gwjax TC_GRID half-width
SPIN_MAX = {"low": 0.05, "high": 0.5}[args.spin_prior]


def main():
    import bilby
    from bilby.core.prior import Uniform, Sine, Cosine, PowerLaw, Constraint, PriorDict
    from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio, AlignedSpin
    from teobresumspa_source import teobresumspa_bns, teobresumspa_bns_relative_binning

    here = Path(__file__).resolve().parent
    data = Path(args.data) if args.data else here / f"data_{args.psd_mode}"
    outdir = args.outdir or f"bilby_gw170817_TEOBResumSPA_{args.psd_mode}_{args.spin_prior}spin"
    meta = json.load(open(data / "metadata.json"))

    def sha(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

    arrays = {"frequency_array": np.load(data / "frequency_array.npy")}
    for det in meta["detectors"]:
        arrays[f"{det}_strain_fd"] = np.load(data / f"{det}_strain_fd.npy")
        arrays[f"{det}_psd"] = np.load(data / f"{det}_psd.npy")
    for k, a in arrays.items():
        if sha(a) != meta["sha256_16"][k]:
            raise RuntimeError(f"checksum mismatch for {k} in {data} -- re-run freeze_gw170817_data.py")
    print(f"frozen GW170817 inputs verified ({data.name}, PSD mode {meta['psd_mode']})")

    bilby.core.utils.setup_logger(outdir=outdir, label=args.label)
    print(f"bilby {bilby.__version__}")
    bilby.core.utils.random.seed(args.seed)
    T, fs = meta["duration"], meta["sampling_frequency"]
    f_min, f_max = meta["minimum_frequency"], meta["maximum_frequency"]
    start, trigger = meta["start_time"], meta["trigger_time"]
    freqs = arrays["frequency_array"]

    ifos = bilby.gw.detector.InterferometerList([])
    for det in meta["detectors"]:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        psd = arrays[f"{det}_psd"].copy()
        psd[~np.isfinite(psd)] = 1.0          # out-of-band bins only (masked by min/max frequency)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=freqs, psd_array=psd)
        ifo.set_strain_data_from_frequency_domain_strain(
            arrays[f"{det}_strain_fd"], sampling_frequency=fs, duration=T, start_time=start)
        ifo.minimum_frequency, ifo.maximum_frequency = f_min, f_max
        ifos.append(ifo)

    conv = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
    wargs = dict(minimum_frequency=f_min, maximum_frequency=f_max)
    wfg_full = bilby.gw.WaveformGenerator(
        duration=T, sampling_frequency=fs, frequency_domain_source_model=teobresumspa_bns,
        parameter_conversion=conv, waveform_arguments=wargs)
    phase_prior = lambda: PriorDict(dict(phase=Uniform(0, 2 * np.pi, name="phase", boundary="periodic")))
    full = bilby.gw.likelihood.GravitationalWaveTransient(ifos, wfg_full, phase_marginalization=True,
                                                          priors=phase_prior())

    def logl(lk, p):
        lk.parameters.update(p)
        return float(lk.log_likelihood_ratio())

    # ── fiducial point (the gwjax notebook's) and the tc window ───────────────────────────────
    fid = dict(chirp_mass=1.1975, mass_ratio=0.85, chi_1=0.0, chi_2=0.0, lambda_1=250.0, lambda_2=250.0,
               luminosity_distance=37.0, theta_jn=2.545, ra=3.44616, dec=-0.408084, psi=0.0, phase=0.0,
               geocent_time=trigger)
    t0 = time.perf_counter()
    ts = trigger + np.linspace(-1.5e-3, 1.5e-3, 31)
    ll = [logl(full, dict(fid, geocent_time=t)) for t in ts]
    ts2 = ts[int(np.argmax(ll))] + np.linspace(-1e-4, 1e-4, 41)
    ll2 = [logl(full, dict(fid, geocent_time=t)) for t in ts2]
    t_peak = float(ts2[int(np.argmax(ll2))])
    tc_center = t_peak - meta["mlgw_tc_peak_offset_s"]
    print(f"TEOBResumSPA tc peak at trigger {1e3 * (t_peak - trigger):+.3f} ms (logLR {max(ll2):.2f}); "
          f"mlgw peak offset {1e3 * meta['mlgw_tc_peak_offset_s']:+.3f} ms -> window centre trigger "
          f"{1e3 * (tc_center - trigger):+.3f} ms +- {1e3 * TC_HALF_WIDTH:.1f} ms  ({time.perf_counter() - t0:.0f} s)")
    fid["geocent_time"] = t_peak

    # ── priors ────────────────────────────────────────────────────────────────────────────────
    pri = bilby.gw.prior.BNSPriorDict(aligned_spin=True)
    pri["chirp_mass"] = UniformInComponentsChirpMass(1.18, 1.21, name="chirp_mass")
    pri["mass_ratio"] = UniformInComponentsMassRatio(0.34, 1.0, name="mass_ratio")
    pri["mass_1"] = Constraint(0.5, 5.0, name="mass_1")
    pri["mass_2"] = Constraint(0.5, 5.0, name="mass_2")
    pri["chi_1"] = AlignedSpin(a_prior=Uniform(0.0, SPIN_MAX), z_prior=Uniform(-1.0, 1.0), name="chi_1")
    pri["chi_2"] = AlignedSpin(a_prior=Uniform(0.0, SPIN_MAX), z_prior=Uniform(-1.0, 1.0), name="chi_2")
    pri["lambda_1"] = Uniform(5.0, 5000.0, name="lambda_1")
    pri["lambda_2"] = Uniform(5.0, 5000.0, name="lambda_2")
    pri["luminosity_distance"] = PowerLaw(alpha=2.0, minimum=1.0, maximum=75.0, name="luminosity_distance")
    pri["theta_jn"] = Sine(name="theta_jn")
    pri["dec"] = Cosine(name="dec")
    pri["ra"] = Uniform(0.0, 2 * np.pi, name="ra", boundary="periodic")
    pri["psi"] = Uniform(0.0, np.pi, name="psi", boundary="periodic")
    pri["phase"] = Uniform(0.0, 2 * np.pi, name="phase", boundary="periodic")
    pri["geocent_time"] = Uniform(tc_center - TC_HALF_WIDTH, tc_center + TC_HALF_WIDTH, name="geocent_time")

    wfg_rb = bilby.gw.WaveformGenerator(
        duration=T, sampling_frequency=fs, frequency_domain_source_model=teobresumspa_bns_relative_binning,
        parameter_conversion=conv, waveform_arguments=dict(wargs))
    like = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
        ifos, wfg_rb, fiducial_parameters=fid, priors=pri, epsilon=args.epsilon,
        phase_marginalization=True, update_fiducial_parameters=False)
    print(f"RB: {len(like.bin_freqs)} bins (epsilon {args.epsilon})")

    if args.benchmark:
        rng_draws = pri.sample(args.benchmark)
        draws = [{k: v[i] for k, v in rng_draws.items()} for i in range(args.benchmark)]
        tic = time.perf_counter()
        for p in draws:
            logl(like, p)
        t_rb = (time.perf_counter() - tic) / len(draws)
        nf = max(1, args.benchmark // 10)
        tic = time.perf_counter()
        for p in draws[:nf]:
            logl(full, p)
        t_full = (time.perf_counter() - tic) / nf
        # accuracy where it matters: small offsets from the fiducial
        offs = [dict(chirp_mass=1.19755), dict(chirp_mass=1.1974, mass_ratio=0.75), dict(lambda_1=300.0, lambda_2=900.0),
                dict(lambda_1=1500.0, lambda_2=2500.0, geocent_time=t_peak - 3e-3),
                dict(geocent_time=t_peak - 3e-3), dict(geocent_time=t_peak + 3e-3),
                dict(chi_1=0.02, chi_2=-0.02), dict(luminosity_distance=30.0, theta_jn=2.2), dict(ra=3.42, dec=-0.38, psi=1.0)]
        acc = [logl(like, dict(fid, **o)) - logl(full, dict(fid, **o)) for o in offs]
        res = dict(rb_s_per_eval=t_rb, full_s_per_eval=t_full, n_bins=len(like.bin_freqs), epsilon=args.epsilon,
                   rb_minus_full_lnL_near_fiducial=acc, threads="1 (single core, bilby per-process cost)",
                   host=__import__("socket").gethostname())
        Path(outdir).mkdir(parents=True, exist_ok=True)
        json.dump(res, open(Path(outdir) / "timing_benchmark.json", "w"), indent=2)
        print(json.dumps(res, indent=2))
        return

    result = bilby.run_sampler(
        like, pri, sampler="dynesty", outdir=outdir, label=args.label,
        nlive=args.nlive, naccept=args.naccept, maxmcmc=args.maxmcmc, sample="acceptance-walk", dlogz=0.1,
        npool=args.npool, check_point_delta_t=1800, check_point_plot=False, plot=False,
        print_method="interval-60",
        conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
        meta_data=dict(frozen_data=str(data), psd_mode=meta["psd_mode"], sha256_16=meta["sha256_16"],
                       start_time=start, trigger_time=trigger, tc_window_center=tc_center,
                       tc_half_width=TC_HALF_WIDTH, teob_tc_peak=t_peak, spin_prior=args.spin_prior,
                       rb_epsilon=args.epsilon, rb_bins=len(like.bin_freqs)))
    print(f"\nlog Z = {result.log_evidence:.3f} +/- {result.log_evidence_err:.3f}   "
          f"log BF = {result.log_bayes_factor:.3f}   sampling time {result.sampling_time}")
    print(f"result: {Path(outdir) / (args.label + '_result.json')}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if sys.platform == "darwin":
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
    main()
