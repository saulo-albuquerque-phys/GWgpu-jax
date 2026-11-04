#!/usr/bin/env python
"""bilby + dynesty (acceptance-walk) PE of GW150914 with LAL SEOBNRv5_ROM — the CPU reference
for COLAB_GW150914_mlgw_bbh_LVKpriors.ipynb (referee comment 1b).

Same frozen inputs as gwjax (data_<psd_mode>/ from freeze_gw150914_data.py, checksum-verified),
same likelihood (no calibration marginalisation), and the same priors expressed with bilby's
own classes:

    chirp_mass           UniformInComponentsChirpMass(15, 50)
    mass_ratio           UniformInComponentsMassRatio(0.1, 1)
    chi_1, chi_2         AlignedSpin(a_prior=Uniform(0, 0.9))        (isotropic, |a| <= 0.9)
    luminosity_distance  PowerLaw(alpha=2, 50, 2000)
    theta_jn / dec / ra / psi / phase   Sine / Cosine / U(0,2pi) / U(0,pi) / U(0,2pi)
    geocent_time         trigger +- 0.1 s

SEOBNRv5_ROM is the (2,2)-mode, aligned-spin LAL ROM of SEOBNRv5 — the model mlgw_bbh_jax's
(2,2) mode is a surrogate of. Sampler: dynesty acceptance-walk, naccept 60, dlogz 0.1.

    python run_bilby_gw150914.py --npool 16                    # reference run (LDG recommended)
    python run_bilby_gw150914.py --npool 8 --marginalize       # faster: distance+time+phase marg.
    python run_bilby_gw150914.py --npool 4 --nlive 60 --naccept 5 --approximant IMRPhenomXAS --outdir smoke
                                    # smoke test (~7 min on 4 cores; XAS needs no ROM data)

--marginalize uses bilby's distance, time and phase marginalisation and reconstructs those
parameters in post-processing (valid for the (2,2)-only model); the posterior is the same in
expectation, the run is several times cheaper. Checkpoints every 30 min: rerun to resume.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--psd-mode", choices=["lvk_welch", "paper"], default="lvk_welch")
ap.add_argument("--data", default=None, help="default: ./data_<psd-mode> next to this script")
ap.add_argument("--outdir", default=None, help="default: bilby_gw150914_v5ROM_<psd-mode>[_marg]")
ap.add_argument("--label", default=None, help="default: gw150914_<approximant>")
ap.add_argument("--npool", type=int, default=16)
ap.add_argument("--nlive", type=int, default=1000)
ap.add_argument("--naccept", type=int, default=60)
ap.add_argument("--marginalize", action="store_true", help="distance + time + phase marginalisation")
ap.add_argument("--approximant", default="SEOBNRv5_ROM")
ap.add_argument("--seed", type=int, default=150914)
ap.add_argument("--maxiter", type=int, default=None,
                help="NOTE: not honoured by bilby 2.8's dynesty wrapper; use a small --nlive for smoke tests")
args = ap.parse_args()


def main():
    import bilby
    from bilby.core.prior import Uniform, Sine, Cosine, PowerLaw, Constraint
    from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio, AlignedSpin

    here = Path(__file__).resolve().parent
    data = Path(args.data) if args.data else here / f"data_{args.psd_mode}"
    tag = args.approximant.replace("_", "")
    outdir = args.outdir or f"bilby_gw150914_{tag}_{args.psd_mode}{'_marg' if args.marginalize else ''}"
    args.label = args.label or f"gw150914_{tag}"
    meta = json.load(open(data / "metadata.json"))

    def sha(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

    arrays = {"frequency_array": np.load(data / "frequency_array.npy")}
    for det in meta["detectors"]:
        arrays[f"{det}_strain_fd"] = np.load(data / f"{det}_strain_fd.npy")
        arrays[f"{det}_psd"] = np.load(data / f"{det}_psd.npy")
    for k, a in arrays.items():
        if sha(a) != meta["sha256_16"][k]:
            raise RuntimeError(f"checksum mismatch for {k} in {data} — re-run freeze_gw150914_data.py")
    print(f"frozen GW150914 inputs verified ({data.name}, PSD mode {meta['psd_mode']})")

    bilby.core.utils.setup_logger(outdir=outdir, label=args.label)
    bilby.core.utils.random.seed(args.seed)
    T, fs = meta["duration"], meta["sampling_frequency"]
    f_min, f_max, f_ref = meta["minimum_frequency"], meta["maximum_frequency"], meta["reference_frequency"]
    start, trigger = meta["start_time"], meta["trigger_time"]
    freqs = arrays["frequency_array"]

    ifos = bilby.gw.detector.InterferometerList([])
    for det in meta["detectors"]:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        psd = arrays[f"{det}_psd"].copy()
        psd[~np.isfinite(psd)] = 1.0          # only out-of-band bins (DC etc.), masked anyway
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=freqs, psd_array=psd)
        ifo.set_strain_data_from_frequency_domain_strain(
            arrays[f"{det}_strain_fd"], sampling_frequency=fs, duration=T, start_time=start)
        ifo.minimum_frequency, ifo.maximum_frequency = f_min, f_max
        ifos.append(ifo)

    # Preflight: SEOBNRv5_ROM needs HDF5-enabled LALSimulation + SEOBNRv5ROM_v1.0.hdf5 on $LAL_DATA_PATH.
    import lal, lalsimulation
    try:
        lalsimulation.SimInspiralChooseFDWaveform(
            35 * lal.MSUN_SI, 30 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0, 4e8 * lal.PC_SI, 0, 0, 0, 0, 0,
            1.0 / T, f_min, f_max, f_ref, None, lalsimulation.GetApproximantFromString(args.approximant))
    except RuntimeError as e:
        raise SystemExit(
            f"{args.approximant} preflight failed ({e}).\n"
            "Needs LALSimulation built with HDF5 and SEOBNRv5ROM_v1.0.hdf5 on $LAL_DATA_PATH:\n"
            "  * LDG: run via run_bilby_gw150914.sh (cvmfs igwn env; copies/uses ../lal_data)\n"
            "  * local: conda install -c conda-forge lalsuite bilby, then put the ROM file\n"
            "    (lalsuite-waveform-data on git.ligo.org) in a dir and export LAL_DATA_PATH=<dir>")

    wfg = bilby.gw.WaveformGenerator(
        duration=T, sampling_frequency=fs,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant=args.approximant, reference_frequency=f_ref,
                                minimum_frequency=f_min, maximum_frequency=f_max))

    pri = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    pri["chirp_mass"] = UniformInComponentsChirpMass(15.0, 50.0, name="chirp_mass")
    pri["mass_ratio"] = UniformInComponentsMassRatio(0.1, 1.0, name="mass_ratio")
    pri["mass_1"] = Constraint(1.0, 1000.0, name="mass_1")      # inactive: (Mc, q) box sets the support
    pri["mass_2"] = Constraint(1.0, 1000.0, name="mass_2")
    pri["chi_1"] = AlignedSpin(a_prior=Uniform(0.0, 0.9), z_prior=Uniform(-1.0, 1.0), name="chi_1")
    pri["chi_2"] = AlignedSpin(a_prior=Uniform(0.0, 0.9), z_prior=Uniform(-1.0, 1.0), name="chi_2")
    pri["luminosity_distance"] = PowerLaw(alpha=2.0, minimum=50.0, maximum=2000.0, name="luminosity_distance")
    pri["theta_jn"] = Sine(name="theta_jn")
    pri["dec"] = Cosine(name="dec")
    pri["ra"] = Uniform(0.0, 2 * np.pi, name="ra", boundary="periodic")
    pri["psi"] = Uniform(0.0, np.pi, name="psi", boundary="periodic")
    pri["phase"] = Uniform(0.0, 2 * np.pi, name="phase", boundary="periodic")
    pri["geocent_time"] = Uniform(trigger - 0.1, trigger + 0.1, name="geocent_time")

    like = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos, wfg, priors=pri,
        distance_marginalization=args.marginalize, time_marginalization=args.marginalize,
        phase_marginalization=args.marginalize, jitter_time=args.marginalize)

    result = bilby.run_sampler(
        like, pri, sampler="dynesty", outdir=outdir, label=args.label,
        nlive=args.nlive, naccept=args.naccept, sample="acceptance-walk", dlogz=0.1,
        npool=args.npool, maxiter=args.maxiter, check_point_delta_t=1800,
        check_point_plot=False, plot=False, print_method="interval-60",
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        meta_data=dict(frozen_data=str(data), psd_mode=meta["psd_mode"], sha256_16=meta["sha256_16"],
                       start_time=start, trigger_time=trigger, marginalized=args.marginalize))
    print(f"\nlog Z = {result.log_evidence:.3f} +/- {result.log_evidence_err:.3f}   "
          f"log BF = {result.log_bayes_factor:.3f}")
    print(f"result: {Path(outdir) / (args.label + '_result.json')}")


if __name__ == "__main__":
    import sys
    if sys.platform == "darwin":          # bilby's npool expects 'fork' (macOS defaults to 'spawn')
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
    main()
