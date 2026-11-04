#!/usr/bin/env python
"""bilby + dynesty ('acceptance-walk') recovery of the frozen mlgw_bbh injection.

CPU half of the mlgw_bbh vs bilby benchmark. Loads the frozen FD strain + PSD
arrays written by `generate_injection.py` (byte-identical to what
`run_gwjax_bbh.py` consumes) and recovers with LAL `SEOBNRv5_ROM` — the
official LAL reduced-order model of the same aligned-spin, (2,2)-mode-only
SEOBNRv5 family that `mlgw_bbh_jax` is a surrogate of (verified locally:
`SEOBNRv5_ROM` is implemented as an FD approximant in this LALSimulation
build; the full `SEOBNRv5HM` name is registered but not wired into the
legacy Choose{TD,FD}Waveform path here, so the ROM is the safe match).

Run on the IGWN/LDG cluster (bilby + lalsuite preinstalled in the cvmfs
`igwn` conda env) or any machine with cores:

    python run_bilby_bbh.py --npool 16

Checkpointing is on: re-running the same command resumes.

Output: <outdir>/<label>_result.json — feed it to make_comparison_plot.py
together with run_gwjax_bbh.py's result.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import bilby

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", default=None, help="frozen data dir (default: ./data next to this script)")
parser.add_argument("--outdir", default="bilby_bbh_case")
parser.add_argument("--label", default="bbh_case")
parser.add_argument("--npool", type=int, default=16)
parser.add_argument("--nlive", type=int, default=2000)
parser.add_argument("--naccept", type=int, default=200)   # matches gwjax's N_TARGET
parser.add_argument("--seed", type=int, default=88170235)
parser.add_argument("--maxiter", type=int, default=None,
                    help="cap on NS iterations (smoke-test only — bounds runtime "
                         "regardless of dlogz convergence; leave unset for real runs)")
args = parser.parse_args()


def main():

    DATA = Path(args.data) if args.data else Path(__file__).resolve().parent / "data"
    meta = json.load(open(DATA / "metadata.json"))

    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)
    bilby.core.utils.random.seed(args.seed)

    duration = meta["duration"]
    fs       = meta["sampling_frequency"]
    start    = meta["start_time"]
    trigger  = meta["trigger_time"]
    f_min    = meta["minimum_frequency"]
    f_max    = meta["maximum_frequency"]
    f_ref    = meta["reference_frequency"]
    inj      = meta["injection"]

    freqs = np.load(DATA / "frequency_array.npy")

    # ── Interferometers from the frozen FD strain + PSD arrays ──────────────────
    ifos = bilby.gw.detector.InterferometerList([])
    for det in meta["detectors"]:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=freqs, psd_array=np.load(DATA / f"{det}_psd.npy"))
        ifo.set_strain_data_from_frequency_domain_strain(
            np.load(DATA / f"{det}_strain_fd.npy"),
            sampling_frequency=fs, duration=duration, start_time=start)
        ifo.minimum_frequency = f_min
        ifo.maximum_frequency = f_max
        ifos.append(ifo)

    # ── Waveform: LAL SEOBNRv5_ROM — the LAL ROM of the SEOBNRv5(HM) family
    #    mlgw_bbh_jax is a surrogate of (aligned-spin, injected here with the
    #    (2,2) mode only) ─────────────────────────────────────────────────────
    # Preflight check: SEOBNRv5_ROM needs HDF5-enabled LALSimulation + the ROM
    # data files on $LAL_DATA_PATH (both standard in the IGWN cvmfs `igwn`
    # conda env, but NOT guaranteed for an arbitrary pip-installed lalsuite —
    # confirmed missing on at least one local dev machine). Fail loudly here
    # instead of deep inside dynesty's first likelihood call.
    import lal, lalsimulation
    try:
        lalsimulation.SimInspiralChooseFDWaveform(
            35.0 * lal.MSUN_SI, 30.0 * lal.MSUN_SI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0e6 * lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0 / duration, f_min, f_max, f_ref, None,
            lalsimulation.GetApproximantFromString("SEOBNRv5_ROM"))
    except RuntimeError as e:
        raise RuntimeError(
            "SEOBNRv5_ROM waveform generation failed a preflight check "
            f"({e}). This almost always means the ROM data files are not on "
            "$LAL_DATA_PATH, or LALSimulation was built without HDF5 support. "
            "On the IGWN/LDG cluster, load the cvmfs `igwn` conda env (bilby + "
            "lalsuite + LAL_DATA_PATH are preconfigured there); locally, "
            "install `lalsuite-extra` / set $LAL_DATA_PATH to a directory "
            "containing the SEOBNRv5ROM*.h5 data files."
        ) from e
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=fs,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant="SEOBNRv5_ROM",
            reference_frequency=f_ref,
            minimum_frequency=f_min, maximum_frequency=f_max),
    )

    # ── Priors: same box as run_gwjax_bbh.py's PARAM_BOUNDS/PRIORS ─────────────
    # BBHPriorDict(aligned_spin=True) supplies isotropic-orientation defaults for
    # theta_jn (Sine)/dec (Cosine)/ra/psi/phase — matching gwjax's "sin"/"cos"
    # PRIORS with no override needed.
    priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    priors["chirp_mass"] = bilby.core.prior.Uniform(
        inj["chirp_mass"] - 2.0, inj["chirp_mass"] + 2.0, name="chirp_mass")
    priors["mass_ratio"] = bilby.core.prior.Uniform(0.6, 1.0, name="mass_ratio")
    priors["chi_1"] = bilby.core.prior.Uniform(minimum=-0.4, maximum=0.4, name="chi_1")
    priors["chi_2"] = bilby.core.prior.Uniform(minimum=-0.4, maximum=0.4, name="chi_2")
    priors["luminosity_distance"] = bilby.core.prior.analytical.PowerLaw(
        alpha=2.0, minimum=50.0, maximum=300.0, name="luminosity_distance")
    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger - 0.1, trigger + 0.1, name="geocent_time")
    priors["mass_1"] = bilby.core.prior.Constraint(minimum=1, maximum=200, name="mass_1")
    priors["mass_2"] = bilby.core.prior.Constraint(minimum=1, maximum=200, name="mass_2")

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos, waveform_generator, priors=priors,
        time_marginalization=False,
        phase_marginalization=False,
        distance_marginalization=False,
    )

    result = bilby.run_sampler(
        likelihood, priors,
        sampler="dynesty",
        outdir=args.outdir, label=args.label,
        nlive=args.nlive,
        naccept=args.naccept,
        sample="acceptance-walk",
        dlogz=0.1,
        use_ratio=True,
        npool=args.npool,
        do_clustering=False,
        maxiter=args.maxiter,
        check_point_delta_t=1800,
        check_point_plot=False, plot=False,
        print_method="interval-60",
        injection_parameters=dict(
            chirp_mass=inj["chirp_mass"], mass_ratio=inj["mass_ratio"],
            chi_1=inj["chi_1"], chi_2=inj["chi_2"],
            luminosity_distance=inj["luminosity_distance"], theta_jn=inj["theta_jn"],
            psi=inj["psi"], phase=inj["phase"], geocent_time=inj["geocent_time"],
            ra=inj["ra"], dec=inj["dec"],
        ),
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    )

    print(f"\nlog_evidence       = {result.log_evidence:.4f} +/- {result.log_evidence_err:.4f}")
    print(f"log_noise_evidence = {result.log_noise_evidence:.4f}")
    print(f"log_bayes_factor   = {result.log_bayes_factor:.4f}")
    print(f"Result: {Path(args.outdir) / (args.label + '_result.json')}")
    print("Copy that file back and run make_comparison_plot.py.")


if __name__ == "__main__":
    # macOS defaults to the 'spawn' start method, which would re-import this
    # script in every pool worker; bilby's npool path expects 'fork' (the
    # Linux default it was developed against, and what the IGWN cluster uses).
    import sys
    if sys.platform == "darwin":
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
    main()
