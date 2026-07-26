#!/usr/bin/env python
"""bilby + dynesty ('acceptance-walk') run on the frozen 4s benchmark data.

This is the CPU half of the red/blue benchmark: it mirrors the reference
paper's `bilby_4s.py` (Prathaban et al. 2025, arXiv:2509.04336, Fig. 1) but
loads the strain and PSD from the frozen arrays in `data/` instead of a
pickle, so it consumes byte-identical inputs to `run_gwjax_frozen.py`.

Run somewhere with CPUs (16 cores recommended to match the paper):

    python run_bilby_frozen.py --npool 16

Requirements: bilby (>=2.x) + lalsuite (for the LAL IMRPhenomD waveform).
Expect O(50) CPU-hours (the paper's run was 2.99 h wall on 16 Icelake cores).
Checkpointing is on: re-running the same command resumes.

Output: <outdir>/<label>_result.json — feed it to make_comparison_plot.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import bilby

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", default=None, help="frozen data dir (default: ./data next to this script)")
parser.add_argument("--outdir", default="bilby_frozen_4s")
parser.add_argument("--label", default="frozen_4s")
parser.add_argument("--npool", type=int, default=16)
parser.add_argument("--nlive", type=int, default=1000)
parser.add_argument("--seed", type=int, default=88170235)  # same as bilby_4s.py
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

    # ── Waveform: LAL IMRPhenomD, f_ref = 50 Hz (as in the reference run) ───────
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=fs,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomD",
            reference_frequency=meta["reference_frequency"],
            minimum_frequency=f_min, maximum_frequency=f_max),
    )

    # ── Priors: paper Table 1 / bilby_4s.py with the power-law(2) distance ──────
    # BBHPriorDict(aligned_spin=True) supplies the defaults for psi/phase/ra/dec/
    # theta_jn, including their periodic boundaries.
    priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    priors["chirp_mass"] = bilby.core.prior.Uniform(25.0, 50.0, name="chirp_mass")
    priors["mass_ratio"] = bilby.core.prior.Uniform(0.25, 1.0, name="mass_ratio")
    priors["chi_1"] = bilby.core.prior.Uniform(minimum=-1.0, maximum=1.0, name="chi_1")
    priors["chi_2"] = bilby.core.prior.Uniform(minimum=-1.0, maximum=1.0, name="chi_2")
    priors["luminosity_distance"] = bilby.core.prior.analytical.PowerLaw(
        alpha=2.0, minimum=100.0, maximum=5000.0, name="luminosity_distance")
    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger - 0.1, trigger + 0.1, name="geocent_time")
    # Inert here (masses stay in ~[15, 121] Msun), kept to mirror bilby_4s.py:
    priors["mass_1"] = bilby.core.prior.Constraint(minimum=1, maximum=1000, name="mass_1")
    priors["mass_2"] = bilby.core.prior.Constraint(minimum=1, maximum=1000, name="mass_2")

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
        naccept=60,
        sample="acceptance-walk",
        dlogz=0.1,
        use_ratio=True,
        npool=args.npool,
        do_clustering=False,
        check_point_delta_t=1800,
        check_point_plot=False, plot=False,
        print_method="interval-60",
        injection_parameters=meta["injection"],
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
    # Linux default it was developed against).
    import sys
    if sys.platform == "darwin":
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
    main()
