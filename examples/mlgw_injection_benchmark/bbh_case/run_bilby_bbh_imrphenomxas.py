#!/usr/bin/env python
"""bilby + dynesty ('acceptance-walk') recovery of the frozen mlgw_bbh injection — IMRPhenomXAS.

A faster companion to `run_bilby_bbh.py` (LAL `SEOBNRv5_ROM`): same frozen
data, same priors, same sampler settings, but LAL `IMRPhenomXAS` instead —
purely analytic, no external ROM/HDF5 data file dependency, so it's both
quicker per likelihood call and free of the `$LAL_DATA_PATH` risk the ROM
run hit. `IMRPhenomXAS` is also *exactly* the waveform gwjax's own
ripplegw cross-check uses (see `generate_injection.py`'s triangulation
check and the notebook's section 8), so this closes the loop cleanly:

* `bilby + IMRPhenomXAS` vs `gwjax + ripplegw (IMRPhenomXAS)` — same
  physical model, different implementation (like `frozen_benchmark`'s
  ripple-vs-LAL parity, already validated to agree to ~1e-9) — a pure
  pipeline/numerics check, independent of the slower SEOBNRv5_ROM run.
* Both `bilby + IMRPhenomXAS` and `gwjax + mlgw_bbh_jax` can still be
  compared, carrying the same ~0.6% characterized IMRPhenomXAS-vs-SEOBNRv5
  systematic already measured in the triangulation check.

Run on the IGWN/LDG cluster (bilby + lalsuite preinstalled in the cvmfs
`igwn` conda env) or any machine with cores:

    python run_bilby_bbh_imrphenomxas.py --npool 16

Checkpointing is on: re-running the same command resumes. Writes to its own
outdir/label (`bilby_bbh_case_imrphenomxas`) so it cannot collide with the
SEOBNRv5_ROM run's checkpoint/output files.

Output: <outdir>/<label>_result.json — feed it to make_comparison_plot.py
together with run_gwjax_bbh.py's / the ripple run's result.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import bilby

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", default=None, help="frozen data dir (default: ./data next to this script)")
parser.add_argument("--outdir", default="bilby_bbh_case_imrphenomxas")
parser.add_argument("--label", default="bbh_case_imrphenomxas")
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

    # ── Waveform: LAL IMRPhenomXAS — purely analytic, no ROM data files ────────
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=fs,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomXAS",
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
