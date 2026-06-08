#!/usr/bin/env python3
"""run_pe_local_realdata.py
==========================
Full GWgpu_jax parameter-estimation pipeline against **real** GWOSC strain.

This is the real-data counterpart of ``examples/run_pe_local.py``. The
synthetic-injection branch is intentionally absent — every run pulls
public LIGO/Virgo strain from GWOSC via ``gwpy``, estimates per-IFO PSDs
from an adjacent off-source segment with Welch's method, and runs the
adaptive nested sampler on the residual.

Default event: **GW150914** (Phys. Rev. Lett. 116, 061102).

Requires the ``[data]`` extra (``pip install "gwgpu_jax[data]"``) — needs
network access to ``gwosc.org``.

Usage
-----
    # GW150914 (default), 4 s @ 4 kHz, 20–1024 Hz, broad priors
    python examples/run_pe_local_realdata.py

    # Same but a different event from KNOWN_EVENTS
    python examples/run_pe_local_realdata.py --event GW190521

    # GW170817 with the BayesWave-cleaned L1 frame (download separately
    # from DCC LIGO-T1700406, then point at the local file)
    python examples/run_pe_local_realdata.py --event GW170817 \\
        --deglitched-l1 /path/to/L-L1_CLEANED_HOFT_C02_T1700406_v1-1187007040-2048.gwf

    # Higher-quality run (slow on CPU; recommend Colab GPU)
    python examples/run_pe_local_realdata.py --num-live 800 \\
        --num-inner-steps 50 --max-iterations 8000

Caveats
-------
- This script uses ``IMRPhenomD`` (aligned-spin BBH) with spins **fixed
  to zero**. The published LVC analysis used the precessing
  ``IMRPhenomPv2``/``SEOBNRv4P`` waveforms with all 15 parameters free,
  so do not expect to reproduce the published posteriors exactly —
  expect a biased posterior, especially in inclination and distance.
- The PSD is a Welch estimate of a 32-s segment ending 8 s before the
  trigger. This is **not** the LVC-released PSD; expect ~10 % differences
  in log-likelihood relative to published numbers.
- **Coalescence-time convention.** ``gwgpu_jax.compat.attach_event_to_network``
  crops the data so the merger sits at ``0.875 * duration`` seconds
  into the segment (3.5 s into a 4-s GW150914 segment). The ripplegw
  IMRPhenomD ``tc`` parameter shifts the template merger to ``t = tc``,
  so the prior here is centered on ``0.875 * duration`` (NOT zero).
  Synthetic injections in ``run_pe_local.py`` use ``tc ≈ 0`` because
  the injection is built at the start of the segment.
- "Truth" overlays in the corner plot are the published median estimates
  for the event (visual reference, not a constraint).

Output
------
- Progress printed to stdout (every 100 NS iterations).
- Posterior summary table.
- ``corner_<event>.png`` with the prior-bounds range and the published
  median overlaid as the truth (purely for visual reference).
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import gwgpu_jax


# ── Published-median plot truths (visual reference only) ─────────────────────
#
# Sources: GWTC-1 (Phys. Rev. X 9, 031040) for GW150914/GW151226/GW170104/
#          GW170814/GW170817; GWTC-2 / GW190521 paper for GW190521.
# These are median values from the published runs — they are NOT used as
# priors or constraints, only as overlays in the corner plot.
PUBLISHED_MEDIAN = {
    "GW150914": dict(m1=35.6,  m2=30.6, distance=440.,  inclination=2.5,
                     ra=2.21, dec=-1.25, psi=0.0, tc=0.0),
    "GW151226": dict(m1=13.7,  m2=7.7,  distance=450.,  inclination=2.2,
                     ra=0.0,  dec=0.0,  psi=0.0, tc=0.0),
    "GW170104": dict(m1=30.8,  m2=20.0, distance=990.,  inclination=2.4,
                     ra=0.0,  dec=0.0,  psi=0.0, tc=0.0),
    "GW170814": dict(m1=30.6,  m2=25.2, distance=600.,  inclination=2.5,
                     ra=0.78, dec=-0.79, psi=0.0, tc=0.0),
    "GW170817": dict(m1=1.46,  m2=1.27, distance=40.0,  inclination=2.5,
                     ra=3.45, dec=-0.41, psi=0.0, tc=0.0),
    "GW190521": dict(m1=85.0,  m2=66.0, distance=5300., inclination=1.4,
                     ra=3.34, dec=0.82,  psi=0.0, tc=0.0),
}


# ── Per-event analysis settings (segment length, sample rate, band) ──────────
#
# Defaults match the LVC-standard analysis windows. BNS gets a longer
# segment and a higher Nyquist than BBH because the inspiral lives at
# higher frequencies for longer.
EVENT_SETTINGS = {
    "GW150914": dict(duration=4.0,  sampling_rate=4096.0, f_min=20.0, f_max=1024.0),
    "GW151226": dict(duration=8.0,  sampling_rate=4096.0, f_min=20.0, f_max=1024.0),
    "GW170104": dict(duration=4.0,  sampling_rate=4096.0, f_min=20.0, f_max=1024.0),
    "GW170814": dict(duration=4.0,  sampling_rate=4096.0, f_min=20.0, f_max=1024.0),
    "GW170817": dict(duration=128., sampling_rate=4096.0, f_min=23.0, f_max=2048.0),
    "GW190521": dict(duration=8.0,  sampling_rate=4096.0, f_min=11.0, f_max=512.0),
}


def setup_network(event: str):
    """Build the IFO network with event-appropriate grid settings."""
    cfg = EVENT_SETTINGS.get(event, EVENT_SETTINGS["GW150914"])
    grid = gwgpu_jax.TimeFrequencyGrid(
        duration=cfg["duration"], sampling_rate=cfg["sampling_rate"],
        f_min=cfg["f_min"], f_max=cfg["f_max"],
    )
    info = gwgpu_jax.compat.get_event_info(event)
    # Use every IFO the event publicly released.
    network = gwgpu_jax.Network.from_names(list(info.detectors), grid)
    print(f"Grid    : {grid}")
    print(f"Network : {[i.name for i in network.interferometers]}")
    return network, grid, info


def fetch_and_attach(network, event: str, deglitched_files: dict | None,
                     psd_segment_duration: float, psd_offset: float):
    """Pull strain from GWOSC, estimate Welch PSDs, attach to every IFO."""
    print(f"\nFetching {event} from GWOSC (gwpy.TimeSeries.fetch_open_data) …")
    t0 = time.perf_counter()
    populated = gwgpu_jax.compat.attach_event_to_network(
        network, event,
        estimate_psd          = True,
        psd_segment_duration  = psd_segment_duration,
        psd_offset            = psd_offset,
        deglitched_files      = deglitched_files,
        verbose               = False,
    )
    print(f"  ↳ done in {time.perf_counter()-t0:.1f} s.")
    print(f"  ↳ populated detectors: {[ifo.name for ifo in populated]}")
    return populated


def make_sampler(network, tc_center: float, two_phase: bool = False):
    """Build the sampler with broad uniform priors over the 8 PE parameters.

    The ``tc`` prior is **centered on ``tc_center``** rather than zero
    because ``attach_event_to_network`` puts the data's merger at
    ``0.875 * duration`` seconds into the segment, and ripplegw's
    ``tc`` parameter places the template merger at ``t = tc``.
    """
    waveform_fn = gwgpu_jax.build_ripplegw_waveform_fn("IMRPhenomD", f_ref=20.0)

    param_bounds = {
        "m1":          (5.0,  100.0),
        "m2":          (5.0,  100.0),
        "distance":    (50.0, 2000.0),
        "inclination": (0.0, float(jnp.pi)),
        "ra":          (0.0, 2.0 * float(jnp.pi)),
        "dec":         (-float(jnp.pi) / 2, float(jnp.pi) / 2),
        "psi":         (0.0, float(jnp.pi)),
        # Half-second window around the expected coalescence time.
        # ±0.5 s comfortably covers the trigger uncertainty for any GWOSC event.
        "tc":          (tc_center - 0.5, tc_center + 0.5),
    }
    fixed_params = {"chi_1": 0.0, "chi_2": 0.0, "phi_c": 0.0}

    cls = (gwgpu_jax.GWgpu_jaxTwoPhaseNestedSampler if two_phase
           else gwgpu_jax.GWgpu_jaxNestedSampler)
    sampler = cls(
        network       = network,
        waveform_fn   = waveform_fn,
        param_bounds  = param_bounds,
        fixed_params  = fixed_params,
        gmst          = 0.0,
    )
    print(f"\nSampling dimension : {len(sampler.param_bounds)}")
    print(f"Free parameters    : {list(sampler.param_bounds)}")
    print(f"Fixed parameters   : {fixed_params}")
    return sampler


def summarise_posterior(result, sampler, reference=None):
    """Print median ± 1σ; flag degenerate columns."""
    print("\nPosterior summary (median, 68% credible interval):")
    print(f"  {'param':12s}  {'median':>10s}  {'-1σ':>8s}  {'+1σ':>8s}  reference")
    degenerate = []
    for name in sampler.param_bounds:
        s = np.asarray(result.posterior_samples[name])
        lo, mid, hi = np.percentile(s, [16, 50, 84])
        if (hi - lo) < 1e-12 * max(abs(mid), 1.0):
            degenerate.append(name)
        ref = f"{reference[name]:+.3f}" if reference and name in reference else "—"
        print(f"  {name:12s}  {mid:+10.3f}  {mid-lo:8.3f}  {hi-mid:8.3f}  {ref}")

    if degenerate:
        print(
            f"\n⚠  posterior columns collapsed (no spread): {degenerate}\n"
            f"   ESS={result.ess:.1f}  →  NS likely didn't converge.\n"
            f"   Re-run with more live points or more inner steps, e.g.:\n"
            f"     --num-live 800 --num-inner-steps 50 --max-iterations 8000"
        )


def plot_corner(result, sampler, reference, outfile: str):
    """Save a corner plot to ``outfile`` (if ``corner`` is installed)."""
    try:
        import corner  # type: ignore
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("\n(`corner`/`matplotlib` not installed — skipping plot.)")
        return

    names   = list(sampler.param_bounds.keys())
    data    = np.column_stack([np.asarray(result.posterior_samples[n]) for n in names])
    truths  = [reference[n] for n in names] if reference else None
    pranges = [sampler.param_bounds[n] for n in names]

    fig = corner.corner(
        data, labels=names, truths=truths,
        range=pranges,
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_kwargs={"fontsize": 10},
        truth_color="C3",   # red truth lines so they stand out against blue
    )
    fig.savefig(outfile, dpi=120, bbox_inches="tight")
    print(f"Corner plot saved to {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--event",   type=str, default="GW150914",
                        choices=sorted(EVENT_SETTINGS),
                        help="Registered event name from gwgpu_jax.compat.KNOWN_EVENTS.")
    parser.add_argument("--deglitched-l1", type=str, default=None,
                        help=("Optional path to a BayesWave-cleaned L1 frame "
                              "(GW170817). Replaces the GWOSC fetch for L1."))
    parser.add_argument("--psd-segment-duration", type=float, default=32.0,
                        help="Off-source duration [s] used for the Welch PSD estimate.")
    parser.add_argument("--psd-offset",   type=float, default=8.0,
                        help="Gap [s] between off-source and on-source segments.")
    parser.add_argument("--num-live",     type=int, default=500)
    parser.add_argument("--num-inner-steps", type=int, default=40,
                        help="Slice-sampling steps per NS iteration (~5*d).")
    parser.add_argument("--num-delete",   type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--log-dlogz-target", type=float, default=-3.0)
    parser.add_argument("--num-posterior", type=int, default=2000)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--corner-plot",  type=str, default=None,
                        help="Output file for the corner plot (default: corner_<event>.png).")
    # ── two-phase NS (bulk + accurate tail) ──────────────────────────────
    parser.add_argument("--two-phase", action="store_true",
                        help="Use GWgpu_jaxTwoPhaseNestedSampler: a fast batch-delete "
                             "phase 1 followed by a Skilling-classical phase 2.")
    parser.add_argument("--phase1-num-delete", type=int, default=None,
                        help="Particles deleted per phase-1 iteration. "
                             "Default: num_live // 20.")
    parser.add_argument("--phase1-delta-logz-threshold", type=float, default=-1.0,
                        help="Switch from phase 1 to phase 2 when "
                             "logZ_live - logZ < this value. Default -1.0.")
    parser.add_argument("--phase1-max-iterations", type=int, default=1000)
    parser.add_argument("--phase2-num-delete", type=int, default=1,
                        help="Particles deleted per phase-2 iteration. "
                             "Default 1 (classical Skilling, unbiased).")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    print(f"JAX devices: {jax.devices()}")
    print(f"Event      : {args.event}")

    # ── 1. Build network + grid ─────────────────────────────────────────
    network, grid, info = setup_network(args.event)
    print(f"Trigger GPS: {info.gps_time}")
    print(f"Analysis   : duration={grid.duration}s, fs={grid.sampling_rate}Hz, "
          f"band=[{grid.f_min},{grid.f_max}]Hz")

    # ── 2. Pull data + estimate PSDs ─────────────────────────────────────
    deglitched = {"L1": args.deglitched_l1} if args.deglitched_l1 else None
    fetch_and_attach(
        network, args.event,
        deglitched_files     = deglitched,
        psd_segment_duration = args.psd_segment_duration,
        psd_offset           = args.psd_offset,
    )

    # ── 3. Sampler ──────────────────────────────────────────────────────
    # attach_event_to_network places the merger at 0.875 * duration into
    # the segment; the ripplegw template's tc parameter must match.
    tc_center = 0.875 * grid.duration
    print(f"Coalescence-time prior centred at tc = {tc_center:.3f} s "
          f"(merger position in the cropped segment).")
    sampler = make_sampler(network, tc_center=tc_center, two_phase=args.two_phase)

    # Update the reference dict so the corner truth marker uses the
    # absolute tc (the published-median table uses tc=0 by convention).
    reference = PUBLISHED_MEDIAN.get(args.event)
    if reference is not None:
        reference = {**reference, "tc": tc_center}

    # ── 4. Run NS ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if args.two_phase:
        phase1_num_delete = (args.phase1_num_delete
                             if args.phase1_num_delete is not None
                             else max(1, args.num_live // 20))
        print(
            f"\nRunning two-phase NS: num_live={args.num_live}, "
            f"phase1_num_delete={phase1_num_delete} "
            f"(switch at Δlog Z < {args.phase1_delta_logz_threshold}), "
            f"phase2_num_delete={args.phase2_num_delete} "
            f"(target Δlog Z < {args.log_dlogz_target})"
        )
        result = sampler.run_two_phase(
            rng_key                      = jax.random.PRNGKey(args.seed),
            num_live                     = args.num_live,
            num_inner_steps              = args.num_inner_steps,
            phase1_num_delete            = phase1_num_delete,
            phase1_delta_logz_threshold  = args.phase1_delta_logz_threshold,
            phase1_max_iterations        = args.phase1_max_iterations,
            phase2_num_delete            = args.phase2_num_delete,
            phase2_max_iterations        = args.max_iterations,
            log_dlogz_target             = args.log_dlogz_target,
            num_posterior_samples        = args.num_posterior,
            verbose                      = True,
        )
    else:
        print(
            f"\nRunning blackjax-ns: num_live={args.num_live}, "
            f"num_inner_steps={args.num_inner_steps}, max_iters={args.max_iterations}"
        )
        result = sampler.run(
            rng_key               = jax.random.PRNGKey(args.seed),
            num_live              = args.num_live,
            num_inner_steps       = args.num_inner_steps,
            num_delete            = args.num_delete,
            max_iterations        = args.max_iterations,
            log_dlogz_target      = args.log_dlogz_target,
            num_posterior_samples = args.num_posterior,
            verbose               = True,
        )
    elapsed = time.perf_counter() - t0

    # ── 5. Report ───────────────────────────────────────────────────────
    print(f"\nNS finished in {elapsed:.1f} s after {result.n_iterations} iterations.")
    if args.two_phase:
        print(f"  phase 1 iters = {result.phase1_iterations}")
        print(f"  phase 2 iters = {result.phase2_iterations}")
    print(f"  log Z  = {result.logZ:+.3f} ± {result.logZ_err:.3f}")
    print(f"  ESS    = {result.ess:.1f}")

    summarise_posterior(result, sampler, reference=reference)

    outfile = args.corner_plot or f"corner_{args.event}.png"
    plot_corner(result, sampler, reference, outfile=outfile)


if __name__ == "__main__":
    main()
