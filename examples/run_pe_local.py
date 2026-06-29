#!/usr/bin/env python3
"""run_pe_local.py
==================
Full GWgpu_jax parameter-estimation pipeline as a single CLI script.

Two modes:

  * ``--synthetic`` (default)
      Inject an IMRPhenomD signal into coloured Gaussian noise (aLIGO PSD),
      then recover the source parameters with the blackjax-ns sampler.

  * ``--real``
      Fetch real GW150914 strain from GWOSC, estimate per-IFO PSDs from
      off-source data via Welch, and run the same sampler against it.
      Requires the ``[data]`` extra (``pip install "gwgpu_jax[data]"``).

Usage
-----
    python examples/run_pe_local.py                       # synthetic, fast
    python examples/run_pe_local.py --real                # real GW150914
    python examples/run_pe_local.py --num-live 800 \\
        --num-inner-steps 30 --max-iterations 5000        # higher quality

Output
------
  * Progress printed to stdout (every 100 NS iterations).
  * Posterior summary table (median, 16th, 84th percentiles).
  * ``corner.png`` if the ``corner`` package is available.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import gwgpu_jax


# Reference parameters used both as the synthetic injection and as plot truths
# for real-data mode (the published GW150914 best-fit, for visual reference only).
GW150914_REFERENCE = dict(
    m1=35.0,  m2=30.0,
    chi_1=0.0, chi_2=0.0,
    distance=410.0,
    inclination=0.4,
    tc=0.0, phi_c=0.0,
    ra=1.375, dec=-1.21, psi=0.0,
)


def build_network(duration: float, sampling_rate: float, f_min: float, f_max: float):
    """Construct a 2-detector H1/L1 network on a shared time/frequency grid."""
    grid = gwgpu_jax.TimeFrequencyGrid(
        duration=duration, sampling_rate=sampling_rate,
        f_min=f_min, f_max=f_max,
    )
    network = gwgpu_jax.Network.from_names(["H1", "L1"], grid)
    return network, grid


def setup_synthetic(network, grid, seed: int):
    """Generate aLIGO Gaussian noise + inject GW150914-like IMRPhenomD signal."""
    network.generate_noise(seed=seed)

    waveform_fn = gwgpu_jax.build_ripplegw_waveform_fn("IMRPhenomD", f_ref=20.0)
    hp, hc = waveform_fn(GW150914_REFERENCE, grid.frequency_domain_array)
    # Project with the SAME tc+dt_ifo convention as the sampler's likelihood, so
    # the injected merger sits at GW150914_REFERENCE["tc"]. network.project_waveform
    # applies ONLY the geometric delay dt_ifo (not tc); it would place the signal
    # at tc=0 — fine while tc=0, but it silently breaks for any nonzero tc.
    freqs = grid.frequency_domain_array
    gmst = float(network.gmst)
    ra, dec, psi = (GW150914_REFERENCE["ra"], GW150914_REFERENCE["dec"],
                    GW150914_REFERENCE["psi"])
    h_dict = {}
    for ifo in network.interferometers:
        Fp, Fc = ifo.antenna_pattern(ra, dec, psi, gmst)
        dt_ifo = ifo.time_delay_from_geocenter(ra, dec, gmst)
        h_dict[ifo.name] = gwgpu_jax.waveform_projection_fd(
            hp, hc, Fp, Fc, freqs, GW150914_REFERENCE["tc"] + dt_ifo)

    snrs = network.optimal_snr(h_dict)
    print("Injecting synthetic IMRPhenomD signal:")
    for ifo_name, snr in snrs.items():
        print(f"    {ifo_name}: SNR = {float(snr):.1f}")
    print(f"    network SNR = {float(network.network_optimal_snr(h_dict)):.1f}")

    network.inject_signal(h_dict, domain="fd")
    return GW150914_REFERENCE


def setup_real():
    """Fetch real GW150914 strain from GWOSC, attach to a fresh network."""
    # GW150914 standard analysis settings: 4 s @ 4 kHz, 20-1024 Hz band.
    network, grid = build_network(
        duration=4.0, sampling_rate=4096.0, f_min=20.0, f_max=1024.0,
    )
    print("Fetching GW150914 from GWOSC (requires gwpy + internet) …")
    gwgpu_jax.compat.attach_event_to_network(
        network, "GW150914",
        estimate_psd=True,
        psd_segment_duration=32.0,
        psd_offset=8.0,
    )
    print("Real strain + Welch PSDs attached to network.")
    # For real data we don't know the truth — return None.
    return network, grid, None


def make_sampler(network, fixed_params, seed: int, two_phase: bool = False):
    """Build a sampler with broad uniform priors on the marginalised params."""
    waveform_fn = gwgpu_jax.build_ripplegw_waveform_fn("IMRPhenomD", f_ref=20.0)

    param_bounds = {
        "m1":          (10.0, 80.0),
        "m2":          (10.0, 80.0),
        "distance":    (50.0, 1500.0),
        "inclination": (0.0, float(jnp.pi)),
        "ra":          (0.0, 2.0 * float(jnp.pi)),
        "dec":         (-float(jnp.pi) / 2, float(jnp.pi) / 2),
        "psi":         (0.0, float(jnp.pi)),
        "tc":          (-0.05, 0.05),
    }

    cls = (gwgpu_jax.GWgpu_jaxTwoPhaseNestedSampler if two_phase
           else gwgpu_jax.GWgpu_jaxNestedSampler)
    sampler = cls(
        network=network,
        waveform_fn=waveform_fn,
        param_bounds=param_bounds,
        fixed_params=fixed_params,
        gmst=0.0,
    )
    return sampler


def summarise_posterior(result, sampler, true_params=None):
    """Print median ± 1σ for every sampled parameter; flag degenerate columns."""
    print("\nPosterior summary (median, 68% credible interval):")
    print(f"  {'param':12s}  {'median':>10s}  {'-1σ':>8s}  {'+1σ':>8s}  truth")
    degenerate = []
    for name in sampler.param_bounds:
        s = np.asarray(result.posterior_samples[name])
        lo, mid, hi = np.percentile(s, [16, 50, 84])
        if (hi - lo) < 1e-12 * max(abs(mid), 1.0):
            degenerate.append(name)
        truth_str = f"{true_params[name]:+.3f}" if true_params and name in true_params else "—"
        print(f"  {name:12s}  {mid:+10.3f}  {mid-lo:8.3f}  {hi-mid:8.3f}  {truth_str}")

    if degenerate:
        print(
            f"\n⚠  posterior columns collapsed (no spread): {degenerate}\n"
            f"   ESS={result.ess:.1f}  →  NS likely didn't converge.\n"
            f"   Re-run with more live points or more inner steps, e.g.:\n"
            f"     --num-live 800 --num-inner-steps 50 --max-iterations 5000"
        )


def plot_corner(result, sampler, true_params, outfile: str):
    """Save a corner plot to ``outfile`` (if `corner` is installed).

    Always passes ``range=`` derived from the prior bounds so the plot
    succeeds even when posterior columns have collapsed (ESS≈1).
    """
    try:
        import corner  # type: ignore
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("\n(`corner`/`matplotlib` not installed — skipping plot.)")
        return

    names  = list(sampler.param_bounds.keys())
    data   = np.column_stack([np.asarray(result.posterior_samples[n]) for n in names])
    truths = [true_params[n] for n in names] if true_params else None
    # Force the plot range to the prior bounds so corner never complains
    # about degenerate columns and so the truth is always visible.
    plot_range = [sampler.param_bounds[n] for n in names]

    fig = corner.corner(
        data, labels=names, truths=truths,
        range=plot_range,
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    fig.savefig(outfile, dpi=120, bbox_inches="tight")
    print(f"Corner plot saved to {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true",
                      help="Inject a synthetic IMRPhenomD signal (default).")
    mode.add_argument("--real", action="store_true",
                      help="Fetch GW150914 strain from GWOSC (needs [data] extra).")
    parser.add_argument("--num-live",         type=int, default=500)
    # 5*d is a common rule of thumb for NSS inner steps; 40 covers the 8-D
    # default parameter set comfortably.
    parser.add_argument("--num-inner-steps",  type=int, default=40)
    parser.add_argument("--num-delete",       type=int, default=1)
    parser.add_argument("--max-iterations",   type=int, default=5000)
    parser.add_argument("--log-dlogz-target", type=float, default=-3.0)
    parser.add_argument("--num-posterior",    type=int, default=2000)
    parser.add_argument("--seed",             type=int, default=0)
    parser.add_argument("--corner-plot",      type=str, default="corner.png")
    # ── two-phase NS (bulk + accurate tail) ─────────────────────────────
    parser.add_argument("--two-phase", action="store_true",
                        help="Use GWgpu_jaxTwoPhaseNestedSampler: a fast batch-delete "
                             "phase 1 followed by a Skilling-classical phase 2.")
    parser.add_argument("--phase1-num-delete", type=int, default=None,
                        help="Particles deleted per phase-1 iteration. "
                             "Default: num_live // 20.")
    parser.add_argument("--phase1-delta-logz-threshold", type=float, default=-1.0,
                        help="Switch from phase 1 to phase 2 when "
                             "logZ_live - logZ < this value. Default -1.0 "
                             "(live points still hold ~37%% of total Z).")
    parser.add_argument("--phase1-max-iterations", type=int, default=1000)
    parser.add_argument("--phase2-num-delete", type=int, default=1,
                        help="Particles deleted per phase-2 iteration. "
                             "Default 1 (classical Skilling, unbiased).")
    args = parser.parse_args()

    # JAX setup — switch to float64 for likelihood accuracy.
    jax.config.update("jax_enable_x64", True)
    print(f"JAX devices: {jax.devices()}")

    # ── 1. Build network + data ──────────────────────────────────────────
    if args.real:
        network, grid, true_params = setup_real()
        # Use the published reference as plot truths (not a constraint).
        plot_truths = GW150914_REFERENCE
    else:
        network, grid = build_network(
            duration=4.0, sampling_rate=2048.0,
            f_min=20.0, f_max=512.0,
        )
        true_params = setup_synthetic(network, grid, seed=args.seed)
        plot_truths = true_params

    # ── 2. Sampler ───────────────────────────────────────────────────────
    fixed_params = {"chi_1": 0.0, "chi_2": 0.0, "phi_c": 0.0}
    sampler = make_sampler(network, fixed_params, seed=args.seed,
                           two_phase=args.two_phase)
    d = len(sampler.param_bounds)
    print(f"\nSampling dimension: {d}")
    print(f"Free parameters   : {list(sampler.param_bounds)}")
    print(f"Fixed parameters  : {fixed_params}")

    # ── 3. Run NS ────────────────────────────────────────────────────────
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
            rng_key=jax.random.PRNGKey(args.seed),
            num_live=args.num_live,
            num_inner_steps=args.num_inner_steps,
            num_delete=args.num_delete,
            max_iterations=args.max_iterations,
            log_dlogz_target=args.log_dlogz_target,
            num_posterior_samples=args.num_posterior,
            verbose=True,
        )
    elapsed = time.perf_counter() - t0

    # ── 4. Report ────────────────────────────────────────────────────────
    print(f"\nNS finished in {elapsed:.1f} s after {result.n_iterations} iterations.")
    if args.two_phase:
        print(f"  phase 1 iters = {result.phase1_iterations}")
        print(f"  phase 2 iters = {result.phase2_iterations}")
    print(f"  log Z  = {result.logZ:+.3f} ± {result.logZ_err:.3f}")
    print(f"  ESS    = {result.ess:.1f}")

    summarise_posterior(result, sampler, true_params=true_params)
    plot_corner(result, sampler, plot_truths, outfile=args.corner_plot)


if __name__ == "__main__":
    main()
