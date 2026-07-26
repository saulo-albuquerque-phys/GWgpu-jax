# Frozen 4s benchmark — gwjax vs bilby, red/blue

A Fig.-1-style head-to-head between **gwgpu_jax + acceptance-walk kernel**
(GPU) and **bilby + dynesty `acceptance-walk`** (CPU) on **byte-identical
frozen data**: the synthetic 4-second BBH injection of Prathaban et al. 2025
(arXiv:2509.04336, Table 2 / Fig. 1). Because both samplers consume the same
frequency-domain strain and PSD arrays, any posterior/evidence difference is
attributable to the sampling pipelines alone — no injection, conditioning, or
PSD-estimation degrees of freedom.

## The frozen data (`data/`)

Copied from the public reference repo `mrosep/blackjax_ns_gw@main`
(`src/4s_*.npy`, verified **bit-identical** to the `4s_H1L1V1.pkl` that the
paper's own `bilby_4s.py` consumed). PSDs are ASD² linearly interpolated onto
the frequency array — verified bit-identical to the PSDs inside that pickle.
`metadata.json` records the injection parameters, times, and sha256 checksums;
both runners verify the checksums before sampling. If `data/` is missing,
`run_gwjax_frozen.py` rebuilds it from the public repo automatically.

* 3 detectors (H1, L1, V1 at O4 design sensitivity), 4 s, fs = 2048 Hz,
  band 20–1024 Hz, network optimal SNR ≈ 40.
* Injection: Mc = 35 M☉, q = 0.9, χ₁ = 0.4, χ₂ = −0.3, d_L = 1000 Mpc,
  θ_JN = 0.4, trigger GPS 1126259642.413, segment start 1126259640.413.

## Matched configuration

| | bilby (blue) | gwjax (red) |
|---|---|---|
| waveform | LAL IMRPhenomD, f_ref = 50 | ripple IMRPhenomD, f_ref = 50 |
| live points | nlive = 1000 | nlive = 1400, num_delete = 700 (effective 1000) |
| walk | naccept = 60 | n_target = 60, max_mcmc = 5000 |
| termination | dlogZ < 0.1 | logZ_live − logZ < ln(e^0.1 − 1) (same criterion) |
| priors | Table 1: Mc U[25,50], q U[0.25,1], spins U[−1,1], d_L ∝ d² [100,5000], sin θ_JN, cos δ, t U[trigger ± 0.1] | identical |
| marginalizations | none | none |

Likelihood parity was verified before shipping: on this frozen data, at the
injection point, gwjax vs bilby+ripple agree to **3×10⁻⁹** in logL and gwjax
vs bilby+LAL to **6×10⁻⁶** (the LAL↔ripple waveform systematic — negligible
at ΔlogL-ratio ≈ 780).

## How to run

**GPU side** (Colab): open `COLAB_run_gwjax_frozen.ipynb`, run all. Or
locally/cluster: `python run_gwjax_frozen.py` (needs gwgpu_jax + the
acceptance-walk kernel fetched via `scripts/fetch_acceptance_walk_kernel.sh`).
~1–2 h on an L4-class GPU. Outputs `gwjax_frozen_4s/{posterior_samples.npz,
result.json}`.

**CPU side** (any machine with cores): copy this folder, then

    pip install bilby lalsuite
    python run_bilby_frozen.py --npool 16

~3 h wall on 16 cores (≈ 48 CPU-h); checkpointed every 30 min, so it resumes
if interrupted. Outputs `bilby_frozen_4s/frozen_4s_result.json`.

On an IGWN/LDG cluster (recommended if you have LIGO.ORG credentials — bilby
and lalsuite are preinstalled in the cvmfs `igwn` conda env, no pip needed):

    ssh albert.einstein@ldas-grid.ligo.caltech.edu
    git clone -b pp_test_investigation https://github.com/saulo-albuquerque-phys/GWgpu-jax.git
    cd GWgpu-jax/examples/frozen_benchmark
    # edit accounting_group / accounting_group_user in run_bilby_frozen.sub
    condor_submit run_bilby_frozen.sub      # then: condor_q / tail -f condor_frozen_4s.out

or interactively on an `ldas-pcdev*` node inside tmux:
`bash run_bilby_frozen.sh 16`. Eviction/interruption is safe either way —
resubmit and it resumes from the checkpoint.

**Overlay**:

    python make_comparison_plot.py \
        --gwjax gwjax_frozen_4s \
        --bilby bilby_frozen_4s/frozen_4s_result.json \
        --out   frozen_4s_comparison.png

Blue = bilby+dynesty, red = gwjax, black lines = truth. The script also prints
the logZ / log-Bayes-factor comparison and per-parameter quantiles.

## What "success" looks like

Paper Fig. 1 / Fig. 4: posteriors visually indistinguishable and logZ
consistent within its ~0.15-nat error bars. The paper's own chains for this
exact injection are on Zenodo (doi:10.5281/zenodo.17012011) for a three-way
check. Note the two runs use different waveform *implementations* (LAL vs
ripple), so agreement here also re-validates ripple — the ~6×10⁻⁶ logL
difference above says this cannot visibly move the corners.
