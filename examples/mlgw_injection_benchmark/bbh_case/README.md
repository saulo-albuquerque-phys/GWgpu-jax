# mlgw_bbh vs bilby — golden BBH injection benchmark

A second red/blue exercise in the spirit of `../../frozen_benchmark/`, but
this time exercising the *production* mlgw waveform models instead of the
reference paper's ripple IMRPhenomD injection. gwjax recovers with
`mlgw_bbh_jax` (SEOBNRv5HM, (2,2)-mode-only, aligned-spin — the same model
used in the real-GW150914 notebooks); bilby recovers with LAL
`SEOBNRv5_ROM`, the official LAL reduced-order model of that same SEOBNRv5
family. Because both sides use the same underlying physical waveform model
(different implementations — NN surrogate vs LAL ROM), any posterior/evidence
disagreement is attributable to the mlgw surrogate or the pipeline, not to
comparing two unrelated waveform families.

## The injection (`generate_injection.py`)

A "golden" aligned-spin BBH injection — clean, unimodal, inside the trained
`mc ∈ [26,32]` box the real-data notebooks use, away from prior edges and
orientation degeneracies (not a literal event reproduction):

* `mc = 28.0 M☉, q = 0.85` (→ m1≈34.9, m2≈29.7 M☉), `chi_1=0.10, chi_2=-0.05`
* `distance = 127 Mpc` (tuned for network SNR ≈ 27 on H1+L1, O4-design PSD —
  much closer than GW150914's ~440 Mpc because `inclination=1.0 rad` and this
  sky location are far less face-on/favourably-oriented than the real event)
* `inclination = 1.0 rad`, `ra=1.375, dec=-1.2108` (reused from the GW150914
  notebooks for continuity), `psi=0.8, phi_c=1.3`
* 4 s @ 4096 Hz, 20–1024 Hz, H1+L1, one fixed-seed (1234) Gaussian noise draw
  colored by the default aLIGO design PSD (not zero-noise)
* `tc` centered at `0.875×duration = 3.5 s` (the real-data notebooks' `tc`
  convention), GPS epoch reused from `frozen_benchmark`'s precedent of
  reusing the GW150914-era trigger time for unrelated synthetic injections

Run `python generate_injection.py` (`--distance`, `--seed` overridable) to
(re)freeze `data/` + `metadata.json`. Before freezing, it also runs the
ripplegw triangulation check (see below) and prints per-IFO/network SNR —
watch for both before trusting a re-generated injection.

### Ripplegw triangulation check

Cross-checks the mlgw_bbh injection against an independent ripplegw
`IMRPhenomXAS` waveform (aligned-spin, closest ripple analytic family to
SEOBNRv5) at the same parameters. mlgw's zero-crossing TD→FD recipe puts the
template merger at `tc − MLGW_FINAL_TIME`, so the injection uses mlgw
`tc = tc_in_segment + MLGW_FINAL_TIME` (stored as `mlgw_tc_param` in
`metadata.json`) and the merger lands exactly on the trigger; ripple is
projected at `tc_in_segment`, and the best residual `dtc` is 0.00 ms. The
script still scans a small `tc` window and reports the time-and-phase-maximized
overlap (the standard mismatch definition): 0.99775 (2.3e-3 mismatch).

**Nov 2026 recipe fix:** `MLGW_FINAL_TIME` is now 0.1 s (was 0.01 s). 0.01 s
cut the ringdown off at ~15% of peak amplitude: mismatch vs SEOBNRv5HM 5.6e-3
→ ~1e-4 (`../../referee_response/waveform_check/`), and the old overlap vs
IMRPhenomXAS was only ≈0.994. The value is written to `metadata.json`
and each `result.json`; runner/plotter read it (data without the key = 0.01 s).
`data/` was regenerated with the fix (same seed/distance; SNR 27.23) — the old
data and the runs on it are in `drive_data/bbh_case/`. **Both the gwjax and the
bilby runs must be redone on the new `data/`.**

ripple is independently validated to agree with LAL
to ~1e-9 (`frozen_benchmark/README.md`), so this triangulates three
implementations (mlgw / ripple / LAL) rather than two.

## Matched configuration

| | bilby (blue) | gwjax (red) |
|---|---|---|
| waveform | LAL `SEOBNRv5_ROM`, f_ref = 20 | `mlgw_bbh_jax` (SEOBNRv5HM, model_4, (2,2) mode) |
| live points | nlive = 2000 | num_live = 2000 |
| walk | naccept = 200 | n_target = 200, max_mcmc = 1200 |
| termination | dlogz = 0.1 | log_dlogz_target = -3 (deeper than bilby's default; not bit-matched — see note) |
| priors | `mc∈[26,30], q∈[0.6,1], chi_1,chi_2∈[-0.4,0.4]`, `d_L∝d²∈[50,300]`, `sinθ_JN, cosδ`, `t∈trigger±0.1` | identical |
| marginalizations | none | none |

Known convention difference (not a bug, same spirit as `frozen_benchmark`):
gwjax's termination criterion (`log_dlogz_target=-3`, reused unchanged from
the validated real-GW150914 notebook) is not bit-matched to bilby's
`dlogz<0.1`; both are deep enough for convergence, so this should not move
the posteriors, only (slightly) the run time.

## How to run

**GPU side** (Colab): open `COLAB_run_gwjax_bbh.ipynb`, run all. It expects
`data/` (this dir) uploaded to your Google Drive at
`MyDrive/mlgw_injection_benchmark/bbh_case/data/` first — see the notebook's
cell 3 markdown; it checksum-verifies against `metadata.json` before
sampling, so a stale/mismatched upload is caught immediately rather than
silently producing a non-comparable run. Or locally/cluster:
`python run_gwjax_bbh.py` (needs gwgpu_jax[mlgw] + the acceptance-walk
kernel — `bash ../../../scripts/fetch_acceptance_walk_kernel.sh` first).
Local smoke test (validates the pipeline, not convergence):

    python run_gwjax_bbh.py --nlive 50 --n-target 10 --max-mcmc 100 \
        --num-delete 10 --max-iters 30 --num-posterior 50 --outdir /tmp/smoke

**CPU side** (IGWN/LDG cluster recommended — bilby, lalsuite, and the
SEOBNRv5_ROM `$LAL_DATA_PATH` data files are preinstalled in the cvmfs `igwn`
conda env). `run_bilby_bbh.py` also accepts `--maxiter N` to cap NS
iterations for a fast, bounded smoke test (dynesty's normal `dlogz`
convergence criterion doesn't shrink with `--nlive`, so an uncapped "small"
local test can still run for a long time — each `SEOBNRv5_ROM` likelihood
call is fast, but many are needed to actually converge):

    ssh albert.einstein@ldas-grid.ligo.caltech.edu
    git clone -b acceptance_walk_sampler https://github.com/saulo-albuquerque-phys/GWgpu-jax.git
    cd GWgpu-jax/examples/mlgw_injection_benchmark/bbh_case
    # edit accounting_group / accounting_group_user in run_bilby_bbh.sub
    condor_submit run_bilby_bbh.sub

or interactively: `bash run_bilby_bbh.sh 16`. `run_bilby_bbh.py` preflights
`SEOBNRv5_ROM` waveform generation before starting dynesty and fails with an
actionable message (check `$LAL_DATA_PATH` / install `lalsuite-extra`) if
the ROM data files aren't available — confirmed missing on at least one
local pip-installed lalsuite, expected present on IGWN cvmfs.

**Overlay**:

    python make_comparison_plot.py \
        --gwjax gwjax_bbh_case \
        --bilby bilby_bbh_case/bbh_case_result.json \
        --out   bbh_case_comparison.png

## What "success" looks like

Same bar as `frozen_benchmark`: posteriors visually consistent, `ΔlogBF`
within ~combined logZ error bars, no railed/degenerate columns.
