# Acceptance-walk nested sampling (GW-validated)

This branch samples with the **acceptance-walk** nested-sampling kernel of
**Prathaban et al. (2025)** — the `bilby`/`dynesty` constrained random-walk move
that is the LIGO/Virgo community standard, ported to GPU inside the
[blackjax-ns](https://github.com/handley-lab/blackjax) framework and shown there
to recover posteriors and evidences **statistically identical** to the CPU
`bilby` pipeline for aligned-spin BBH (arXiv:2509.04336).

Your gwgpu_jax **likelihood, waveform, network, and priors are used unchanged** —
only the sampling kernel differs from `GWgpu_jaxNestedSampler` (BlackJAX-NS
Nested Slice Sampling). Use this driver when you want a nested-sampling kernel
that has itself been validated on gravitational-wave data.

## 1. Fetch the kernel (once)

The kernel lives in a separate, **separately-licensed** repository
([`mrosep/blackjax_ns_gw`](https://github.com/mrosep/blackjax_ns_gw)) and is
**not vendored** into gwgpu_jax. Fetch just the `custom_kernels` package (no
large data files) into the git-ignored `external/` directory:

```bash
bash scripts/fetch_acceptance_walk_kernel.sh
```

At run time the sampler finds the kernel via, in order:
1. an already-importable `custom_kernels` package;
2. `GWJAX_ACCEPTANCE_WALK_SRC` (path to the `src/` dir containing `custom_kernels/`);
3. `<repo>/external/blackjax_ns_gw/src`.

It pins the **same** blackjax commit gwgpu_jax uses
(`dedbf11da33eb5ca286f6731e2c51f2b254b953f`), so there is no version conflict.

## 2. Run it — standard network + waveform likelihood

Drop-in for `GWgpu_jaxNestedSampler` / `GWgpu_jaxTwoPhaseNestedSampler`:

```python
import jax, gwgpu_jax
from gwgpu_jax import GWgpu_jaxAcceptanceWalkSampler

sampler = GWgpu_jaxAcceptanceWalkSampler(
    network      = net,
    waveform_fn  = waveform_fn,
    param_bounds = PARAM_BOUNDS,
    priors       = PRIORS,           # same sin/cos/volumetric aliases as NSS
    gmst         = None,             # None -> net.gmst
    periodic_params = ("phi_c", "ra"),
)

result = sampler.run(
    rng_key   = jax.random.PRNGKey(0),
    num_live  = 1000,
    n_target  = 60,      # accepted MCMC steps per new point (bilby nact-style)
    max_mcmc  = 5000,    # hard cap on proposals per new point
    num_delete = 1,      # classical single replacement (lowest bias)
    log_dlogz_target      = -3.0,
    num_posterior_samples = 2000,
    verbose   = True,
)
result.posterior_samples["m1"]      # physical-space posterior (dict of arrays)
result.logZ, result.logZ_err, result.ess, result.n_iterations
```

`result.posterior_samples` is in **physical** parameter space (the unit-cube
draws are mapped back through the prior transform), so it matches
`NestedSamplingResult`.

## 3. Run it — arbitrary likelihood (e.g. relative binning)

For a custom likelihood (marginalised relative binning, etc.), build the sampler
from a physical log-likelihood directly:

```python
sampler = GWgpu_jaxAcceptanceWalkSampler.from_loglikelihood(
    loglikelihood_fn = log_L,        # log_L(params_dict) -> scalar, jit/vmap-safe
    param_bounds     = PARAM_BOUNDS, # keys must match log_L's expected params
    priors           = PRIORS,
    periodic_params  = ("phi_c",),
)
result = sampler.run(jax.random.PRNGKey(0), num_live=1000, n_target=60)
```

## Tuning notes

- **`n_target`** (default 60) is the target number of *accepted* walk steps per
  new live point — bilby's `nact` control. Too low ⇒ under-dispersed posteriors
  (a quick 2-D Gaussian check showed σ pulled ~5–10 % low at `n_target=20`);
  60 is the safe default. Higher = more decorrelated, slower.
- **`num_delete`** = live points replaced per NS step. `1` is the classical,
  lowest-bias choice; larger batches are faster on GPU at a small evidence-bias
  cost.
- **`periodic_params`** should list genuinely periodic angles (e.g. `phi_c`,
  `ra`) so their unit-cube coordinate wraps during the walk.

## Validation status in this repo

- The unit-cube prior transform is built from gwgpu_jax's **own** `PriorSpec`
  objects (added `icdf`), so the sampling prior is *identical* to the NSS path
  (verified to sampling noise).
- End-to-end synthetic PE runs and returns finite evidence + physical posterior.
- A 2-D Gaussian recovers mean/σ and logZ with healthy ESS.
- Full GW calibration (PP test) should be run on GPU — see the Colab notebooks
  `examples/FINALRESULTSCOLAB..._acceptance_walk.ipynb`.

## Citation

- **Prathaban, Yallup, Alvey, Yang, Templeton & Handley (2025)**, "Gravitational-wave
  inference at GPU speed: A bilby-like nested sampling kernel within blackjax-ns",
  arXiv:2509.04336 — the acceptance-walk kernel.
- **Cabezas et al. (2024)**, "BlackJAX", arXiv:2402.10797 — the framework.
- **Speagle (2020)**, dynesty, MNRAS 493, 3132; **Ashton et al. (2019)**, Bilby,
  ApJS 241, 27 — the acceptance-walk algorithm reproduced by the kernel.

## Licensing

The acceptance-walk kernel repository declares no license at the time of writing,
so it is **not** copied into gwgpu_jax; it is fetched at your request into the
git-ignored `external/`. Do not redistribute its code as part of this repository.
Confirm terms of use with its authors before publishing derived work.
