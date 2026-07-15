"""GWgpu_jax — JAX-based gravitational-wave parameter-estimation toolkit.

Public, lightweight API
-----------------------
``gwgpu_jax.TimeFrequencyGrid``, ``gwgpu_jax.Interferometer``, ``gwgpu_jax.Network``,
``gwgpu_jax.GWgpu_jaxNestedSampler``, ``gwgpu_jax.NestedSamplingResult``,
PSD helpers, and the likelihood/SNR utilities are re-exported here so
typical workflows can use::

    import gwgpu_jax
    grid = gwgpu_jax.TimeFrequencyGrid(duration=4.0, sampling_rate=2048.0)
    net  = gwgpu_jax.Network.from_names(["H1", "L1"], grid)

Lazy attributes
---------------
Symbols that pull in heavy backends (TensorFlow, the BBH/BNS ML models,
gwpy) are loaded on first access via PEP 562 ``__getattr__`` so that
``import gwgpu_jax`` stays fast and side-effect free:

  * ``gwgpu_jax.GWgpu_jaxWaveformGenerator``      → unified waveform front-end
  * ``gwgpu_jax.build_ripplegw_waveform_fn``  → ripplegw helper for the sampler
  * ``gwgpu_jax.compat``                      → :mod:`gwgpu_jax.gwgpu_jax_compatibility`
                                            (gwpy real-data ingest)

For example, ``gwgpu_jax.GWgpu_jaxWaveformGenerator("mlgw_bbh_jax")`` triggers
TensorFlow only at that point, not at ``import gwgpu_jax``.
"""

from __future__ import annotations

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "0.1.0"

# ── Lightweight re-exports ────────────────────────────────────────────────────
from gwgpu_jax.gwgpu_jax_timefrequencydomain_utils import TimeFrequencyGrid

from gwgpu_jax.gwgpu_jax_psd_utils import (
    get_psd,
    psd_from_data,
    psd_from_file,
    psd_aLIGO,
    psd_AdV,
    psd_KAGRA,
    psd_ET_D,
    psd_CE,
    psd_flat,
)

from gwgpu_jax.gwgpu_jax_likelihood_utils import (
    precompute_weight,
    inner_product,
    optimal_snr_squared,
    matched_filter_snr_squared,
    log_likelihood_ifo,
    log_likelihood_network,
    make_likelihood_fn,
    waveform_projection_fd,
)

from gwgpu_jax.gwgpu_jax_interferometer_utils import Interferometer
from gwgpu_jax.gwgpu_jax_network_ifos_utils  import Network

# User-supplied waveform wrapper — pure JAX, no heavy backends.
from gwgpu_jax.gwgpu_jax_custom_waveform import CustomWaveform

# Non-uniform prior definitions — pure JAX, jit-traceable, drop-in for the
# blackjax-ns prior. Lightweight (jax only), so eagerly re-exported.
from gwgpu_jax.gwgpu_jax_prior_definitions import (
    PriorSpec,
    Uniform,
    SinUniform,
    CosUniform,
    PowerLaw,
    Volumetric,
    build_prior,
    sample_prior,
    resolve_priors,
    PRIOR_ALIASES,
)

# Nested sampler — pure blackjax-ns + ripplegw-light, no TF at import time.
from gwgpu_jax.gwgpu_jax_samplers import (
    GWgpu_jaxNestedSampler,
    NestedSamplingResult,
)
from gwgpu_jax.gwgpu_jax_two_phase_sampler import (
    GWgpu_jaxTwoPhaseNestedSampler,
    TwoPhaseNestedSamplingResult,
)
# Conservative, GW-validated alternative: acceptance-walk kernel
# (Prathaban et al. 2025). Importing the class is cheap and side-effect free;
# the external kernel is only fetched/imported when you call .run().
from gwgpu_jax.gwgpu_jax_acceptance_walk import (
    GWgpu_jaxAcceptanceWalkSampler,
    AcceptanceWalkResult,
)


# ── Lazy heavy imports (TF, gwpy) via PEP 562 module __getattr__ ─────────────

_LAZY = {
    # name on the gwgpu_jax namespace     →  (module path, attribute or None)
    "GWgpu_jaxWaveformGenerator":      ("gwgpu_jax.gwgpu_jax_waveformgenerator", "GWgpu_jaxWaveformGenerator"),
    "build_ripplegw_waveform_fn":  ("gwgpu_jax.gwgpu_jax_samplers",          "build_ripplegw_waveform_fn"),
    "compat":                      ("gwgpu_jax.gwgpu_jax_compatibility",     None),
}


def __getattr__(name: str):
    """Lazy attribute loader (PEP 562).

    Importing ``gwgpu_jax`` does not pay for TensorFlow or gwpy. Accessing a
    lazy symbol triggers its import on first use, then caches it on the
    module so subsequent accesses are free.
    """
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        import importlib
        mod = importlib.import_module(module_path)
        value = mod if attr is None else getattr(mod, attr)
        globals()[name] = value          # cache for subsequent accesses
        return value
    raise AttributeError(f"module 'gwgpu_jax' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "__version__",
    # core
    "TimeFrequencyGrid",
    "Interferometer",
    "Network",
    # likelihood / SNR
    "precompute_weight",
    "inner_product",
    "optimal_snr_squared",
    "matched_filter_snr_squared",
    "log_likelihood_ifo",
    "log_likelihood_network",
    "make_likelihood_fn",
    "waveform_projection_fd",
    # PSDs
    "get_psd", "psd_from_data", "psd_from_file",
    "psd_aLIGO", "psd_AdV", "psd_KAGRA", "psd_ET_D", "psd_CE", "psd_flat",
    # user-supplied waveform wrapper
    "CustomWaveform",
    # prior definitions (non-uniform priors)
    "PriorSpec",
    "Uniform",
    "SinUniform",
    "CosUniform",
    "PowerLaw",
    "Volumetric",
    "build_prior",
    "sample_prior",
    "resolve_priors",
    "PRIOR_ALIASES",
    # samplers
    "GWgpu_jaxNestedSampler",
    "NestedSamplingResult",
    "GWgpu_jaxTwoPhaseNestedSampler",
    "TwoPhaseNestedSamplingResult",
    "GWgpu_jaxAcceptanceWalkSampler",
    "AcceptanceWalkResult",
    # lazy
    "GWgpu_jaxWaveformGenerator",
    "build_ripplegw_waveform_fn",
    "compat",
]
