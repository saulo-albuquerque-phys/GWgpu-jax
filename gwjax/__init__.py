"""GWjax — JAX-based gravitational-wave parameter-estimation toolkit.

Public, lightweight API
-----------------------
``gwjax.TimeFrequencyGrid``, ``gwjax.Interferometer``, ``gwjax.Network``,
``gwjax.GWjaxNestedSampler``, ``gwjax.NestedSamplingResult``,
PSD helpers, and the likelihood/SNR utilities are re-exported here so
typical workflows can use::

    import gwjax
    grid = gwjax.TimeFrequencyGrid(duration=4.0, sampling_rate=2048.0)
    net  = gwjax.Network.from_names(["H1", "L1"], grid)

Lazy attributes
---------------
Symbols that pull in heavy backends (TensorFlow, the BBH/BNS ML models,
gwpy) are loaded on first access via PEP 562 ``__getattr__`` so that
``import gwjax`` stays fast and side-effect free:

  * ``gwjax.GWjaxWaveformGenerator``      → unified waveform front-end
  * ``gwjax.build_ripplegw_waveform_fn``  → ripplegw helper for the sampler
  * ``gwjax.compat``                      → :mod:`gwjax.gwjax_compatibility`
                                            (gwpy real-data ingest)

For example, ``gwjax.GWjaxWaveformGenerator("mlgw_bbh_jax")`` triggers
TensorFlow only at that point, not at ``import gwjax``.
"""

from __future__ import annotations

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "0.1.0"

# ── Lightweight re-exports ────────────────────────────────────────────────────
from gwjax.gwjax_timefrequencydomain_utils import TimeFrequencyGrid

from gwjax.gwjax_psd_utils import (
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

from gwjax.gwjax_likelihood_utils import (
    precompute_weight,
    inner_product,
    optimal_snr_squared,
    matched_filter_snr_squared,
    log_likelihood_ifo,
    log_likelihood_network,
    make_likelihood_fn,
    waveform_projection_fd,
)

from gwjax.gwjax_interferometer_utils import Interferometer
from gwjax.gwjax_network_ifos_utils  import Network

# User-supplied waveform wrapper — pure JAX, no heavy backends.
from gwjax.gwjax_custom_waveform import CustomWaveform

# Nested sampler — pure blackjax-ns + ripplegw-light, no TF at import time.
from gwjax.gwjax_samplers import (
    GWjaxNestedSampler,
    NestedSamplingResult,
)


# ── Lazy heavy imports (TF, gwpy) via PEP 562 module __getattr__ ─────────────

_LAZY = {
    # name on the gwjax namespace     →  (module path, attribute or None)
    "GWjaxWaveformGenerator":      ("gwjax.gwjax_waveformgenerator", "GWjaxWaveformGenerator"),
    "build_ripplegw_waveform_fn":  ("gwjax.gwjax_samplers",          "build_ripplegw_waveform_fn"),
    "compat":                      ("gwjax.gwjax_compatibility",     None),
}


def __getattr__(name: str):
    """Lazy attribute loader (PEP 562).

    Importing ``gwjax`` does not pay for TensorFlow or gwpy. Accessing a
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
    raise AttributeError(f"module 'gwjax' has no attribute {name!r}")


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
    # samplers
    "GWjaxNestedSampler",
    "NestedSamplingResult",
    # lazy
    "GWjaxWaveformGenerator",
    "build_ripplegw_waveform_fn",
    "compat",
]
