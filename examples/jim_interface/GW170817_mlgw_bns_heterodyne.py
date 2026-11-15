"""GW170817 with Jim's flowMC sampler, relative binning and the mlgw_bns surrogate.

Jim's own ``examples/GW170817_heterodyne.py`` with the waveform swapped for
``MLGWBNSJimWaveform`` (TEOBResumS surrogate) and the frequency range and
priors kept inside the surrogate's range.  See README.md in this folder for
the environment.  Set ``JIM_SMOKE=1`` for a tiny run that only checks the
plumbing.
"""

import os
import time
from pathlib import Path

import corner
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    PowerLawPrior,
    SinePrior,
    UniformPrior,
)
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.helpers import (
    get_default_likelihood_transforms,
    get_default_periodic,
    get_default_sample_transforms,
)
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.samplers.config import FlowMCConfig

from gwgpu_jax.gwgpu_jax_jim_interface import MLGWBNSJimWaveform

SMOKE = os.environ.get("JIM_SMOKE") == "1"

# ----------------------------------------------------------------------
# Fetch data
# ----------------------------------------------------------------------

# fetch a 128s segment centered on GW170817
gps = 1187008882.43
duration = 128.0
# Request a segment with 2.0 s post-merger
start = gps + 2.0 - duration
end = start + duration

# fetch 2048s of data to estimate the PSD (4096s has a gap in V1; 2048s is clean for all three)
psd_start = start - 2048
psd_end = start

# frequency range of the analysis: 23 - 1500 Hz as in the GWgpu_jax GW170817
# runs, well inside the band of the surrogate
fmin = 23.0
fmax = 1500.0

# initialize detectors
ifos = [get_H1(), get_L1(), get_V1()]

for ifo in ifos:
    strain_data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(strain_data)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    ifo.set_psd(
        psd_data.to_psd(
            nperseg=int(strain_data.duration * strain_data.sampling_frequency)
        )
    )

# ----------------------------------------------------------------------
# Waveform model
# ----------------------------------------------------------------------

waveform = MLGWBNSJimWaveform()

# ----------------------------------------------------------------------
# Prior
# ----------------------------------------------------------------------

prior = CombinePrior(
    [
        UniformPrior(1.197, 1.199, parameter_names=["M_c"]),
        UniformPrior(0.5, 1.0, parameter_names=["q"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s1_z"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(1.0, 100.0, 2.0, parameter_names=["d_L"]),
        UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
    ]
)

# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------

sample_transforms = get_default_sample_transforms(prior, ifos)

likelihood_transforms = get_default_likelihood_transforms(prior, ifos, waveform)

# ----------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------

likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    n_bins=1000,
    f_min=fmin,
    f_max=fmax,
    prior=prior,
    likelihood_transforms=likelihood_transforms,
    phase_marginalization=True,
)

prior = CombinePrior([p for p in prior.base_prior if p.parameter_names[0] != "phase_c"])

# ----------------------------------------------------------------------
# Sample
# ----------------------------------------------------------------------

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    periodic=get_default_periodic(prior, sample_transforms),
    sampler_config=FlowMCConfig(
        n_chains=20 if SMOKE else 1000,
        n_local_steps=5 if SMOKE else 100,
        n_global_steps=5 if SMOKE else 1000,
        n_training_loops=1 if SMOKE else 50,
        n_production_loops=1 if SMOKE else 10,
        n_NFproposal_batch_size=20 if SMOKE else 100,
        global_thinning=1 if SMOKE else 100,
    ),
    verbose=True,
)

start_time = time.time()
jim.sample()
end_time = time.time()
print(f"Sampling took {(end_time - start_time) / 60:.2f} mins")

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

diagnostics = jim.get_diagnostics()
print(f"Likelihood evaluations: {diagnostics['n_likelihood_evaluations']:,}")

chains = jim.get_samples()

parameter_labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "q": r"$q$",
    "s1_z": r"$\chi_1$",
    "s2_z": r"$\chi_2$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
}

if not SMOKE:
    fig = corner.corner(
        np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
        labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
    )
    fig.savefig(Path(__file__).parent / "GW170817_mlgw_bns_heterodyne.png")
