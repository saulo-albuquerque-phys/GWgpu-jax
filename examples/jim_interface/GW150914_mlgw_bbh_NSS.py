"""GW150914 with Jim's BlackJAX NSS nested sampler and the mlgw_bbh surrogate.

Jim's own ``examples/GW150914_NSS.py`` with the waveform swapped for
``MLGWBBHJimWaveform`` (SEOBNRv5HM surrogate, (2, 2) mode).  See README.md in
this folder for the environment.  Set ``JIM_SMOKE=1`` for a tiny run that only
checks the plumbing.
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
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.helpers import (
    get_default_likelihood_transforms,
    get_default_periodic,
    get_default_sample_transforms,
)
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.samplers.config import BlackJAXNSSConfig

from gwgpu_jax.gwgpu_jax_jim_interface import MLGWBBHJimWaveform

SMOKE = os.environ.get("JIM_SMOKE") == "1"

# ----------------------------------------------------------------------
# Fetch data
# ----------------------------------------------------------------------

gps = 1126259462.4
duration = 4.0
start = gps + 2.0 - duration
end = start + duration

psd_start = start - 2048
psd_end = start

fmin = 20.0
fmax = 896.0

ifos = [get_H1(), get_L1()]

for ifo in ifos:
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)

    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    ifo.set_psd(psd_data.to_psd(nperseg=int(data.duration * data.sampling_frequency)))

# ----------------------------------------------------------------------
# Waveform model
# ----------------------------------------------------------------------

# The surrogate is time-domain: give it the duration and sampling rate of the
# data so that Jim's frequencies fall on its FFT grid.  modes=None would sum
# all seven modes of model_4 (then drop phase_marginalization below and add a
# phase_c prior).
waveform = MLGWBBHJimWaveform(
    duration=duration,
    sampling_frequency=float(data.sampling_frequency),
    modes=(2, 2),
)

# ----------------------------------------------------------------------
# Prior (inside the surrogate's range: q >= 0.1, |s_z| <= 0.9)
# ----------------------------------------------------------------------

prior = CombinePrior(
    [
        UniformPrior(20.0, 40.0, parameter_names=["M_c"]),
        UniformPrior(0.125, 1.0, parameter_names=["q"]),
        UniformPrior(-0.9, 0.9, parameter_names=["s1_z"]),
        UniformPrior(-0.9, 0.9, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"]),
        UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
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

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    f_min=fmin,
    f_max=fmax,
    phase_marginalization=True,
)

# ----------------------------------------------------------------------
# Sample
# ----------------------------------------------------------------------

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    periodic=get_default_periodic(prior, sample_transforms),
    sampler_config=BlackJAXNSSConfig(
        n_live=100 if SMOKE else 2000,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=2 if SMOKE else 20,
        termination_dlogz=50.0 if SMOKE else 0.1,
    ),
)

start_time = time.time()
jim.sample()
end_time = time.time()
print(f"Sampling took {(end_time - start_time) / 60:.2f} mins")

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

diagnostics = jim.get_diagnostics()
print(f"log Z = {diagnostics['log_Z']:.2f} ± {diagnostics['log_Z_error']:.2f}")
print(f"Likelihood evaluations: {diagnostics['n_likelihood_evaluations']:,}")

chains = jim.get_samples()

parameter_labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "q": r"$q$",
    "s1_z": r"$s_{1,z}$",
    "s2_z": r"$s_{2,z}$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
}

if not SMOKE:
    fig = corner.corner(
        np.stack([chains[key] for key in jim.prior.parameter_names]).T,
        labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
    )
    fig.savefig(Path(__file__).parent / "GW150914_mlgw_bbh_NSS.png")
