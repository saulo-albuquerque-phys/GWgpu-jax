# mlgw_bbh / mlgw_bns in Jim

`gwgpu_jax.gwgpu_jax_jim_interface` exposes the two surrogates bundled with
GWgpu_jax as waveform models of [Jim](https://github.com/GW-JAX-Team/Jim)
(`jimgw >= 0.6`):

| class | registry name | model | extra parameters |
|---|---|---|---|
| `MLGWBBHJimWaveform(duration, sampling_frequency, modes=None)` | `MLGW_BBH` | mlgw_bbh model_4 (SEOBNRv5HM, aligned spins, modes 22 21 32 33 43 44 55) | – |
| `MLGWBNSJimWaveform()` | `MLGW_BNS` | mlgw_bns (TEOBResumS, (2, 2) mode, tides) | `lambda_1`, `lambda_2` |

Both take Jim's parameter names (`M_c`, `eta`, `s1_z`, `s2_z`, `d_L`,
`phase_c`, `iota`) and return `{"p": h_plus, "c": h_cross}` in the LAL
convention with the merger at t = 0, so they are drop-in replacements for
`RippleIMRPhenomD` and friends:

```python
from gwgpu_jax.gwgpu_jax_jim_interface import MLGWBBHJimWaveform, MLGWBNSJimWaveform

waveform = MLGWBBHJimWaveform(duration=4.0, sampling_frequency=4096.0, modes=(2, 2))
likelihood = TransientLikelihoodFD(ifos, waveform=waveform, f_min=20.0, f_max=896.0,
                                   phase_marginalization=True)

waveform = MLGWBNSJimWaveform()
likelihood = HeterodynedTransientLikelihoodFD(ifos, waveform=waveform, n_bins=1000, ...)
```

After the import they are also reachable as
`ripplegw.waveform("MLGW_BBH", duration=..., sampling_frequency=...)` and
`ripplegw.waveform("MLGW_BNS")`.  The `jim-run` command line builds waveforms
from its own fixed table and does not see them.

## Things to know

* **mlgw_bbh is a time-domain model.**  The adapter evaluates it on its own
  time grid and Fourier transforms it, so give it the duration and sampling
  rate of the data.  Frequencies on the FFT grid (`TransientLikelihoodFD`,
  `inject_signal`) cost one rFFT; other frequencies (the bin edges of
  `HeterodynedTransientLikelihoodFD`) are served by a direct transform that
  needs an `n_freq × n_samples` complex matrix — fine for a few thousand bin
  edges, not for a full frequency series.
* **Higher modes and phase marginalisation.**  With `modes=None` the BBH
  model sums all seven modes and `phase_c` is no longer an overall phase:
  sample `phase_c` instead of using `phase_marginalization` (Jim logs a
  warning if you do not).  `modes=(2, 2)` and the BNS model are exact.
* **Stay inside the training range.**  BBH: `q = m2/m1 >= 0.1`,
  `|s_z| <= 0.9`.  BNS: `lambda` in [0, 5000], small aligned spins, and
  frequencies between 3.6 and 2926 Hz × (2.8 M_sun / M).

## Environment

Jim 0.6 needs Python >= 3.12, `jax >= 0.8.2`, `ripplegw >= 0.5` and
`blackjax >= 1.7.1`, while GWgpu_jax pins a blackjax fork, so install Jim
first and GWgpu_jax without its dependencies:

```bash
pip install "jimgw>=0.6"
pip install h5py                                   # mlgw_bns
pip install "tensorflow>=2.16" "tf2jax>=0.3.6" joblib tqdm   # mlgw_bbh
pip install --no-deps -e /path/to/GWjax            # or put the repo on PYTHONPATH
```

Tested with jax 0.12.0, jimgw 0.6.1, ripplegw 0.5.1, tensorflow 2.21, tf2jax
0.3.8 on Linux.  Two platform notes:

* Intel Macs: there is no TensorFlow build compatible with `jax >= 0.9`, so
  only the BNS adapter can be used there (jax/jaxlib from conda-forge).
* CPUs without AVX2: the PyPI jaxlib wheels do not run; take
  `jax jaxlib tensorflow` from conda-forge and `pip install jimgw tf2jax`
  on top.

## Files

* `GW150914_mlgw_bbh_NSS.py` — Jim's GW150914 nested-sampling example with
  mlgw_bbh.
* `GW170817_mlgw_bns_heterodyne.py` — Jim's GW170817 relative-binning example
  with mlgw_bns.
* `../../tests/test_jim_interface.py` — convention and likelihood tests
  (`pytest tests/test_jim_interface.py` in a Jim environment).

`JIM_SMOKE=1 python <script>` runs either example with a tiny sampler
configuration to check the installation.
