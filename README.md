# GWjax

GWjax provides a user-friendly interface between JAX waveform generators and JAX sampling algorithms for GPU-accelerated parameter estimation in gravitational wave data analysis. It supplies JAX likelihoods, detector-network construction, and real-data import pipelines.

## Waveform generators

| Generator | Model | Domain | Source type | Reference |
|---|---|---|---|---|
| [ripplegw](https://github.com/tedwards2412/ripple) | IMRPhenomD | frequency | BBH | [Phys.Rev.D 106 (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.044029) |
| mlgw_bns_jax | ML-GW-BNS | frequency | BNS | local subdir |
| mlgw_bbh_jax | SEOBNRv5HM (model_4) | time | BBH | [Phys.Rev.D 108 (2023)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.124035) |

mlgw_bns_jax and mlgw_bbh_jax live under `gwjax/mlgw_jax/` and are loaded via `sys.path` — no separate pip install is required.

## Sampler

[blackjax-ns](https://github.com/handley-lab/blackjax) — a nested-sampling fork of BlackJAX, pinned to commit `dedbf11da33eb5ca286f6731e2c51f2b254b953f`.  
**Do not upgrade to a later commit** — the `PartitionedState` API was removed in subsequent commits.

## Installation

### Requirements

- Python 3.10 / 3.11 / 3.12
- Git

### Install from GitHub (pip — recommended)

```bash
# CPU only (laptops, macOS, plain Linux)
pip install "gwjax @ git+https://github.com/saulo-albuquerque-phys/GWjax.git"

# + real-data ingest (gwpy + GWOSC)
pip install "gwjax[data] @ git+https://github.com/saulo-albuquerque-phys/GWjax.git"

# + NVIDIA GPU (CUDA 12)
pip install "gwjax[gpu,data] @ git+https://github.com/saulo-albuquerque-phys/GWjax.git"

# + heavy ML waveform models (TensorFlow + tf2jax for SEOBNRv5HM / mlgw_bns_jax)
pip install "gwjax[mlgw,data] @ git+https://github.com/saulo-albuquerque-phys/GWjax.git"
```

### Google Colab quickstart (GPU runtime)

Colab GPU runtimes ship with a CUDA-enabled JAX, so the `gpu` extra is **not** needed — just install the package itself:

```python
!pip install -q "gwjax[data] @ git+https://github.com/saulo-albuquerque-phys/GWjax.git"

import jax
print(jax.devices())                       # → [CudaDevice(id=0), …]

import gwjax
grid = gwjax.TimeFrequencyGrid(duration=4.0, sampling_rate=2048.0,
                               f_min=20.0, f_max=512.0)
net  = gwjax.Network.from_names(["H1", "L1"], grid)
net.generate_noise(seed=0)

# Real GW150914 strain from GWOSC + Welch PSD from off-source noise
gwjax.compat.attach_event_to_network(net, "GW150914", estimate_psd=True)
```

### Editable local install (development)

```bash
git clone https://github.com/saulo-albuquerque-phys/GWjax.git
cd GWjax
pip install -e ".[data,mlgw,samplers]"     # all extras except [gpu]
```

### Bash setup scripts (legacy)

The `setup_env_*.sh` scripts still work and build pinned environments for
reproducibility — useful if you need to match a specific JAX / TF / blackjax
version combination exactly.

### CPU-only environment (ripplegw + blackjax-ns only)

```bash
bash setup_env_cpu.sh            # creates ./venv by default
source venv/bin/activate
```

### CUDA 12 environment

```bash
bash setup_env_cuda.sh           # creates ./venv_cuda by default
source venv_cuda/bin/activate
```

### Manual setup

```bash
python3.11 -m venv venv_full
source venv_full/bin/activate
pip install --upgrade pip

# JAX (CPU)
pip install "jax[cpu]==0.4.31" "jaxlib==0.4.31"

# Pinned blackjax nested-sampling fork
pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git@dedbf11da33eb5ca286f6731e2c51f2b254b953f"

# TensorFlow 2.13 + tf2jax (required for mlgw_bbh_jax)
pip install "tensorflow==2.13.0" "tf2jax==0.3.6"

# Pin numpy (TF may downgrade it; 1.26.4 is required by JAX 0.4.31)
pip install "numpy==1.26.4"

# Remaining dependencies
pip install -r requirements-full.txt

# GWjax in editable mode
pip install -e .
```

### Verifying the installation

```bash
python test_all_waveforms.py 2>/dev/null
```

Expected output:

```
============================================================
1. ripplegw  |  IMRPhenomD  |  frequency-domain  |  BBH
============================================================
  hp shape : (4096,)   hc shape : (4096,)   STATUS : OK

============================================================
2. mlgw_bns_jax  |  ML-BNS  |  frequency-domain  |  BNS
============================================================
  hp shape : (2048,)   hc shape : (2048,)   STATUS : OK

============================================================
3. mlgw_bbh_jax  |  SEOBNRv5HM  |  time-domain  |  BBH
============================================================
  hp shape : (8192,)   hc shape : (8192,)   STATUS : OK

============================================================
4. blackjax-ns  |  nested sampling  |  sampler
============================================================
  version  : 0.1.dev707+gdedbf11da   STATUS : OK

============================================================
ALL FOUR COMPONENTS WORKING IN THE SAME ENVIRONMENT
============================================================
```

### GPU memory (CUDA only)

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8   # use 80 % of GPU memory
```

## mlgw_bbh_jax — notes on model loading

The mlgw_bbh_jax models are saved in Keras 3.x `.keras` format.  
The bundled `NN_model.py` has been patched to load them in a Keras 2.13.1 environment (which ships with TF 2.13.0) by parsing the layer specs directly rather than calling `from_config`.  
The active model is **model_4** (SEOBNRv5HM, modes 22 21 32 33 43 44 55, $q \in [1,10]$, $\chi_{1,2} \in [-0.9, 0.9]$).

