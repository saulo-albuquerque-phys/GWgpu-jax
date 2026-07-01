# GWgpu-jax

GWgpu_jax provides a user-friendly interface between JAX waveform generators and JAX sampling algorithms for GPU-accelerated parameter estimation in gravitational wave data analysis. It supplies JAX likelihoods, detector-network construction, and real-data import pipelines.

## Waveform generators

| Generator | Model | Domain | Source type | Reference |
|---|---|---|---|---|
| [ripplegw](https://github.com/tedwards2412/ripple) | IMRPhenomD | frequency | BBH | [Phys.Rev.D 106 (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.044029) |
| mlgw_bns_jax | MLGW-BNS | frequency | BNS | [Phys. Rev. D 107 (2023)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.084037) |
| mlgw_bbh_jax | SEOBNRv5HM (model_4) | time | BBH | [Phys.Rev.D 108 (2023)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.124035) |

mlgw_bns_jax and mlgw_bbh_jax live under `gwgpu_jax/mlgw_jax/` and are loaded via `sys.path` — no separate pip install is required.

## Sampler

[blackjax-ns](https://github.com/handley-lab/blackjax) — a nested-sampling fork of BlackJAX, pinned to commit `dedbf11da33eb5ca286f6731e2c51f2b254b953f`.  
**Do not upgrade to a later commit** — the `PartitionedState` API was removed in subsequent commits.

## Installation

### Requirements

- Python 3.10 / 3.11 / 3.12
- Git

### Install from GitHub (pip — recommended)

The repo is currently **private**, so a [GitHub Personal Access Token](https://github.com/settings/tokens?type=beta) with `Contents: read-only` is required. Embed it once into the URL:

```bash
# Set the token as a shell variable (don't echo it into your history).
export GH_TOKEN=$(< ~/.config/gwgpu_jax_pat.txt)   # or read -s -p "PAT: " GH_TOKEN

# CPU only (laptops, macOS, plain Linux)
pip install "gwgpu_jax @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"

# + real-data ingest (gwpy + GWOSC)
pip install "gwgpu_jax[data] @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"

# + NVIDIA GPU (CUDA 12)
pip install "gwgpu_jax[gpu,data] @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"

# + heavy ML waveform models (TensorFlow + tf2jax for SEOBNRv5HM / mlgw_bns_jax)
pip install "gwgpu_jax[mlgw,data] @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"
```

> When the repo becomes public, drop `$GH_TOKEN@` from every URL.

### Google Colab quickstart (GPU runtime)

Colab GPU runtimes ship with a CUDA-enabled JAX, so the `gpu` extra is **not** needed. Two ways to provide the token:

- **Colab Secrets** (recommended) — key icon 🔑 in the sidebar → add a secret called `GH_TOKEN` → toggle *Notebook access*.
- One-off `getpass.getpass()` prompt — the cell below falls back to it if no secret is set.

```python
import os, getpass

GH_TOKEN = None
try:
    from google.colab import userdata
    GH_TOKEN = userdata.get("GH_TOKEN")
except Exception:
    pass
if not GH_TOKEN:
    GH_TOKEN = getpass.getpass("GitHub PAT: ")
os.environ["GH_TOKEN"] = GH_TOKEN

!pip install -q "gwgpu_jax[data] @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"

del os.environ["GH_TOKEN"]; del GH_TOKEN

import jax, gwgpu_jax
print(jax.devices())                       # → [CudaDevice(id=0), …]
grid = gwgpu_jax.TimeFrequencyGrid(4.0, 2048.0, f_min=20.0, f_max=512.0)
net  = gwgpu_jax.Network.from_names(["H1", "L1"], grid)
gwgpu_jax.compat.attach_event_to_network(net, "GW150914", estimate_psd=True)
```

The full ready-to-run version is in [`examples/gwgpu_jax_colab_pe.ipynb`](examples/gwgpu_jax_colab_pe.ipynb).

### Editable local install (development)

```bash
git clone https://github.com/saulo-albuquerque-phys/GWgpu-jax.git
cd GWgpu-jax
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

# GWgpu_jax in editable mode
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

