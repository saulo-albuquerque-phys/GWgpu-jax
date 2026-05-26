# GWjax

This code provides a user-friendly interface between JAX waveform generators and JAX sampling algorithms for GPU-accelerated parameter estimation in gravitational wave data analysis. We provide JAX likelihoods written for the JAX waveform generators, build the detector networks, and connect them with well-established pipelines for real-data importing.

## Installation

GWjax depends on [ripplegw](https://github.com/tedwards2412/ripple) for JAX waveform models and on the [blackjax-ns](https://github.com/handley-lab/blackjax) nested-sampling fork (pinned to a specific commit — see note below).

### Requirements

- Python 3.11
- Git
- For GPU usage: Linux x86_64 with CUDA 12.x drivers and an NVIDIA GPU

### Option 1 — automated setup scripts (recommended)

**CPU / macOS / Linux (no GPU)**

```bash
bash setup_env_cpu.sh          # creates ./venv by default
source venv/bin/activate
```

**CUDA 12 / Linux + NVIDIA GPU**

```bash
bash setup_env_cuda.sh         # creates ./venv_cuda by default
source venv_cuda/bin/activate
```

Both scripts create the virtual environment, install all dependencies in the correct order, and run a smoke test to confirm that `ripplegw` and `blackjax-ns` are working.

### Option 2 — manual setup

```bash
# 1. Create and activate a Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 2. Install JAX — choose ONE of the following:
pip install "jax[cpu]==0.4.31" "jaxlib==0.4.31"       # CPU / macOS
pip install "jax[cuda12]==0.4.31"                       # CUDA 12 (Linux)

# 3. Install the pinned blackjax nested-sampling fork
#    (do NOT use a later commit — the PartitionedState API was removed)
pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git@dedbf11da33eb5ca286f6731e2c51f2b254b953f"

# 4. Install the remaining dependencies
pip install -r requirements-cpu.txt    # CPU
pip install -r requirements-cuda.txt   # CUDA

# 5. Install GWjax in editable mode
pip install -e .
```

### Verifying the installation

```python
import jax, jax.numpy as jnp
from ripplegw.waveforms import IMRPhenomD
import blackjax
from blackjax.ns.utils import finalise
from blackjax import nss
import anesthetic

# Generate a test waveform
theta = jnp.array([30.0, 0.25, 0.0, 0.0, 400.0, 0.0, 0.0, 20.0])
freqs = jnp.linspace(20.0, 1024.0, 1000)
hp, hc = IMRPhenomD.gen_IMRPhenomD_hphc(freqs, theta, 20.0)
print(f"JAX backend: {jax.default_backend()}")
print(f"ripplegw OK — hp shape: {hp.shape}")
print(f"blackjax-ns OK — version: {blackjax.__version__}")
```

### GPU memory (CUDA only)

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8   # use 80% of GPU memory
```

