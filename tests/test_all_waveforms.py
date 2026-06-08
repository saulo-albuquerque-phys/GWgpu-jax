"""
test_all_waveforms.py
=====================
Smoke-test that all three waveform generators run in the same environment.
"""

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "gwgpu_jax/mlgw_jax/mlgw_bbh_jax")

import jax
import jax.numpy as jnp

# ─── 1. ripplegw – IMRPhenomD (frequency-domain, BBH) ────────────────────────
print("=" * 60)
print("1. ripplegw  |  IMRPhenomD  |  frequency-domain  |  BBH")
print("=" * 60)

from ripplegw.waveforms import IMRPhenomD

theta_ripple = jnp.array([28.3, 0.247, 0.5, -0.3, 440.0, 7.0, 0.0])
f = jnp.linspace(20.0, 1024.0, 4096)
hp_r, hc_r = IMRPhenomD.gen_IMRPhenomD_hphc(f, theta_ripple, f_ref=20.0)
print(f"  hp shape : {hp_r.shape}")
print(f"  hc shape : {hc_r.shape}")
print("  STATUS   : OK\n")

# ─── 2. mlgw_bns_jax – BNS (frequency-domain) ────────────────────────────────
print("=" * 60)
print("2. mlgw_bns_jax  |  ML-BNS  |  frequency-domain  |  BNS")
print("=" * 60)

sys.path.insert(0, "gwgpu_jax/mlgw_jax/mlgw_bns_jax")
jax.config.update("jax_enable_x64", True)
from jax_import_n_predict import load_predict  # noqa: E402

model_path = "gwgpu_jax/mlgw_jax/mlgw_bns_jax/mlgw_bns_jax_model.h5"
predict_fn = load_predict(model_path)

freqs = jnp.linspace(20.0, 1024.0, 2048)
# predict(params[q,l1,l2,chi1,chi2], freqs, total_mass, distance_mpc, incl)
params = jnp.array([1.2, 400.0, 300.0, 0.02, -0.01])
hp_b, hc_b = predict_fn(params, freqs,
                         jnp.array(2.8), jnp.array(100.0), jnp.array(0.3))
print(f"  hp shape : {hp_b.shape}")
print(f"  hc shape : {hc_b.shape}")
print("  STATUS   : OK\n")

# ─── 3. mlgw_bbh_jax – SEOBNRv5HM (time-domain, BBH) ────────────────────────
print("=" * 60)
print("3. mlgw_bbh_jax  |  SEOBNRv5HM  |  time-domain  |  BBH")
print("=" * 60)

from mlgw.GW_generator import GW_generator  # noqa: E402

gen = GW_generator(4)  # model_4 = SEOBNRv5HM (Keras 3.13.2, modes 22 21 32 33 43 44 55)
# [m1 [M☉], m2 [M☉], s1z, s2z]
theta_bbh = jnp.array([30.0, 10.0, 0.5, -0.3])
t_grid = jnp.linspace(-8.0, 0.005, 8192)
hp_bbh, hc_bbh = gen.get_WF(theta_bbh, t_grid)
print(f"  hp shape : {hp_bbh.shape}")
print(f"  hc shape : {hc_bbh.shape}")
print("  STATUS   : OK\n")

# ─── 4. blackjax-ns – nested sampling (imports only) ─────────────────────────
print("=" * 60)
print("4. blackjax-ns  |  nested sampling  |  sampler")
print("=" * 60)

import blackjax
from blackjax.ns.utils import finalise
from blackjax import nss

print(f"  version  : {blackjax.__version__}")
print("  nss      : OK")
print("  finalise : OK")
print("  STATUS   : OK\n")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("ALL FOUR COMPONENTS WORKING IN THE SAME ENVIRONMENT")
print("=" * 60)
