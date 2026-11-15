"""Tests for gwgpu_jax.gwgpu_jax_jim_interface (mlgw_bbh / mlgw_bns as Jim waveforms).

Needs a Jim environment (jimgw >= 0.6, ripplegw >= 0.5, Python >= 3.12); the
whole module is skipped otherwise.  The tests that load the real mlgw_bbh
weights are skipped when TensorFlow / tf2jax are not installed.

    pytest tests/test_jim_interface.py
"""

import importlib.util

import numpy as np
import pytest

pytest.importorskip("ripplegw.interfaces", reason="needs ripplegw >= 0.5 (Jim >= 0.6)")
pytest.importorskip("jimgw", reason="needs jimgw >= 0.6")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import ripplegw
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    TransientLikelihoodFD,
)

from gwgpu_jax.gwgpu_jax_jim_interface import MLGWBBHJimWaveform, MLGWBNSJimWaveform

HAS_TF = all(importlib.util.find_spec(m) for m in ("tensorflow", "tf2jax", "joblib"))
needs_tf = pytest.mark.skipif(not HAS_TF, reason="mlgw_bbh needs tensorflow + tf2jax")

T, FS = 4.0, 2048.0
F0, TAU = 150.0, 0.02
BBH = dict(M_c=30.0, eta=0.24, s1_z=0.1, s2_z=-0.2, d_L=400.0, phase_c=0.4, iota=0.3)
BNS = dict(M_c=1.1975, eta=0.2485, s1_z=0.01, s2_z=-0.02, lambda_1=400.0,
           lambda_2=600.0, d_L=40.0, phase_c=0.3, iota=0.6)
SKY = dict(ra=1.2, dec=-0.4, psi=0.7, t_c=0.01)


class SineGaussian:
    """Stand-in for MLGWBBHGenerator with a known Fourier transform (peak at t = 0)."""

    available_modes = [(2, 2), (3, 3)]

    def generate(self, theta, t, modes=None):
        env = 1e-21 * theta[0] / theta[4] * jnp.exp(-0.5 * (t / TAU) ** 2)
        arg = 2 * jnp.pi * F0 * t + theta[6]
        return env * jnp.cos(arg), env * jnp.sin(arg)


def sine_gaussian_fd(f, params):
    m1 = 0.5 * params["M_c"] / params["eta"] ** 0.6 * (1 + np.sqrt(1 - 4 * params["eta"]))
    a = 1e-21 * m1 / params["d_L"] * TAU * np.sqrt(2 * np.pi) / 2
    g = lambda x: np.exp(-0.5 * (2 * np.pi * TAU * x) ** 2)
    ph = params["phase_c"]
    return a * (g(f - F0) * np.exp(1j * ph) + g(f + F0) * np.exp(-1j * ph))


def rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


@pytest.fixture(scope="module")
def mock_bbh():
    return MLGWBBHJimWaveform(T, FS, generator=SineGaussian())


@pytest.fixture(scope="module")
def bns():
    return MLGWBNSJimWaveform()


def make_ifos(waveform, inj, duration, f_min, f_max):
    ifos = [get_H1(), get_L1()]
    for i, ifo in enumerate(ifos):
        ifo.load_and_set_psd()
        ifo.simulate_and_set_noise(
            duration=duration, sampling_frequency=2 * f_max, trigger_time=1126259462.4,
            f_min=f_min, f_max=f_max, rng_key=jax.random.key(i),
        )
        ifo.inject_signal(waveform, inj, allow_wraparound=True)
    return ifos


# ── time-domain -> frequency-domain machinery (no model weights needed) ──────

def test_bbh_fft_matches_analytic_transform(mock_bbh):
    """Merger at t = 0 and LAL sign convention, on the FFT grid."""
    f = jnp.arange(int(20 * T), int(896 * T) + 1) / T
    assert mock_bbh._use_fft(f)
    assert rel(mock_bbh(f, BBH)["p"], sine_gaussian_fd(np.asarray(f), BBH)) < 1e-5


def test_bbh_dtft_equals_fft_on_grid_and_works_off_grid(mock_bbh):
    f = jnp.arange(int(20 * T), int(896 * T) + 1) / T
    dtft = MLGWBBHJimWaveform(T, FS, generator=SineGaussian(), frequency_mode="dtft")
    assert rel(dtft(f, BBH)["p"], mock_bbh(f, BBH)["p"]) < 1e-12
    off = jnp.array([33.3, 101.07, 149.99, 300.123])
    assert not mock_bbh._use_fft(off)
    assert rel(mock_bbh(off, BBH)["p"], sine_gaussian_fd(np.asarray(off), BBH)) < 1e-5


def test_bbh_traced_frequency_falls_back_to_dtft(mock_bbh):
    f = jnp.arange(int(20 * T), int(200 * T) + 1) / T
    traced = jax.jit(lambda ff: mock_bbh(ff, BBH)["p"])(f)
    assert rel(traced, mock_bbh(f, BBH)["p"]) < 1e-12


def test_bbh_rejects_bad_frequencies(mock_bbh):
    with pytest.raises(ValueError, match="outside"):
        mock_bbh(jnp.array([2000.0]), BBH)
    strict = MLGWBBHJimWaveform(T, FS, generator=SineGaussian(), frequency_mode="fft")
    with pytest.raises(ValueError, match="on the grid"):
        strict(jnp.array([33.3]), BBH)


def test_bbh_metadata_and_registry(mock_bbh):
    assert mock_bbh.waveform_metadata["domain"] == "FD"
    assert mock_bbh.waveform_metadata["highest_mode"] == 3
    only22 = MLGWBBHJimWaveform(T, FS, modes=(2, 2), generator=SineGaussian())
    assert only22.modes == (2, 2) and only22.waveform_metadata["highest_mode"] == 2
    assert {"MLGW_BBH", "MLGW_BNS"} <= set(ripplegw.list_waveforms(domain="FD"))


def test_bbh_through_jim_likelihoods(mock_bbh):
    inj = {**BBH, **SKY}
    ifos = make_ifos(mock_bbh, inj, T, 20.0, 1024.0)
    full = jax.jit(TransientLikelihoodFD(ifos, waveform=mock_bbh, f_min=20.0, f_max=1024.0).evaluate)
    het = jax.jit(HeterodynedTransientLikelihoodFD(
        ifos, waveform=mock_bbh, f_min=20.0, f_max=1024.0,
        reference_parameters=dict(inj), n_bins=200,
    ).evaluate)
    pert = dict(inj, d_L=440.0, iota=0.4)
    assert np.isfinite(float(full(inj)))
    assert abs(float(het(inj)) - float(full(inj))) < 1e-6
    assert abs(float(het(pert)) - float(full(pert))) < 1e-2


# ── mlgw_bns (real weights, h5py only) ───────────────────────────────────────

def test_bns_conventions_match_ripple(bns):
    ref = ripplegw.waveform("IMRPhenomD_NRTidalv2", f_ref=20.0)
    assert bns.parameter_names == ref.parameter_names
    f = jnp.arange(20 * 64, 1000 * 64 + 1) / 64.0
    h, r = bns(f, BNS), ref(f, BNS)
    hp, hc, rp = (np.asarray(x) for x in (h["p"], h["c"], r["p"]))
    assert np.isfinite(hp).all() and np.isfinite(hc).all()
    ci = np.cos(BNS["iota"])
    assert rel(hc, -1j * hp * ci / ((1 + ci**2) / 2)) < 1e-12           # h_cross convention
    assert abs(np.median(np.abs(hp) / np.abs(rp)) - 1) < 0.02            # amplitude / units
    # same Fourier sign and merger time: phase difference linear in f, slope < 10 ms
    dphi = np.unwrap(np.angle(hp * np.conj(rp)))
    slope, _ = np.polyfit(np.asarray(f), dphi, 1)
    assert abs(slope / (2 * np.pi)) < 1e-2
    # phase_c is an overall exp(2i phase_c), as in ripple
    for model, base in ((bns, hp), (ref, rp)):
        shifted = np.asarray(model(f, dict(BNS, phase_c=BNS["phase_c"] + 0.5))["p"])
        assert rel(shifted, base * np.exp(1j)) < 1e-9
    assert rel(bns.at_unit_distance(f, BNS)["p"] / BNS["d_L"], hp) < 1e-12


def test_bns_jit_grad_vmap(bns):
    f = jnp.linspace(30.0, 500.0, 512)
    power = lambda mc: jnp.sum(jnp.abs(bns(f, dict(BNS, M_c=mc))["p"]) ** 2)
    assert np.isfinite(float(jax.jit(jax.grad(power))(BNS["M_c"])))
    batch = jax.jit(jax.vmap(lambda mc: bns(f, dict(BNS, M_c=mc))["p"]))(jnp.linspace(1.19, 1.2, 4))
    assert batch.shape == (4, 512)


def test_bns_through_jim_likelihoods(bns):
    inj = {**BNS, **SKY}
    ifos = make_ifos(bns, inj, 32.0, 40.0, 1024.0)
    full = jax.jit(TransientLikelihoodFD(ifos, waveform=bns, f_min=40.0, f_max=1024.0).evaluate)
    het = jax.jit(HeterodynedTransientLikelihoodFD(
        ifos, waveform=bns, f_min=40.0, f_max=1024.0,
        reference_parameters=dict(inj), n_bins=300,
    ).evaluate)
    pert = dict(inj, M_c=inj["M_c"] * (1 + 1e-5), d_L=44.0, lambda_1=500.0)
    assert float(full(inj)) > float(full(dict(inj, M_c=inj["M_c"] * 1.01)))
    assert abs(float(het(inj)) - float(full(inj))) < 1e-3
    assert abs(float(het(pert)) - float(full(pert))) < 0.1


# ── mlgw_bbh (real weights: TensorFlow + tf2jax) ─────────────────────────────

@pytest.fixture(scope="module")
def bbh22():
    return MLGWBBHJimWaveform(T, FS, modes=(2, 2))


@needs_tf
def test_real_bbh_matches_gwgpu_jax_pipeline(bbh22):
    """Same spectrum as the GWgpu_jax route, up to the documented time reference."""
    from gwgpu_jax import TimeFrequencyGrid
    from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import hphc_td_to_fd

    grid = TimeFrequencyGrid(duration=T, sampling_rate=FS, mlgw_final_time=bbh22.final_time)
    assert rel(bbh22.times, grid.time_domain_array_mlgw) < 1e-14
    hp_td, hc_td = bbh22.time_domain(BBH)
    hp_ref, _ = hphc_td_to_fd(
        hp_td, hc_td, 1.0 / FS, alpha_left=bbh22.alpha_left, alpha_right=bbh22.alpha_right
    )
    f = jnp.arange(int(20 * T), int(896 * T) + 1) / T
    shift = np.exp(-2j * np.pi * np.asarray(f) * bbh22.final_time)
    assert rel(bbh22(f, BBH)["p"], np.asarray(hp_ref)[int(20 * T):int(896 * T) + 1] * shift) < 1e-10


@needs_tf
def test_real_bbh_conventions_match_ripple(bbh22):
    """Merger at t = 0, LAL sign / polarisation conventions, sensible match."""
    ref = ripplegw.waveform("IMRPhenomD", f_ref=20.0)
    f = jnp.arange(int(30 * T), int(400 * T) + 1) / T
    h, r = bbh22(f, BBH), ref(f, BBH)
    hp, hc, rp = (np.asarray(x) for x in (h["p"], h["c"], r["p"]))
    ci = np.cos(BBH["iota"])
    # exact only for an analytic FD model; ~6e-4 for the windowed TD transform
    assert rel(hc, -1j * hp * ci / ((1 + ci**2) / 2)) < 1e-2
    w = np.abs(rp)
    dphi = np.unwrap(np.angle(hp * np.conj(rp)))
    slope, _ = np.polyfit(np.asarray(f), dphi, 1, w=w)
    # match maximised over time and phase (white noise, 30-400 Hz)
    taus = np.linspace(-0.02, 0.02, 2001)
    cross = (np.conj(rp) * hp) @ np.exp(2j * np.pi * np.outer(np.asarray(f), taus))
    overlap = np.abs(cross).max() / np.sqrt(np.vdot(rp, rp).real * np.vdot(hp, hp).real)
    print("time offset [s]:", slope / (2 * np.pi), "match:", overlap,
          "amp ratio:", np.median(np.abs(hp) / np.abs(rp)))
    # IMRPhenomD's own peak-time convention differs from the true peak by a few
    # ms at this mass (measured offset 5.9 ms); a wrong time reference in the
    # adapter would show up as final_time = 120 ms.
    assert abs(slope / (2 * np.pi)) < 2e-2
    assert overlap > 0.97
    assert rel(bbh22.at_unit_distance(f, BBH)["p"] / BBH["d_L"], hp) < 1e-10


@needs_tf
def test_real_bbh_through_jim(bbh22):
    inj = {**BBH, **SKY}
    ifos = make_ifos(bbh22, inj, T, 20.0, 1024.0)
    like = TransientLikelihoodFD(ifos, waveform=bbh22, f_min=20.0, f_max=896.0)
    logl = jax.jit(jax.vmap(lambda mc: like.evaluate(dict(inj, M_c=mc))))(
        jnp.array([29.0, 30.0, 31.0])
    )
    assert np.isfinite(np.asarray(logl)).all() and int(np.argmax(logl)) == 1
    grad = jax.jit(jax.grad(lambda mc: like.evaluate(dict(inj, M_c=mc))))(30.0)
    assert np.isfinite(float(grad))
