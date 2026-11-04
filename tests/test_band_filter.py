"""The optional data band-limiting filter must leave the analysis band untouched.

Regression test for the Nov 2026 finding: gwpy's default highpass (Chebyshev-I, 2 dB ripple,
filtfilt) rippled between 0.63 and 1.0 in-band, rescaling the signal relative to the unfiltered
PSD and templates (GW170817 whitened to 0.5-0.6 between 40 and 200 Hz).
"""
import numpy as np
import pytest

gwpy = pytest.importorskip("gwpy")
from gwpy.timeseries import TimeSeries
from scipy.signal import welch

from gwgpu_jax.gwgpu_jax_compatibility import _flat_band_filter


def _gain(low, high, fs=4096.0, seconds=512):
    rng = np.random.default_rng(3)
    x = TimeSeries(rng.standard_normal(int(seconds * fs)), sample_rate=fs)
    y = _flat_band_filter(x, low, high)
    f, px = welch(x.value, fs=fs, nperseg=int(16 * fs))
    _, py = welch(y.value, fs=fs, nperseg=int(16 * fs))
    return f, np.sqrt(py / px)


@pytest.mark.parametrize("low,high", [(23.0, 2048.0), (20.0, 500.0)])
def test_flat_in_band(low, high):
    f, g = _gain(low, high)
    band = (f >= low) & (f <= min(high, 0.95 * 2048.0))
    # Welch-ratio noise on 512 s of white noise is ~1%; the filter itself is flat to 0.15%
    assert np.all(np.abs(g[band] - 1.0) < 0.03), g[band].min()
    assert abs(np.median(g[band]) - 1.0) < 0.005


def test_suppresses_below_band():
    f, g = _gain(23.0, 2048.0)
    assert np.all(g[(f > 1.0) & (f < 8.0)] < 1e-3)


def test_input_not_modified():
    x = TimeSeries(np.random.default_rng(0).standard_normal(4096 * 8), sample_rate=4096.0)
    x0 = x.value.copy()
    _flat_band_filter(x, 23.0, 2048.0)
    assert np.array_equal(x.value, x0)
