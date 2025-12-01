"""Minimal smoke tests for the MPFST–EEG–Jacobian kit.

These tests only use synthetic data and are designed to be fast; they mainly
check that the public API runs without raising and that shapes are consistent.
"""

from __future__ import annotations

import numpy as np

from mpfst_eeg_jacobian.coherence import windowed_exponents
from mpfst_eeg_jacobian.avalanches import coherence_avalanches_from_signal
from mpfst_eeg_jacobian.jacobian import estimate_local_jacobian, jacobian_spectrum_metrics
from mpfst_eeg_jacobian.pipeline import run_eeg_to_jacobian_avalanches


def _synthetic_envelope(fs: float, seconds: float = 10.0):
    rng = np.random.default_rng(1)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    carrier = np.sin(2 * np.pi * 15.0 * t)
    envelope = 1.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    noise = 0.3 * rng.standard_normal(n)
    return envelope * carrier + noise


def test_windowed_exponents_smoke():
    fs = 200.0
    x = _synthetic_envelope(fs, seconds=20.0)
    times, mu, gamma, H, m = windowed_exponents(x, fs, window_sec=5.0, step_sec=1.0)
    assert times.shape == mu.shape == gamma.shape == H.shape == m.shape
    assert np.all(np.isfinite(m))


def test_coherence_avalanches_smoke():
    fs = 200.0
    x = _synthetic_envelope(fs, seconds=30.0)
    result = coherence_avalanches_from_signal(
        x,
        fs,
        window_sec=5.0,
        step_sec=1.0,
    )
    # At least one window, exponents finite
    assert result.mu.size > 0
    assert np.all(np.isfinite(result.m_l))
    # Avalanches may be empty but code should still return a dict
    assert isinstance(result.avalanches, dict)


def test_jacobian_and_pipeline_smoke():
    fs = 200.0
    n_channels = 4
    seconds = 20.0
    n_samples = int(seconds * fs)
    rng = np.random.default_rng(2)
    eeg = rng.standard_normal((n_channels, n_samples))
    out = run_eeg_to_jacobian_avalanches(
        eeg=eeg,
        fs=fs,
        channel_indices=[0, 1, 2],
        band="beta",
        window_sec=5.0,
        step_sec=1.0,
    )
    J = out["jacobian"]
    assert J.shape[0] == J.shape[1]
    metrics = jacobian_spectrum_metrics(J)
    assert "max_real_eig" in metrics
    assert np.isfinite(metrics["max_real_eig"])
