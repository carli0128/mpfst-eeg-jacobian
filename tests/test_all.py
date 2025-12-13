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
from mpfst_eeg_jacobian.occupant_fields import compute_occupant_fields
from mpfst_eeg_jacobian.illusions_field import fractional_operator_fft, simulate_d
from mpfst_eeg_jacobian.meltdownfrac import meltdown_indicator, meltdownFrac_windowed
from mpfst_eeg_jacobian.meltdown_events import extract_meltdown_events


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


def test_occupant_fields_band_mapping():
    fs = 400.0
    t = np.arange(0, 2.0, 1 / fs)
    freqs = [10.0, 6.0, 20.0, 40.0, 100.0]  # alpha, theta, beta, gamma, high-gamma
    eeg = []
    for f in freqs:
        eeg.append(np.sin(2 * np.pi * f * t))
    eeg = np.vstack(eeg)
    U, u_sum = compute_occupant_fields(eeg, fs=fs, aggregate=False)
    assert U.shape[0] == 5 and U.shape[1] == eeg.shape[0]
    assert u_sum.shape == (eeg.shape[0], eeg.shape[1])
    for i in range(5):
        ch_means = U[i].mean(axis=1)
        assert np.argmax(ch_means) == i


def test_fractional_operator_and_illusions_stability():
    fs = 200.0
    x = np.sin(2 * np.pi * 5 * np.arange(0, 2, 1 / fs))
    out0 = fractional_operator_fft(x, alpha=0.0, fs=fs)
    out1 = fractional_operator_fft(x, alpha=0.5, fs=fs)
    assert np.all(np.isfinite(out0))
    assert np.all(np.isfinite(out1))
    assert np.linalg.norm(out0 - out1) > 1e-6
    d = simulate_d(out1, alpha=0.2, lam=0.1, sigma=0.05, fs=fs)
    assert d.shape == out1.shape
    assert np.all(np.isfinite(d))


def test_meltdownfrac_definition():
    S = np.array([0, 1, 2, 5, 10, 2, 1], float)
    Mth = 10.0
    ind = meltdown_indicator(S, Mth, frac=0.8)
    expected_indicator = np.array([0, 0, 0, 0, 1, 0, 0], float)
    assert np.array_equal(ind, expected_indicator)
    frac, times = meltdownFrac_windowed(ind, window=3, step=1)
    assert np.allclose(frac, np.array([0, 0, 1 / 3, 1 / 3, 1 / 3]))
    assert times[0] == 0


def test_meltdown_event_extraction():
    fs = 1.0
    S = np.array([1, 6, 7, 2, 0, 8, 9, 10], float)
    events = extract_meltdown_events(S, fs=fs, Mth=10.0, frac=0.5)
    assert events["start"].shape[0] == 2
    assert np.allclose(events["start"], [1.0, 5.0])
    assert np.allclose(events["end"], [2.0, 7.0])
    assert np.allclose(events["duration"], [2.0, 3.0])
    assert events["size"][0] > 0 and events["size"][1] > events["size"][0]
