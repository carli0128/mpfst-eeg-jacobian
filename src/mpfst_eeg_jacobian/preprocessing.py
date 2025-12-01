"""Lightweight EEG-style preprocessing utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .utils import zscore


CANONICAL_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


def bandpass_filter(x: np.ndarray, fs: float, fmin: float, fmax: float, order: int = 4):
    """Zero-phase bandpass filter using Butterworth + filtfilt."""
    x = np.asarray(x, float)
    nyq = 0.5 * fs
    eps = 1e-6
    low = max(fmin / nyq, eps)
    high = min(fmax / nyq, 1.0 - eps)
    if not 0 < low < high < 1:
        raise ValueError(
            f"Invalid band ({fmin}, {fmax}) for sampling rate {fs}. Ensure 0 < fmin < fmax < fs/2."
        )
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, x)


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """Return magnitude of analytic signal (Hilbert envelope)."""
    return np.abs(signal.hilbert(np.asarray(x, float)))


def band_envelope(
    x: np.ndarray,
    fs: float,
    band: str,
    order: int = 4,
    zscore_output: bool = True,
) -> np.ndarray:
    """Compute band-limited analytic envelope for a 1D signal."""
    if band not in CANONICAL_BANDS:
        raise ValueError(f"Unknown band '{band}'. Known: {list(CANONICAL_BANDS)}")
    fmin, fmax = CANONICAL_BANDS[band]
    xf = bandpass_filter(x, fs, fmin, fmax, order=order)
    env = analytic_envelope(xf)
    if zscore_output:
        env = zscore(env)
    return env


def mean_reference(eeg: np.ndarray, axis: int = 0) -> np.ndarray:
    """Subtract the mean across channels along the given axis."""
    eeg = np.asarray(eeg, float)
    m = eeg.mean(axis=axis, keepdims=True)
    return eeg - m
