"""High-level orchestration pipeline.

The main entry point is :func:`run_eeg_to_jacobian_avalanches`, which accepts
a simple NumPy EEG array and returns a dictionary with coherence exponents,
avalanches and Jacobian metrics.
"""

from __future__ import annotations

from typing import Dict, Any, Sequence

import numpy as np
from scipy.signal import detrend

from .preprocessing import mean_reference, band_envelope
from .avalanches import coherence_avalanches_from_signal
from .jacobian import build_bandpower_latent, estimate_local_jacobian, jacobian_spectrum_metrics


def run_eeg_to_jacobian_avalanches(
    eeg: np.ndarray,
    fs: float,
    channel_indices: Sequence[int] | None = None,
    band: str = "beta",
    window_sec: float = 10.0,
    step_sec: float = 2.0,
    latent_bands: Sequence[str] | None = None,
    jacobian_ridge: float = 1e-3,
    valve_quantile: float = 0.7,
    gate_q1: float = 0.33,
    gate_q2: float = 0.66,
) -> Dict[str, Any]:
    """Run the full pipeline on a multi-channel EEG array.

    Parameters
    ----------
    eeg : ndarray, shape (n_channels, n_samples)
        Multi-channel time series.
    fs : float
        Sampling rate in Hz.
    channel_indices : sequence of int or None
        Which channels to average; if None, average all.
    band : str
        Canonical band name (delta, theta, alpha, beta, gamma) used for coherence.
    window_sec, step_sec : float
        Parameters for coherence / valve windows.
    latent_bands : sequence of str or None
        Bands to stack when building the latent trajectory. Defaults to (alpha, beta, gamma).
    jacobian_ridge : float
        Ridge regularization strength passed to `estimate_local_jacobian`.
    valve_quantile : float
        Upper quantile threshold on V(t) used for avalanche segmentation.
    gate_q1, gate_q2 : float
        Quantiles that define the soft gate thresholds m₁ and m₂.

    Returns
    -------
    results : dict
        Contains keys:
        - 'coherence'  : CoherenceAvalancheResult dataclass
        - 'latent'     : latent trajectory (T', D)
        - 'jacobian'   : Jacobian matrix
        - 'jacobian_metrics' : spectrum diagnostics
    """
    eeg = np.asarray(eeg, float)
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2D (n_channels, n_samples)")
    n_channels, n_samples = eeg.shape
    if channel_indices is None:
        channel_indices = list(range(n_channels))
    channel_indices = np.asarray(channel_indices, int)
    if channel_indices.size == 0:
        raise ValueError("channel_indices must be non-empty")

    # simple mean reference and ROI average
    eeg_ref = mean_reference(eeg, axis=0)
    roi = eeg_ref[channel_indices]
    if roi.ndim == 1:
        roi = roi[None, :]

    # single-band envelope for coherence diagnostics
    roi_mean = roi.mean(axis=0)
    env = band_envelope(roi_mean, fs=fs, band=band, order=4, zscore_output=True)

    coh = coherence_avalanches_from_signal(
        env,
        fs,
        window_sec=window_sec,
        step_sec=step_sec,
        valve_quantile=valve_quantile,
        gate_q1=gate_q1,
        gate_q2=gate_q2,
    )

    # construct a latent by stacking summary stats from multiple bands
    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    from .utils import sliding_window_view_1d

    if latent_bands is None:
        latent_bands = ("alpha", "beta", "gamma")
    if len(latent_bands) == 0:
        raise ValueError("latent_bands must contain at least one band name")

    latents = []
    for latent_band in latent_bands:
        env_band = band_envelope(roi, fs=fs, band=latent_band, order=4, zscore_output=False)
        env_band = detrend(env_band, axis=-1, type="linear")
        band_mean = env_band.mean(axis=-1, keepdims=True)
        band_std = env_band.std(axis=-1, keepdims=True) + 1e-9
        env_band = (env_band - band_mean) / band_std
        env_band_mean = env_band.mean(axis=0)
        windows_band, _ = sliding_window_view_1d(env_band_mean, win, step)
        means_b = windows_band.mean(axis=1)
        vars_b = windows_band.var(axis=1)
        skew_b = ((windows_band - means_b[:, None]) ** 3).mean(axis=1) / (vars_b + 1e-9) ** 1.5
        latents.append(np.stack([means_b, vars_b, skew_b], axis=1))

    band_latent = np.concatenate(latents, axis=1)

    X = build_bandpower_latent(band_latent, downsample=1, center=True)
    # meter is sampled once per window
    if coh.times.size != X.shape[0]:
        # align by truncation to the shorter length
        L = min(coh.times.size, X.shape[0])
        X = X[:L]
    dt = (coh.times[1] - coh.times[0]) if coh.times.size > 1 else 1.0
    J = estimate_local_jacobian(X, dt=dt, ridge=jacobian_ridge)
    J_metrics = jacobian_spectrum_metrics(J)

    return {
        "coherence": coh,
        "latent": X,
        "jacobian": J,
        "jacobian_metrics": J_metrics,
    }
