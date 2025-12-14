"""High-level orchestration pipeline.

The canonical Manuscript-10 pathway is :func:`run_eeg_to_meltdownfrac`, which
maps occupant fields + illusions proxy into meltdownFrac and Jacobian metrics.

The legacy v9-style coherence meter + valve pipeline remains available as
``run_eeg_to_jacobian_avalanches`` for diagnostics.
"""

from __future__ import annotations

from typing import Dict, Any, Sequence

import numpy as np
from scipy.signal import detrend

from .preprocessing import mean_reference, band_envelope
from .avalanches import coherence_avalanches_from_signal
from .jacobian import build_bandpower_latent, estimate_local_jacobian, jacobian_spectrum_metrics
from .occupant_fields import compute_occupant_fields
from .illusions_field import simulate_d
from .meltdownfrac import estimate_Mth, meltdown_indicator, meltdownFrac_windowed
from .meltdown_events import extract_meltdown_events, fit_event_size_tail_exponent


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
    """Legacy v9-style coherence meter + valve pipeline (diagnostic).

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


def run_eeg_to_meltdownfrac(
    eeg: np.ndarray,
    fs: float,
    channel_indices: Sequence[int] | None = None,
    band_mapping: Dict[str, str] | None = None,
    illusions_alpha: float = 0.05,
    illusions_lambda: float = 0.05,
    illusions_sigma: float = 0.1,
    threshold_frac: float = 0.8,
    mth_method: str = "quantile",
    mth_q: float = 0.999,
    mth_fixed: float | None = None,
    meltdown_window_sec: float = 5.0,
    meltdown_step_sec: float = 1.0,
    jacobian_ridge: float = 1e-3,
    diagnostics_band: str = "beta",
    min_event_duration: float = 0.0,
) -> Dict[str, Any]:
    """Canonical Manuscript-10 pipeline: occupant fields → illusions → meltdownFrac."""
    eeg = np.asarray(eeg, float)
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2D (n_channels, n_samples)")
    n_channels, n_samples = eeg.shape
    if channel_indices is None:
        channel_indices = list(range(n_channels))
    ch_idx = np.asarray(channel_indices, int)
    if ch_idx.size == 0:
        raise ValueError("channel_indices must be non-empty")

    eeg_ref = mean_reference(eeg, axis=0)
    roi = eeg_ref[ch_idx]
    if roi.ndim == 1:
        roi = roi[None, :]

    U, u_sum = compute_occupant_fields(roi, fs=fs, band_mapping=band_mapping, aggregate=True)
    d = simulate_d(u_sum, alpha=illusions_alpha, lam=illusions_lambda, sigma=illusions_sigma, fs=fs)
    S = u_sum + d

    Mth = estimate_Mth(S, method=mth_method, q=mth_q, fixed=mth_fixed)
    indicator = meltdown_indicator(S, Mth, frac=threshold_frac)
    win = int(round(meltdown_window_sec * fs))
    step = int(round(meltdown_step_sec * fs))
    if win <= 0 or step <= 0:
        raise ValueError("meltdown window/step must be > 0")
    mf, mf_times = meltdownFrac_windowed(indicator, window=win, step=step)
    events = extract_meltdown_events(S, fs=fs, Mth=Mth, frac=threshold_frac, min_dur=min_event_duration)
    tail = fit_event_size_tail_exponent(events.get("size", []))

    latent_series = np.vstack([U, d[None, :]]).T  # (n_samples, 6)
    downsample = max(1, int(round(fs * meltdown_step_sec)))
    X = build_bandpower_latent(latent_series, downsample=downsample, center=True)
    if X.shape[0] < 5:
        J = np.zeros((latent_series.shape[1], latent_series.shape[1]))
        J_metrics = {"is_structured": False, "is_near_critical": False}
    else:
        dt = downsample / fs
        J = estimate_local_jacobian(X, dt=dt, ridge=jacobian_ridge)
        J_metrics = jacobian_spectrum_metrics(J)

    roi_mean = roi.mean(axis=0)
    env_diag = band_envelope(roi_mean, fs=fs, band=diagnostics_band, order=4, zscore_output=True)
    diag_coh = coherence_avalanches_from_signal(env_diag, fs, window_sec=meltdown_window_sec, step_sec=meltdown_step_sec)

    return {
        "U": U,
        "d": d,
        "S": S,
        "Mth": Mth,
        "meltdownFrac": {"fraction": mf, "start_idx": mf_times},
        "events": {"table": events, "tail_fit": tail},
        "jacobian": J,
        "jacobian_metrics": J_metrics,
        "diagnostics": {"coherence": diag_coh},
    }
