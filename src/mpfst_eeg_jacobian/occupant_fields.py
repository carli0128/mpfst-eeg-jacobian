"""Occupant doping field proxies u₄…u₈ from EEG band envelopes."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .preprocessing import band_envelope

# Default mapping from occupant plane to canonical EEG band
DEFAULT_OCCUPANT_MAPPING: Dict[str, str] = {
    "u4": "alpha",
    "u5": "theta",
    "u6": "beta",
    "u7": "gamma",
    "u8": "high_gamma",
}


def compute_occupant_fields(
    eeg: np.ndarray,
    fs: float,
    band_mapping: Dict[str, str] | None = None,
    aggregate: bool = True,
    channel_indices: Sequence[int] | None = None,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute occupant fields u₄…u₈ as band envelopes.

    Parameters
    ----------
    eeg : ndarray, shape (n_channels, n_samples)
        Multi-channel EEG/LFP array.
    fs : float
        Sampling rate in Hz.
    band_mapping : dict or None
        Map from 'u4'...'u8' → band name in preprocessing.CANONICAL_BANDS.
    aggregate : bool
        If True, average envelopes across selected channels (shape (5, n_samples)).
        If False, keep spatial dimension (shape (5, n_channels, n_samples)).
    channel_indices : sequence of int or None
        Optional subset of channels to use. Defaults to all.
    order : int
        Butterworth filter order passed to `band_envelope`.
    """
    eeg = np.asarray(eeg, float)
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2D (n_channels, n_samples)")
    n_channels, n_samples = eeg.shape
    if channel_indices is None:
        channel_indices = list(range(n_channels))
    ch_idx = np.asarray(channel_indices, int)
    if ch_idx.size == 0:
        raise ValueError("channel_indices must be non-empty")

    mapping = DEFAULT_OCCUPANT_MAPPING if band_mapping is None else dict(band_mapping)
    occupant_keys = ["u4", "u5", "u6", "u7", "u8"]
    envs = []
    for key in occupant_keys:
        band = mapping.get(key)
        if band is None:
            raise ValueError(f"Missing band mapping for {key}")
        per_ch = []
        for idx in ch_idx:
            env = band_envelope(eeg[idx], fs=fs, band=band, order=order, zscore_output=False)
            per_ch.append(env)
        arr = np.stack(per_ch, axis=0)  # (n_channels, n_samples)
        if aggregate:
            arr = arr.mean(axis=0)
        envs.append(arr)

    U = np.stack(envs, axis=0)
    u_sum = U.sum(axis=0)
    return U, u_sum
