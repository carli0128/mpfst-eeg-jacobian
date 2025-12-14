"""Canonical meltdownFrac definition."""

from __future__ import annotations

import numpy as np

from .utils import sliding_window_view_1d


def estimate_Mth(S: np.ndarray, method: str = "quantile", q: float = 0.999, fixed: float | None = None) -> float:
    """Estimate meltdown threshold M_th."""
    S = np.asarray(S, float)
    if method == "fixed":
        if fixed is None:
            raise ValueError("fixed Mth requested but `fixed` is None")
        return float(fixed)
    if method == "quantile":
        return float(np.quantile(S, q))
    raise ValueError(f"Unknown method '{method}'")


def meltdown_indicator(S: np.ndarray, Mth: float, frac: float = 0.8) -> np.ndarray:
    """Binary indicator for S(t) > frac * M_th."""
    S = np.asarray(S, float)
    return (S > frac * Mth).astype(float)


def meltdownFrac_windowed(indicator: np.ndarray, window: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute windowed meltdown fraction."""
    indicator = np.asarray(indicator, float)
    windows, idx = sliding_window_view_1d(indicator, window=window, step=step)
    frac = windows.mean(axis=1)
    times = idx.astype(float)
    return frac, times
