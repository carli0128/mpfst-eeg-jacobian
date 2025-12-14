"""Meltdown event segmentation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .coherence import csn_powerlaw_fit

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:  # pragma: no cover
    _trapz = np.trapz


def extract_meltdown_events(
    S: np.ndarray,
    fs: float,
    Mth: float,
    frac: float = 0.8,
    min_dur: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Extract contiguous events where S(t) exceeds frac * M_th."""
    S = np.asarray(S, float)
    if S.ndim != 1:
        raise ValueError("S must be 1D")
    thr = frac * Mth
    above = S > thr
    starts = []
    ends = []
    peaks = []
    sizes = []
    durations = []
    n = S.size
    in_event = False
    start_idx = 0
    for i, flag in enumerate(above):
        if flag and not in_event:
            in_event = True
            start_idx = i
        elif (not flag or i == n - 1) and in_event:
            end_idx = i if flag else i - 1
            dur = (end_idx - start_idx + 1) / fs
            if dur >= min_dur:
                starts.append(start_idx / fs)
                ends.append(end_idx / fs)
                seg = S[start_idx : end_idx + 1]
                durations.append(dur)
                peaks.append(np.max(seg))
                sizes.append(_trapz(seg - thr, dx=1.0 / fs))
            in_event = False
    return {
        "start": np.asarray(starts, float),
        "end": np.asarray(ends, float),
        "duration": np.asarray(durations, float),
        "peak": np.asarray(peaks, float),
        "size": np.asarray(sizes, float),
    }


def fit_event_size_tail_exponent(
    sizes: np.ndarray, xmin: float | None = None, nboot: int = 200
) -> Dict[str, Any]:
    """Power-law tail fit for meltdown event sizes."""
    sizes = np.asarray(sizes, float)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if sizes.size < 10:
        return {"ok": False, "reason": "not_enough_events"}
    if xmin is None:
        xmin, mu_hat, ks, p = csn_powerlaw_fit(sizes, nboot=nboot)
    else:
        xt = sizes[sizes >= xmin]
        if xt.size < 10:
            return {"ok": False, "reason": "not_enough_tail"}
        mu_hat, ks, p = np.nan, np.nan, np.nan
        xmin = float(xmin)
    return {
        "ok": True,
        "xmin": float(xmin),
        "mu_hat": float(mu_hat),
        "ks": float(ks),
        "p_boot": float(p),
        "n": int(sizes.size),
    }
