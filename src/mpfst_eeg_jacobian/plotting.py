"""Matplotlib helpers used by the demo script.

No specific style is enforced so that labs can plug this into their own
figure pipelines.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_meter_and_valve(times, m_l, m_hat, V):
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(times, m_l)
    ax[0].set_ylabel("m_l")
    ax[1].plot(times, m_hat)
    ax[1].set_ylabel("m_hat")
    ax[2].plot(times, V)
    ax[2].set_ylabel("Valve V")
    ax[2].set_xlabel("Time (a.u.)")
    fig.tight_layout()
    return fig


def plot_avalanche_ccdf(sizes, xmin=None, ax=None):
    sizes = np.asarray(sizes, float)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure
    if sizes.size == 0:
        ax.text(0.5, 0.5, "No avalanches", ha="center", va="center")
        return fig
    s = np.sort(sizes)
    ccdf = 1.0 - np.arange(1, s.size + 1) / s.size
    ax.loglog(s, ccdf, ".", ms=3)
    if xmin is not None:
        ax.axvline(xmin, color="k", ls="--", lw=1)
    ax.set_xlabel("Avalanche size")
    ax.set_ylabel("P(S > s)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    return fig


def plot_synergy_with_events(times: np.ndarray, S: np.ndarray, threshold: float, events: dict):
    """Plot S(t) with meltdown threshold and shaded events."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, S, label="S(t)")
    ax.axhline(threshold, color="k", ls="--", lw=1, label="0.8 Mth")
    for s, e in zip(events.get("start", []), events.get("end", [])):
        ax.axvspan(s, e, color="red", alpha=0.15)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Synergy S")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_meltdown_fraction(times: np.ndarray, mf: np.ndarray):
    """Plot windowed meltdown fraction."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(times, mf, color="purple")
    ax.set_xlabel("Window start (sample idx)")
    ax.set_ylabel("meltdownFrac")
    fig.tight_layout()
    return fig
