"""Fractional-memory illusions field proxy for Plane-9."""

from __future__ import annotations

from typing import Literal

import numpy as np


def fractional_operator_fft(x: np.ndarray, alpha: float, fs: float) -> np.ndarray:
    """Apply a fractional operator in the frequency domain."""
    x = np.asarray(x, float)
    if x.ndim != 1:
        raise ValueError("fractional_operator_fft expects a 1D signal")
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    omega = 2 * np.pi * freqs
    multiplier = np.abs(omega) ** alpha
    if multiplier.size:
        multiplier[0] = 0.0
    X = np.fft.rfft(x)
    return np.fft.irfft(multiplier * X, n=n)


def _fractional_weights(alpha: float, n: int) -> np.ndarray:
    """Grünwald–Letnikov weights (-1)^k * binom(alpha, k)."""
    w = np.zeros(n, float)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (alpha - (k - 1)) / k * (-1.0)
    return w


def _nonlinearity(x: np.ndarray, kind: Literal["softplus", "relu"] = "softplus") -> np.ndarray:
    if kind == "softplus":
        return np.log1p(np.exp(x))
    if kind == "relu":
        return np.maximum(0.0, x)
    raise ValueError(f"Unknown nonlinearity '{kind}'")


def simulate_d(
    u_sum: np.ndarray,
    alpha: float = 0.1,
    lam: float = 0.05,
    sigma: float = 0.1,
    fs: float = 1000.0,
    nonlinearity: Literal["softplus", "relu"] = "softplus",
) -> np.ndarray:
    """Simulate fractional-memory illusions field d(t).

    Euler step:
    d_{t+1} = d_t + Δt (-λ d_t + 𝔇^α[d]_t + σ g(u_sum(t)))
    """
    u_sum = np.asarray(u_sum, float)
    if u_sum.ndim != 1:
        raise ValueError("u_sum must be 1D")
    n = u_sum.size
    d = np.zeros(n, float)
    dt = 1.0 / fs
    w = _fractional_weights(alpha, n)
    for t in range(1, n):
        idx = np.arange(t + 1)
        frac_term = float(np.sum(w[: t + 1] * d[t - idx]))
        drive = sigma * _nonlinearity(np.asarray([u_sum[t - 1]]), kind=nonlinearity)[0]
        d[t] = d[t - 1] + dt * (-lam * d[t - 1] + frac_term + drive)
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
