"""Dynamic coherence meter, valve, and avalanche statistics.

Implements a simplified form of the MPFST avalanche valve:

- A dynamic coherence meter m_ℓ(t) from windowed exponents (µ, γ, H);
 - A smoothed latent meter m̂_ℓ(t) via exponential smoothing;
- A soft two-tier gate function G(m_ℓ) with thresholds (m₁, m₂);
- A leaky valve V(t) driven by G(m_ℓ) and a simple coherence flux proxy;
- Avalanche segmentation and CSN tail fits for avalanche sizes.

The construction follows the MPFST Avalanche Addendum and the coarse-grained
derivation from the 11D action to (m_ℓ,V) dynamics.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np

from .coherence import windowed_exponents, csn_powerlaw_fit
from .utils import sliding_window_view_1d

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:  # pragma: no cover - executed only on older NumPy
    _trapz = np.trapz


@dataclass
class CoherenceAvalancheResult:
    times: np.ndarray
    mu: np.ndarray
    gamma: np.ndarray
    H: np.ndarray
    m_l: np.ndarray
    m_hat: np.ndarray
    valve: np.ndarray
    avalanches: Dict[str, np.ndarray]
    tail_fit: Dict[str, Any]


def _smooth_exponential(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x, float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def _gate_soft(m: np.ndarray, m1: float, m2: float, width: float = 0.05) -> np.ndarray:
    """Smooth two-tier gate in [0,1], approximate of the MPFST Ω(m_ℓ)."""
    m = np.asarray(m, float)
    def S(center):
        # symmetric soft step around `center` with width
        return 0.5 * (1.0 + np.tanh((m - center) / width))
    # lower tier starts near m1, upper tier near m2
    return 0.5 * S(m1) + 0.5 * S(m2)


def compute_dynamic_meter_and_valve(
    x: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    step_sec: float = 2.0,
    m1: float | None = None,
    m2: float | None = None,
    smooth_alpha: float = 0.2,
    valve_tau: float = 10.0,
    gate_q1: float = 0.33,
    gate_q2: float = 0.66,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute dynamic m_ℓ(t), m̂_ℓ(t) and valve V(t).

    Parameters
    ----------
    x : 1D array
        Band-limited amplitude or raw signal.
    fs : float
        Sampling rate in Hz (for consistency of window sizes).
    window_sec, step_sec : float
        Sliding window parameters passed to `windowed_exponents`.
    m1, m2 : float or None
        Gate thresholds. If None, they are set from the empirical distribution:
        m1 = 0.33 quantile, m2 = 0.66 quantile.
    smooth_alpha : float
    Exponential smoothing factor for the latent meter m̂_ℓ.
    valve_tau : float
        Valve time constant in **number of meter samples** (not seconds).

    Returns
    -------
    times, mu, gamma, H, m_l, m_hat, V
    """
    (
        times,
        mu,
        gamma,
        H,
        m_l,
    ) = windowed_exponents(x, fs, window_sec=window_sec, step_sec=step_sec)

    gate_q1 = float(np.clip(gate_q1, 0.0, 1.0))
    gate_q2 = float(np.clip(gate_q2, 0.0, 1.0))
    if m1 is None:
        m1 = float(np.quantile(m_l, gate_q1))
    if m2 is None:
        m2 = float(np.quantile(m_l, gate_q2))

    m_hat = _smooth_exponential(m_l, alpha=smooth_alpha)
    gate = _gate_soft(m_hat, m1=m1, m2=m2)

    # simple coherence flux proxy: finite differences of m_hat
    if m_hat.size < 2:
        dm = np.zeros_like(m_hat)
    else:
        dm = np.gradient(m_hat)
    flux = gate * dm

    # leaky valve integration
    dt = 1.0  # one step in meter-time
    lam = dt / max(valve_tau, 1e-6)
    V = np.zeros_like(m_hat)
    for i in range(1, len(V)):
        V[i] = V[i - 1] + lam * (-V[i - 1] + flux[i])

    return times, mu, gamma, H, m_l, m_hat, V


def extract_avalanches(
    times: np.ndarray,
    V: np.ndarray,
    m_hat: np.ndarray,
    m2: float,
    valve_quantile: float = 0.8,
):
    """Segment avalanches from valve trace.

    We require both:
    - m_hat >= m2 (high-gate regime),
    - V >= V_star (upper valve quantile),

    to define avalanche-supporting intervals.

    Returns
    -------
    avalanches : dict of arrays
        Keys: 'start', 'end', 'duration', 'size'.
    """
    times = np.asarray(times, float)
    V = np.asarray(V, float)
    m_hat = np.asarray(m_hat, float)
    if times.ndim != 1 or V.ndim != 1:
        raise ValueError("times and V must be 1D")
    if times.size != V.size or V.size != m_hat.size:
        raise ValueError("times, V, m_hat must have same length")

    V_star = float(np.quantile(V, valve_quantile))
    active = (m_hat >= m2) & (V >= V_star)
    aval_starts = []
    aval_ends = []
    sizes = []
    durations = []
    in_aval = False
    start_idx = 0
    for i, a in enumerate(active):
        if a and not in_aval:
            in_aval = True
            start_idx = i
        elif (not a) and in_aval:
            end_idx = i - 1
            in_aval = False
            aval_starts.append(times[start_idx])
            aval_ends.append(times[end_idx])
            dt = np.diff(times[start_idx : end_idx + 1]).mean() if end_idx > start_idx else 0.0
            durations.append((end_idx - start_idx + 1) * max(dt, 1.0))
            sizes.append(_trapz(V[start_idx : end_idx + 1] - V_star, times[start_idx : end_idx + 1]))
    if in_aval:
        end_idx = len(active) - 1
        aval_starts.append(times[start_idx])
        aval_ends.append(times[end_idx])
        dt = np.diff(times[start_idx : end_idx + 1]).mean() if end_idx > start_idx else 0.0
        durations.append((end_idx - start_idx + 1) * max(dt, 1.0))
        sizes.append(_trapz(V[start_idx : end_idx + 1] - V_star, times[start_idx : end_idx + 1]))

    return {
        "start": np.asarray(aval_starts, float),
        "end": np.asarray(aval_ends, float),
        "duration": np.asarray(durations, float),
        "size": np.asarray(sizes, float),
    }


def fit_avalanche_tail_exponent(
    sizes: np.ndarray, xmin: float | None = None, nboot: int = 200
) -> Dict[str, Any]:
    """Fit a power-law tail to avalanche sizes and return µ + diagnostics."""
    sizes = np.asarray(sizes, float)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if sizes.size < 10:
        return {"ok": False, "reason": "not_enough_avalanches"}
    if xmin is None:
        xmin, mu_hat, ks, p = csn_powerlaw_fit(sizes, nboot=nboot)
    else:
        xt = sizes[sizes >= xmin]
        if xt.size < 10:
            return {"ok": False, "reason": "not_enough_tail"}
        mu_hat, ks, p = np.nan, np.nan, np.nan  # simple placeholder
        xmin = float(xmin)
    return {
        "ok": True,
        "xmin": float(xmin),
        "mu_hat": float(mu_hat),
        "ks": float(ks),
        "p_boot": float(p),
        "n": int(sizes.size),
    }


def coherence_avalanches_from_signal(
    x: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    step_sec: float = 2.0,
    smooth_alpha: float = 0.2,
    valve_tau: float = 10.0,
    valve_quantile: float = 0.7,
    gate_q1: float = 0.33,
    gate_q2: float = 0.66,
) -> CoherenceAvalancheResult:
    """High-level convenience wrapper.

    Takes a 1D signal (typically a band-limited envelope) and returns all the
    ingredients needed to relate coherence gating, avalanches and Jacobian
    metrics.
    """
    (
        times,
        mu,
        gamma,
        H,
        m_l,
        m_hat,
        V,
    ) = compute_dynamic_meter_and_valve(
        x,
        fs,
        window_sec=window_sec,
        step_sec=step_sec,
        smooth_alpha=smooth_alpha,
        valve_tau=valve_tau,
        gate_q1=gate_q1,
        gate_q2=gate_q2,
    )
    # thresholds already from compute call, but recompute for clarity
    m1 = float(np.quantile(m_l, gate_q1))
    m2 = float(np.quantile(m_l, gate_q2))
    avalanches = extract_avalanches(times, V, m_hat, m2=m2, valve_quantile=valve_quantile)
    tail = fit_avalanche_tail_exponent(avalanches["size"])
    return CoherenceAvalancheResult(
        times=times,
        mu=mu,
        gamma=gamma,
        H=H,
        m_l=m_l,
        m_hat=m_hat,
        valve=V,
        avalanches=avalanches,
        tail_fit=tail,
    )
