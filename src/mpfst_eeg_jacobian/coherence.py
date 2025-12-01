"""MPFST-style coherence exponents and meter.

This module implements:

- CSN power-law tail fitting for dwell times (µ),
- low-frequency PSD slope (γ),
- DFA/Hurst exponent (H),
- a combined coherence meter m_ℓ = f(µ, γ, H).

The implementations are adapted from the MPFST v3 Addendum toolbox.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from scipy import stats, signal

from .utils import sliding_window_view_1d, safe_log10


# ---------------------------------------------------------------------
# CSN tail fitting (µ)
# ---------------------------------------------------------------------


def _fit_mu_mle(x_tail: np.ndarray) -> float:
    """Continuous power-law MLE for CSN tail.

    P(X >= x) ~ x^{-(µ-1)} for x >= xmin
    """
    x_tail = np.asarray(x_tail, float)
    if x_tail.size == 0:
        raise ValueError("empty tail")
    xmin = x_tail.min()
    return 1.0 + x_tail.size / np.sum(np.log(x_tail / xmin))


def _ks_stat(x_tail: np.ndarray, mu: float) -> float:
    x_tail = np.sort(np.asarray(x_tail, float))
    n = x_tail.size
    emp_ccdf = 1.0 - np.arange(1, n + 1) / n
    model_ccdf = (x_tail / x_tail.min()) ** (1.0 - mu)
    return np.max(np.abs(emp_ccdf - model_ccdf))


def csn_powerlaw_fit(
    x: np.ndarray,
    xmin_grid: np.ndarray | None = None,
    nboot: int = 200,
    rng: np.random.Generator | None = None,
):
    """CSN-style power-law tail fit for dwell times.

    Parameters
    ----------
    x : array_like
        Positive dwell times or avalanche sizes.
    xmin_grid : array_like or None
        Candidate xmin values. If None, use unique data above median.
    nboot : int
        Number of bootstrap samples for the KS p-value.
    rng : np.random.Generator or None

    Returns
    -------
    xmin : float
    mu_hat : float
    ks_min : float
    p_boot : float
        Bootstrap p-value for the KS statistic.
    """
    rng = default_rng() if rng is None else rng
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 10:
        raise ValueError("need at least ~10 positive samples for tail fit")
    xsort = np.sort(x)
    if xmin_grid is None:
        xmin_grid = np.unique(xsort[xsort >= np.percentile(xsort, 50.0)])

    best = None
    for xm in xmin_grid:
        xt = x[x >= xm]
        if xt.size < 50:
            continue
        mu = _fit_mu_mle(xt)
        ks = _ks_stat(xt, mu)
        if (best is None) or (ks < best[2]):
            best = (xm, mu, ks, xt.copy())

    if best is None:
        # fall back to using minimum as xmin
        xt = x
        mu = _fit_mu_mle(xt)
        ks = _ks_stat(xt, mu)
        xmin, mu_hat, ks_min = xt.min(), mu, ks
    else:
        xmin, mu_hat, ks_min, xt = best

    # bootstrap p-value under fitted power law
    pcount = 0
    n = xt.size
    for _ in range(nboot):
        u = rng.random(n)
        pl = xmin * (1.0 - u) ** (-1.0 / (mu_hat - 1.0))
        mu_b = _fit_mu_mle(pl)
        ks_b = _ks_stat(pl, mu_b)
        if ks_b >= ks_min:
            pcount += 1
    p_boot = pcount / float(nboot)
    return float(xmin), float(mu_hat), float(ks_min), float(p_boot)


def csn_bootstrap_ci(
    x: np.ndarray, xmin: float, nboot: int = 500, rng: np.random.Generator | None = None
):
    """Non-parametric bootstrap CI for µ.

    Returns (mu_mean, (lo, hi)).
    """
    rng = default_rng() if rng is None else rng
    x = np.asarray(x, float)
    xt = x[x >= xmin]
    if xt.size < 10:
        raise ValueError("not enough tail samples for bootstrap")
    mus = []
    for _ in range(nboot):
        xt_b = rng.choice(xt, size=xt.size, replace=True)
        mus.append(_fit_mu_mle(xt_b))
    mus = np.asarray(mus, float)
    lo, hi = np.percentile(mus, [16.0, 84.0])
    return float(mus.mean()), (float(lo), float(hi))


# ---------------------------------------------------------------------
# PSD slope γ and DFA H
# ---------------------------------------------------------------------


def psd_slope_gamma(
    x: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: int | None = None,
    fmin: float = 1e-2,
    fmax: float | None = None,
):
    """Estimate 1/f^γ slope from Welch PSD with robust regression.

    Returns (gamma, f, Pxx).
    """
    x = np.asarray(x, float)
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="density")
    mask = (f >= fmin) & ((fmax is None) or (f <= fmax))
    f, Pxx = f[mask], Pxx[mask]
    X = safe_log10(f)
    Y = safe_log10(Pxx)

    A = np.vstack([np.ones_like(X), X]).T
    w = np.ones_like(X)
    # Huber-style IRLS
    for _ in range(6):
        beta, *_ = np.linalg.lstsq(A * w[:, None], Y * w, rcond=None)
        resid = Y - (beta[0] + beta[1] * X)
        s = 1.4826 * np.median(np.abs(resid))
        w = 1.0 / np.maximum(1.0, np.abs(resid / (2.5 * s + 1e-12)))
    intercept, slope = beta
    gamma = -slope
    return float(gamma), f, Pxx


def dfa_H(x: np.ndarray, min_box: int = 16, max_box: int | None = None, nbox: int = 20):
    """Standard DFA-2 Hurst exponent estimate."""
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    N = len(y)
    if max_box is None:
        max_box = N // 4
    sizes = np.unique(
        np.logspace(np.log10(min_box), np.log10(max_box), nbox).astype(int)
    )
    F = []
    for s in sizes:
        nseg = N // s
        if nseg < 4:
            continue
        z = y[: nseg * s].reshape(nseg, s)
        t = np.arange(s, dtype=float)
        rms = []
        for seg in z:
            b = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - (b[0] * t + b[1])) ** 2)))
        F.append(np.sqrt(np.mean(np.square(rms))))
    sizes = np.array(sizes[: len(F)], float)
    F = np.array(F, float)
    X = safe_log10(sizes)
    Y = safe_log10(F)
    H = np.polyfit(X, Y, 1)[0]
    return float(H), sizes, F


# ---------------------------------------------------------------------
# Coherence meter m_ell
# ---------------------------------------------------------------------


def mel_from_exponents(
    mu: float,
    gamma: float,
    H: float,
    w_mu: float = 0.5,
    w_gamma: float = 0.35,
    w_H: float = 0.15,
    mu_center: float = 2.0,
    mu_scale: float = 0.5,
    gamma_center: float = 2.0,
    gamma_scale: float = 1.0,
    H_center: float = 0.7,
    H_scale: float = 0.1,
) -> float:
    """Combine (mu, gamma, H) into a dimensionless coherence score m_l in [0, 1].

    This version keeps dynamic range even when the exponents are large by
    (1) centering each exponent around a soft reference value and
    (2) passing them through a tanh nonlinearity before mixing.

    Each transformed term lies in [-1, 1]. The weighted sum is then
    squashed back into [0, 1] via m = 0.5 * (score + 1).

    Parameters
    ----------
    mu, gamma, H : float
        Heavy-tail index, PSD slope, and Hurst exponent.
    w_mu, w_gamma, w_H : float
        Weights that sum to 1.
    *_center, *_scale : float
        Soft centers and scales used to normalize each exponent.

    Returns
    -------
    m : float
        Coherence score in [0, 1]. NaN if any input exponent is not finite.
    """  # noqa: D401
    import numpy as _np

    if not (_np.isfinite(mu) and _np.isfinite(gamma) and _np.isfinite(H)):
        return float("nan")

    # heavier tails (mu < 2) -> positive contribution
    mu_term = _np.tanh((mu_center - mu) / mu_scale)
    # steeper PSD (gamma > gamma_center) -> positive
    gamma_term = _np.tanh((gamma - gamma_center) / gamma_scale)
    # stronger long-range correlations (H > H_center) -> positive
    H_term = _np.tanh((H - H_center) / H_scale)

    score = w_mu * mu_term + w_gamma * gamma_term + w_H * H_term
    m = 0.5 * (score + 1.0)
    return float(_np.clip(m, 0.0, 1.0))


def _extract_excursion_durations(
    x: np.ndarray, fs: float, amp_percentile: float = 75.0
) -> np.ndarray:
    """Simple burst-duration proxy for dwell times.

    We take the analytic envelope, threshold it at the given percentile, and
    compute contiguous above-threshold segment durations.

    This is a pragmatic approximation of the dwell-time statistics used in
    the MPFST exponents; for real analyses you may wish to customize this.
    """
    x = np.asarray(x, float)
    analytic = signal.hilbert(x)
    amp = np.abs(analytic)
    thr = np.percentile(amp, amp_percentile)
    above = amp >= thr
    if not np.any(above):
        return np.array([], float)
    # run-length encode
    durations = []
    in_run = False
    run_len = 0
    for a in above:
        if a:
            in_run = True
            run_len += 1
        elif in_run:
            durations.append(run_len / fs)
            in_run = False
            run_len = 0
    if in_run and run_len > 0:
        durations.append(run_len / fs)
    return np.asarray(durations, float)


def windowed_exponents(
    x: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    step_sec: float = 2.0,
    amp_percentile: float = 75.0,
    fmin: float = 1e-1,
    fmax: float = 50.0,
):
    """Compute µ, γ, H, m_ℓ in sliding windows over a 1D signal.

    Returns
    -------
    times : ndarray, shape (n_windows,)
        Center time of each window.
    mu, gamma, H, m : ndarrays
        Exponent sequences.
    """
    x = np.asarray(x, float)
    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    windows, starts = sliding_window_view_1d(x, win, step)
    times = (starts + win / 2) / fs
    mus = []
    gammas = []
    Hs = []
    ms = []
    for w in windows:
        durations = _extract_excursion_durations(w, fs, amp_percentile)
        if durations.size >= 30:
            # full CSN fit when we have a decent tail
            xmin, mu_hat, _, _ = csn_powerlaw_fit(durations)
        elif durations.size >= 10:
            # lighter-weight tail fit: use the median as xmin
            xmin = np.percentile(durations, 50.0)
            xt = durations[durations >= xmin]
            try:
                mu_hat = _fit_mu_mle(xt)
            except ZeroDivisionError:
                mu_hat = np.nan
        else:
            # too few excursions to say anything sensible
            mu_hat = np.nan
        gamma_hat, _, _ = psd_slope_gamma(w, fs, nperseg=min(1024, win // 2), fmin=fmin, fmax=fmax)
        H_hat, _, _ = dfa_H(w, min_box=16, max_box=min(win // 4, 1024))
        m_hat = mel_from_exponents(mu_hat, gamma_hat, H_hat)
        mus.append(mu_hat)
        gammas.append(gamma_hat)
        Hs.append(H_hat)
        ms.append(m_hat)
    return (
        times,
        np.asarray(mus, float),
        np.asarray(gammas, float),
        np.asarray(Hs, float),
        np.asarray(ms, float),
    )
