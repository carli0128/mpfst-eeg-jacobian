"""Low-dimensional Jacobian / criticality analysis.

This module does not attempt to reimplement any specific paper verbatim.
Instead it provides a compact bridge between MPFST-style coherence/avalanche
metrics and low-dimensional ODE views of band-power dynamics, as advocated in
recent work on critical bifurcations and information capacity.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any


def build_bandpower_latent(
    band_envelopes: np.ndarray,
    downsample: int = 1,
    center: bool = True,
) -> np.ndarray:
    """Construct a low-dimensional latent trajectory from band envelopes.

    Parameters
    ----------
    band_envelopes : ndarray, shape (T, D)
        Columns correspond to different bands or regions (e.g. θ, α, β, γ).
    downsample : int
        Keep every `downsample`-th sample to reduce temporal resolution.
    center : bool
        If True, subtract the mean from each dimension.

    Returns
    -------
    X : ndarray, shape (T', D)
        Latent trajectory suitable for Jacobian estimation.
    """
    X = np.asarray(band_envelopes, float)
    if X.ndim != 2:
        raise ValueError("band_envelopes must be 2D (time x dims)")
    if downsample > 1:
        X = X[::downsample]
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    return X


def estimate_local_jacobian(
    X: np.ndarray,
    dt: float,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Estimate a single global linear Jacobian J from a trajectory.

    We estimate dX/dt ≈ J X in a least-squares sense, using central finite
    differences for dX/dt and a small ridge penalty for numerical stability.
    """
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (time x dims)")
    if X.shape[0] < 5:
        raise ValueError("need at least ~5 time points for Jacobian estimate")

    # central finite differences for interior, forward/backward at ends
    dX = np.empty_like(X)
    dX[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    dX[0] = (X[1] - X[0]) / dt
    dX[-1] = (X[-1] - X[-2]) / dt

    # solve dX ≈ X J^T  -> J^T = (X^T X + λI)^{-1} X^T dX
    XtX = X.T @ X
    dim = X.shape[1]
    XtX_reg = XtX + ridge * np.eye(dim)
    J_T = np.linalg.solve(XtX_reg, X.T @ dX)
    J = J_T.T
    return J


def jacobian_spectrum_metrics(J: np.ndarray) -> Dict[str, Any]:
    """Return eigenvalues and basic criticality diagnostics for J.

    We focus on:
    - max_real_eig: largest real part of eigenvalues (distance to instability);
    - spectral_radius: max |lambda|;
    - trace, determinant;
    - frob_norm: Frobenius norm ||J||_F;
    - is_structured: whether J has non-trivial magnitude;
    - is_near_critical: heuristic flag for a structured J near instability.
    """
    J = np.asarray(J, float)
    eigvals = np.linalg.eigvals(J)
    max_real = float(np.max(np.real(eigvals)))
    spectral_radius = float(np.max(np.abs(eigvals)))
    trace = float(np.trace(J))
    det = float(np.linalg.det(J))
    frob = float(np.linalg.norm(J, ord="fro"))

    structured = spectral_radius > 1e-3 and frob > 1e-3
    is_near_critical = bool(structured and abs(max_real) < 0.05)

    return {
        "eigvals": eigvals,
        "max_real_eig": max_real,
        "spectral_radius": spectral_radius,
        "trace": trace,
        "det": det,
        "frob_norm": frob,
        "is_structured": structured,
        "is_near_critical": is_near_critical,
    }
