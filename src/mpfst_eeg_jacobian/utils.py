import numpy as np


def zscore(x, axis=None, eps=1e-8):
    """Z-score along a given axis.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or None
        Axis along which to compute mean/std. If None, use global.
    eps : float
        Small constant to avoid division by zero.
    """
    x = np.asarray(x, float)
    mu = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, ddof=1, keepdims=True)
    return (x - mu) / (std + eps)


def sliding_window_view_1d(x, window, step):
    """Cheap 1D sliding window view.

    Returns an array of shape (n_windows, window).
    """
    x = np.asarray(x, float)
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be > 0")
    n = x.size
    if n < window:
        raise ValueError("series too short for given window")
    idx_starts = np.arange(0, n - window + 1, step, dtype=int)
    windows = np.stack([x[s:s+window] for s in idx_starts], axis=0)
    return windows, idx_starts


def safe_log10(x, eps=1e-30):
    x = np.asarray(x, float)
    return np.log10(np.abs(x) + eps)


def print_header(title):
    line = "=" * len(title)
    print(f"\n{title}\n{line}")
