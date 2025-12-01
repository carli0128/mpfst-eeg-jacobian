#!/usr/bin/env python
"""Utility to extract Dryad FLIP/vFLIP LFP arrays into MPFST-ready .npy files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _resolve_field(mat_dict: dict, path: str) -> np.ndarray:
    """Return ndarray referenced by dotted path (e.g. data.example1_vlPFC_lfp)."""
    parts = path.split(".")
    node: object = mat_dict
    for part in parts:
        if isinstance(node, dict):
            if part not in node:
                raise KeyError(
                    f"Field {path!r} not found. Missing segment {part!r}. Available keys: {sorted(node.keys())}"
                )
            node = node[part]
        else:
            raise KeyError(
                f"Cannot descend into {type(node)} at segment {part!r} while resolving {path!r}"
            )
    return np.asarray(node)


def load_lfp_field(mat_path: Path, field: str) -> np.ndarray:
    """Extract an LFP field as a (n_channels, n_samples, *extra) ndarray."""
    mat = loadmat(mat_path, simplify_cells=True)
    try:
        arr = _resolve_field(mat, field)
    except KeyError as exc:
        raise SystemExit(str(exc))
    return arr


def flatten_trials(arr: np.ndarray, mode: str) -> np.ndarray:
    """Convert 3D (channels, trials, time) to (channels, samples)."""
    if arr.ndim != 3:
        raise ValueError("flatten_trials expects a 3D array")
    if mode == "concat":
        # transpose to (channels, time, trials) then reshape
        n_ch, n_trials, n_samp = arr.shape
        return arr.transpose(0, 2, 1).reshape(n_ch, n_trials * n_samp)
    if mode == "mean_trials":
        return arr.mean(axis=1)
    raise ValueError(f"Unsupported mode {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Dryad (.mat) LFP fields (example1_vIPFC_lfp, raw, etc.) into "
            "NumPy arrays shaped (n_channels, n_samples) for the MPFST pipeline."
        )
    )
    parser.add_argument("--mat", required=True, help="Path to data.mat or vFLIP2_testdata.mat")
    parser.add_argument(
        "--field",
        required=True,
        help=(
            "MAT struct field to extract. Supports dotted paths like "
            "'data.example1_vlPFC_lfp' or 'raw'."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["concat", "mean_trials"],
        default="concat",
        help="How to handle 3D (channels, trials, time) arrays. Ignored for 2D arrays.",
    )
    parser.add_argument("--out-npy", required=True, help="Destination .npy file path")
    args = parser.parse_args()

    mat_path = Path(args.mat).expanduser()
    if not mat_path.exists():
        raise SystemExit(f"MAT file not found: {mat_path}")

    arr = load_lfp_field(mat_path, args.field)
    if arr.ndim == 2:
        eeg = arr
    elif arr.ndim == 3:
        eeg = flatten_trials(arr, mode=args.mode)
    else:
        raise SystemExit(
            f"Unsupported shape for field {args.field}: {arr.shape}. Expected 2D or 3D array."
        )

    out_path = Path(args.out_npy).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.asarray(eeg, dtype=np.float32))
    print(f"Saved {eeg.shape} array to {out_path}")


if __name__ == "__main__":
    main()
