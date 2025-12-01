"""Template for downloading and converting monkey EEG datasets.

This script is deliberately non-executable out-of-the-box, because the exact
layout and naming of public datasets (e.g. OSF projects) can change.

To use it, fill in the TODO parts with paths/URLs for the dataset you plan to
use (for example, the OSF project `tuhsk` for Sandhaeger et al. 2019).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def convert_example_mat_to_npy(mat_path: str, out_path: str):
    """Example: convert a MATLAB file with EEG to a (n_channels, n_samples) .npy.

    This assumes the MATLAB file contains a variable like `eeg` with shape
    (n_samples, n_channels) or (n_channels, n_samples). Adjust as needed.

    Dependencies
    ------------
    - h5py or scipy.io.loadmat, depending on the MATLAB format.
    """
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Please `pip install h5py` to use this helper.") from exc

    mat_path = Path(mat_path)
    out_path = Path(out_path)
    with h5py.File(mat_path, "r") as f:
        # TODO: inspect keys and set `data_key` appropriately
        data_key = "eeg"
        data = np.array(f[data_key])
    # ensure (n_channels, n_samples)
    if data.shape[0] < data.shape[1]:
        eeg = data
    else:
        eeg = data.T
    np.save(out_path, eeg)
    print(f"Saved NumPy EEG array to {out_path}")


if __name__ == "__main__":
    print(
        "This is a template. Please edit `convert_example_mat_to_npy` with the "
        "appropriate keys for your dataset before running."
    )
