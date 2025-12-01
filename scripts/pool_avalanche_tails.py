#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from mpfst_eeg_jacobian.avalanches import fit_avalanche_tail_exponent


def _discover_subjects(result_dir: Path) -> List[str]:
    json_paths = list(result_dir.glob("S???_R??_results.json"))
    subs = {path.name.split("_")[0] for path in json_paths}
    return sorted(subs)


def _safe_float(value):
    return None if value is None else float(value)


def _aggregate_subject(
    subj: str, result_dir: Path, xmin: float | None, nboot: int
) -> dict:
    sizes: List[float] = []
    durations: List[float] = []
    for json_path in sorted(result_dir.glob(f"{subj}_R??_results.json")):
        with json_path.open() as f:
            payload = json.load(f)
        aval = payload.get("avalanches", {})
        sizes.extend(aval.get("size", []))
        durations.extend(aval.get("duration", []))
    size_arr = np.asarray(sizes, float)
    size_arr = size_arr[np.isfinite(size_arr) & (size_arr > 0)]
    duration_arr = np.asarray(durations, float)
    duration_arr = duration_arr[np.isfinite(duration_arr) & (duration_arr > 0)]

    tail = fit_avalanche_tail_exponent(size_arr, xmin=xmin, nboot=nboot)
    ok = bool(tail.get("ok"))

    row = {
        "subject": subj,
        "n_sizes": int(size_arr.size),
        "size_mu_hat": tail.get("mu_hat"),
        "size_tail_ok": ok,
        "size_tail_reason": tail.get("reason", "ok") if not ok else "ok",
        "size_xmin": tail.get("xmin"),
        "size_ks": tail.get("ks"),
        "size_p_boot": tail.get("p_boot"),
        "n_durations": int(duration_arr.size),
        "duration_mean": _safe_float(duration_arr.mean()) if duration_arr.size else None,
        "duration_std": _safe_float(duration_arr.std(ddof=0)) if duration_arr.size else None,
        "duration_p99": _safe_float(np.percentile(duration_arr, 99)) if duration_arr.size else None,
    }
    return row


def process_directory(result_dir: Path, xmin: float | None, nboot: int) -> Path:
    if not result_dir.exists():
        raise FileNotFoundError(f"{result_dir} does not exist")
    subjects = _discover_subjects(result_dir)
    if not subjects:
        raise FileNotFoundError(
            f"No S???_R??_results.json files found under {result_dir}"
        )
    rows = [
        _aggregate_subject(subj, result_dir=result_dir, xmin=xmin, nboot=nboot)
        for subj in subjects
    ]
    out_path = result_dir / "subject_tail_metrics.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=
            [
                "subject",
                "n_sizes",
                "size_mu_hat",
                "size_tail_ok",
                "size_tail_reason",
                "size_xmin",
                "size_ks",
                "size_p_boot",
                "n_durations",
                "duration_mean",
                "duration_std",
                "duration_p99",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pool avalanche sizes/durations across runs per subject and fit the "
            "size tail exponent µ̂."
        )
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="One or more result directories produced by run_eegmmidb_batch.py",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=None,
        help="Optional fixed xmin to pass to fit_avalanche_tail_exponent",
    )
    parser.add_argument(
        "--nboot",
        type=int,
        default=200,
        help="Number of bootstrap samples for the CSN tail fitter",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    for dir_arg in args.dirs:
        result_dir = Path(dir_arg).expanduser()
        path = process_directory(result_dir, xmin=args.xmin, nboot=args.nboot)
        print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
