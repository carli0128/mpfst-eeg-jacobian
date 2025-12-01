from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import mne

from mpfst_eeg_jacobian.pipeline import run_eeg_to_jacobian_avalanches
from mpfst_eeg_jacobian.utils import print_header


def _parse_subjects(root: Path, subjects_arg: Sequence[str] | None) -> List[str]:
    if subjects_arg:
        subs = []
        for item in subjects_arg:
            if item.upper().startswith("S"):
                subs.append(item.upper())
            else:
                subs.append(f"S{int(item):03d}")
        return subs
    # auto-discover
    subs = [
        d.name
        for d in sorted(root.iterdir())
        if d.is_dir() and d.name.upper().startswith("S")
    ]
    return subs


def _parse_runs(runs_arg: Sequence[int] | None) -> List[int]:
    if runs_arg:
        return [int(r) for r in runs_arg]
    return list(range(1, 15))  # 1..14 inclusive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run MPFST EEG→Jacobian pipeline on the PhysioNet eegmmidb dataset."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="physionet.org/files/eegmmidb/1.0.0",
        help="Root directory that contains S001/, S002/, ...",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/eegmmidb_results",
        help="Where to write per-run JSON and summary CSV.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        help="Optional list of subjects (e.g. 1 2 3 or S001 S002). "
        "If omitted, all S??? directories under --root are used.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="*",
        help="Optional list of run numbers (1..14). If omitted, all are tried.",
    )
    parser.add_argument(
        "--band",
        type=str,
        default="beta",
        help="Oscillatory band to use for the envelope (delta/theta/alpha/beta/gamma).",
    )
    parser.add_argument(
        "--latent-bands",
        type=str,
        default="alpha,beta,gamma",
        help="Comma-separated list of bands to stack for the latent trajectory (default: alpha,beta,gamma).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=30.0,
        help="Window length in seconds for coherence / latent (recommended: 30).",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=15.0,
        help="Step in seconds between windows (recommended: 15).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of channel indices (0-based). If omitted, all EEG channels are used.",
    )
    parser.add_argument(
        "--jacobian-ridge",
        type=float,
        default=1e-3,
        help="Ridge regularization strength for the Jacobian regression (default: 1e-3).",
    )
    parser.add_argument(
        "--valve-quantile",
        type=float,
        default=0.7,
        help="Quantile threshold on valve V(t) for avalanche detection (0-1).",
    )
    parser.add_argument(
        "--gate-q1",
        type=float,
        default=0.33,
        help="Quantile for lower coherence gate m1 (default 0.33).",
    )
    parser.add_argument(
        "--gate-q2",
        type=float,
        default=0.66,
        help="Quantile for upper coherence gate m2 (default 0.66).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = _parse_subjects(root, args.subjects)
    runs = _parse_runs(args.runs)

    print_header("MPFST pipeline on eegmmidb")
    print(f"Root     : {root}")
    print(f"Subjects : {subjects}")
    print(f"Runs     : {runs}")
    print(f"Band     : {args.band}")
    print(f"Window   : {args.window_sec}s, step {args.step_sec}s")
    latent_band_list = [b.strip() for b in args.latent_bands.split(",") if b.strip()]
    if not latent_band_list:
        raise SystemExit("--latent-bands must specify at least one band")
    print(f"Latent bands : {latent_band_list}")
    print(f"Jacobian ridge: {args.jacobian_ridge}")
    print(f"Valve quantile: {args.valve_quantile}")
    print(f"Gate quantiles: ({args.gate_q1}, {args.gate_q2})")

    summary_rows = []
    for subj in subjects:
        subj_dir = root / subj
        if not subj_dir.exists():
            print(f"[WARN] subject dir {subj_dir} does not exist, skipping.")
            continue
        for run in runs:
            edf_path = subj_dir / f"{subj}R{run:02d}.edf"
            if not edf_path.exists():
                # some subjects lack certain runs; that's fine
                continue
            print_header(f"{subj} run {run:02d}")
            print(f"EDF: {edf_path}")

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
                raw.pick_types(eeg=True, meg=False, eog=False, stim=False)
                fs = float(raw.info["sfreq"])
                eeg = raw.get_data()

                res = run_eeg_to_jacobian_avalanches(
                    eeg=eeg,
                    fs=fs,
                    channel_indices=args.channels,
                    band=args.band,
                    window_sec=args.window_sec,
                    step_sec=args.step_sec,
                    latent_bands=latent_band_list,
                    jacobian_ridge=args.jacobian_ridge,
                    valve_quantile=args.valve_quantile,
                    gate_q1=args.gate_q1,
                    gate_q2=args.gate_q2,
                )
                coh = res["coherence"]
                Jm = res["jacobian_metrics"]
                tail = coh.tail_fit

                mu_med = float(np.nanmedian(coh.mu))
                gamma_med = float(np.nanmedian(coh.gamma))
                H_med = float(np.nanmedian(coh.H))
                m_med = float(np.nanmedian(coh.m_l))
                n_aval = int(len(coh.avalanches["size"]))
                mu_A = (
                    float(tail.get("mu_hat"))
                    if isinstance(tail, dict)
                    and "mu_hat" in tail
                    and np.isfinite(tail["mu_hat"])
                    else None
                )

                # store full per-run JSON
                json_safe_eigvals = [
                    [float(ev.real), float(ev.imag)]
                    for ev in Jm["eigvals"]
                ]
                run_payload = {
                    "subject": subj,
                    "run": run,
                    "fs": fs,
                    "n_channels": int(eeg.shape[0]),
                    "n_samples": int(eeg.shape[1]),
                    "mu": coh.mu.tolist(),
                    "gamma": coh.gamma.tolist(),
                    "H": coh.H.tolist(),
                    "m_l": coh.m_l.tolist(),
                    "m_hat": coh.m_hat.tolist(),
                    "valve": coh.valve.tolist(),
                    "avalanches": {
                        k: np.asarray(v, float).tolist()
                        for k, v in coh.avalanches.items()
                    },
                    "avalanche_tail_fit": tail,
                    "jacobian_metrics": {
                        **{k: float(v) for k, v in Jm.items()
                           if isinstance(v, (int, float, np.floating))},
                        "eigvals": json_safe_eigvals,
                        "is_structured": bool(Jm.get("is_structured", False)),
                        "is_near_critical": bool(Jm.get("is_near_critical", False)),
                        "frob_norm": float(Jm.get("frob_norm", 0.0)),
                    },
                }
                json_path = out_dir / f"{subj}_R{run:02d}_results.json"
                with json_path.open("w") as f:
                    json.dump(run_payload, f, indent=2)

                summary_rows.append(
                    {
                        "subject": subj,
                        "run": run,
                        "fs": fs,
                        "n_channels": eeg.shape[0],
                        "n_samples": eeg.shape[1],
                        "mu_median": mu_med,
                        "gamma_median": gamma_med,
                        "H_median": H_med,
                        "m_l_median": m_med,
                        "n_avalanches": n_aval,
                        "mu_A": mu_A,
                        "max_real_eig": Jm.get("max_real_eig"),
                        "spectral_radius": Jm.get("spectral_radius"),
                        "trace": Jm.get("trace"),
                        "det": Jm.get("det"),
                        "frob_norm": Jm.get("frob_norm"),
                        "is_structured": Jm.get("is_structured"),
                        "is_near_critical": Jm.get("is_near_critical"),
                    }
                )

            except Exception as exc:  # pragmatic guard
                print(f"[ERROR] Failed on {edf_path}: {exc}")
                continue

    # write manifest CSV
    if summary_rows:
        csv_path = out_dir / "summary_manifest.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print_header("Done")
        print(f"Processed {len(summary_rows)} runs.")
        print(f"Summary CSV: {csv_path}")
    else:
        print("[WARN] No runs were successfully processed.")


if __name__ == "__main__":
    main()
