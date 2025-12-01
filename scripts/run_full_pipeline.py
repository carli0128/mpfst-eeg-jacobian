from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mpfst_eeg_jacobian.pipeline import run_eeg_to_jacobian_avalanches
from mpfst_eeg_jacobian.utils import print_header
from mpfst_eeg_jacobian.plotting import plot_meter_and_valve, plot_avalanche_ccdf
from mpfst_eeg_jacobian.avalanches import CoherenceAvalancheResult


def _generate_synthetic_eeg(n_channels: int, n_samples: int, fs: float) -> np.ndarray:
    """Toy synthetic 1/f-like noise with occasional coherent bursts.

    This is only for smoke testing; it is *not* meant to mimic any specific
    dataset.
    """
    rng = np.random.default_rng(0)
    t = np.arange(n_samples) / fs
    eeg = []
    for ch in range(n_channels):
        white = rng.standard_normal(n_samples)
        # flicker component via filtering in frequency domain
        X = np.fft.rfft(rng.standard_normal(n_samples))
        f = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        f[0] = f[1]
        X /= f ** (0.5)
        flicker = np.fft.irfft(X, n=n_samples)
        # coherent burst
        burst = np.sin(2 * np.pi * 20.0 * t)
        window = 0.5 * (1.0 + np.tanh((t - 1.5) / 0.2)) * (1.0 - 0.5 * np.tanh((t - 3.5) / 0.2))
        eeg.append(0.5 * white + 0.5 * flicker + 0.8 * window * burst)
    return np.vstack(eeg)


def main():
    parser = argparse.ArgumentParser(description="Run MPFST–EEG–Jacobian pipeline.")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic data.")
    parser.add_argument("--eeg-npy", type=str, help="Path to .npy array (n_channels, n_samples).")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--channel", type=int, nargs="*", default=None, help="Channel indices to use.")
    parser.add_argument("--band", type=str, default="beta", help="Oscillatory band to use.")
    parser.add_argument(
        "--latent-bands",
        type=str,
        default="theta,alpha,beta,gamma",
        help="Comma-separated list of bands to stack for the latent trajectory (default: theta,alpha,beta,gamma).",
    )
    parser.add_argument(
        "--jacobian-ridge",
        type=float,
    default=1e-4,
    help="Ridge regularization for the Jacobian estimate (default: 1e-4).",
    )
    parser.add_argument(
        "--valve-quantile",
        type=float,
    default=0.35,
    help="Quantile on valve V(t) used to define avalanches (default: 0.35).",
    )
    parser.add_argument(
        "--gate-q1",
        type=float,
    default=0.20,
    help="Quantile for lower coherence gate m1 (default 0.20).",
    )
    parser.add_argument(
        "--gate-q2",
        type=float,
    default=0.60,
    help="Quantile for upper coherence gate m2 (default 0.60).",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to write figures/JSON.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print_header("Running synthetic demo")
        fs = args.fs
        n_channels = 8
        # The pipeline expects several sliding windows (10 s window, 2 s step by default)
        # so make the synthetic demo long enough to yield ≥5 windows for the Jacobian.
        n_samples = int(30 * fs)  # 30 seconds
        eeg = _generate_synthetic_eeg(n_channels, n_samples, fs)
    else:
        if args.eeg_npy is None:
            raise SystemExit("Either --demo or --eeg-npy must be provided.")
        eeg = np.load(args.eeg_npy)
        fs = args.fs

    channels = args.channel
    if channels is None:
        channels = list(range(eeg.shape[0]))

    latent_bands = [b.strip() for b in args.latent_bands.split(",") if b.strip()]
    if not latent_bands:
        raise SystemExit("--latent-bands must list at least one band")

    results = run_eeg_to_jacobian_avalanches(
        eeg=eeg,
        fs=fs,
        channel_indices=channels,
        band=args.band,
        latent_bands=latent_bands,
        jacobian_ridge=args.jacobian_ridge,
        valve_quantile=args.valve_quantile,
        gate_q1=args.gate_q1,
        gate_q2=args.gate_q2,
    )

    coh: CoherenceAvalancheResult = results["coherence"]

    print_header("Coherence exponents (median)")
    print(f"mu   median={np.median(coh.mu):.3f}")
    print(f"gamma median={np.median(coh.gamma):.3f}")
    print(f"H     median={np.median(coh.H):.3f}")
    print(f"m_l   median={np.median(coh.m_l):.3f}")

    print_header("Avalanche tail fit")
    print(results["coherence"].tail_fit)

    print_header("Jacobian metrics")
    jm = results["jacobian_metrics"]
    print(f"max_real_eig    = {jm['max_real_eig']:.4f}")
    print(f"spectral_radius = {jm['spectral_radius']:.4f}")
    print(f"is_near_critical = {jm['is_near_critical']}")

    # save metrics
    with (out_dir / "jacobian_metrics.json").open("w") as f:
        json.dump(
            {
                "jacobian_metrics": jm,
                "tail_fit": results["coherence"].tail_fit,
            },
            f,
            default=lambda o: o if isinstance(o, (int, float, str, bool)) else None,
            indent=2,
        )

    # figures
    import matplotlib.pyplot as plt

    fig1 = plot_meter_and_valve(coh.times, coh.m_l, coh.m_hat, coh.valve)
    fig1.savefig(out_dir / "meter_and_valve.png", dpi=150)
    plt.close(fig1)

    sizes = coh.avalanches["size"]
    fig2 = plot_avalanche_ccdf(sizes, xmin=results["coherence"].tail_fit.get("xmin"))
    fig2.savefig(out_dir / "avalanche_ccdf.png", dpi=150)
    plt.close(fig2)

    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
