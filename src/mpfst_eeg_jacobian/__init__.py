"""Top-level package for the MPFST–EEG–Jacobian avalanche kit."""

from .coherence import (
    csn_powerlaw_fit,
    csn_bootstrap_ci,
    psd_slope_gamma,
    dfa_H,
    mel_from_exponents,
)
from .avalanches import (
    CoherenceAvalancheResult,
    compute_dynamic_meter_and_valve,
    extract_avalanches,
    fit_avalanche_tail_exponent,
)
from .jacobian import (
    build_bandpower_latent,
    estimate_local_jacobian,
    jacobian_spectrum_metrics,
)
from .pipeline import run_eeg_to_jacobian_avalanches

__all__ = [
    # coherence
    "csn_powerlaw_fit",
    "csn_bootstrap_ci",
    "psd_slope_gamma",
    "dfa_H",
    "mel_from_exponents",
    # avalanches
    "CoherenceAvalancheResult",
    "compute_dynamic_meter_and_valve",
    "extract_avalanches",
    "fit_avalanche_tail_exponent",
    # jacobian
    "build_bandpower_latent",
    "estimate_local_jacobian",
    "jacobian_spectrum_metrics",
    # high-level pipeline
    "run_eeg_to_jacobian_avalanches",
]
