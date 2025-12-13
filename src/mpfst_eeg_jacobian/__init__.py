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
from .pipeline import run_eeg_to_jacobian_avalanches, run_eeg_to_meltdownfrac
from .occupant_fields import compute_occupant_fields, DEFAULT_OCCUPANT_MAPPING
from .illusions_field import fractional_operator_fft, simulate_d
from .meltdownfrac import estimate_Mth, meltdown_indicator, meltdownFrac_windowed
from .meltdown_events import extract_meltdown_events, fit_event_size_tail_exponent

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
    "run_eeg_to_meltdownfrac",
    # canonical modules
    "compute_occupant_fields",
    "DEFAULT_OCCUPANT_MAPPING",
    "fractional_operator_fft",
    "simulate_d",
    "estimate_Mth",
    "meltdown_indicator",
    "meltdownFrac_windowed",
    "extract_meltdown_events",
    "fit_event_size_tail_exponent",
]
