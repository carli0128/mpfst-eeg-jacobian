# MPFST–EEG–Jacobian Avalanche Kit

This repository implements a reproducible analysis pipeline that links the MPFST
coherence–gating + avalanche framework to low–dimensional Jacobian / criticality
analyses on EEG-like data.

The design goal is **drop–in compatibility** with:
- The MPFST v9 / Addendum analysis primitives (coherence exponents, gate tiers,
  avalanche valve)
- Low–dimensional criticality / bifurcation analyses based on local Jacobians
  of population trajectories, as in recent work on temporal scale–invariant
   dynamics and information-maximizing bifurcations.

It is organized so that a lab like **MillerLab** can:
1. Plug in their own EEG/LFP or multi-unit data from PFC tasks.
2. Recompute MPFST-style coherence meters, gate tiers and avalanche statistics.
3. Construct a low-dimensional latent trajectory (e.g. band-power manifold).
4. Estimate local Jacobians and basic criticality metrics from that trajectory.
5. Relate avalanche exponents / gate occupancy to Jacobian spectra.

A small synthetic demo is provided so the full pipeline can be smoke-tested
without external data. For real data (e.g. monkey EEG from Sandhaeger et al.
2019 with Miller as co-author), see the **`data/`** instructions below.

## Quick start

```bash
# 1. Create a virtual environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies and the package (src/ layout requires this)
pip install -r requirements.txt
pip install -e .

# 3. Run the synthetic smoke test (no external data required)
pytest

# 4. Run the end-to-end demo
python scripts/run_full_pipeline.py --demo
```

Because the project uses a `src/` layout, `pip install -e .` (or an equivalent
editable install) is required so `mpfst_eeg_jacobian` can be imported from the
tests, scripts and notebooks. Skipping that step will reproduce the
`ModuleNotFoundError` mentioned in the issues list.

The demo uses the same defaults as the PhysioNet batch config. The most
important CLI knobs and their defaults are:

- `--latent-bands=theta,alpha,beta,gamma`
- `--gate-q1=0.20`, `--gate-q2=0.60`
- `--valve-quantile=0.35`
- `--jacobian-ridge=1e-4`

**Reviewer note:** the Dryad excerpts are only a few minutes long. To obtain
≥10 avalanches for a stable tail fit you may need to lower `--valve-quantile`
from 0.35 to ~0.30 or concatenate multiple runs before re-running the pipeline.
The raw vFLIP montage also benefits from light preprocessing (remove obvious
bad channels or high-pass at ≥0.5 Hz) before feeding it into the MPFST pipeline
to avoid NaN coherence meters.

These match `configs/eegmmidb_default.yaml`, so you can go from the CLI to the
batch config without translating hyperparameters.

If everything is installed correctly you should see:

- A few console summaries of exponents (µ, γ, H), coherence meter m_ℓ,
  and an avalanche tail exponent β\_aval.
- A Jacobian estimate for a low-dimensional latent trajectory (mean/var/skew
  features stacked across the requested bands) with a spectrum close to
  marginal stability (max Re(λ) ≈ 0).

Figures and intermediate results are written to `outputs/`.

## Repository layout

- `src/mpfst_eeg_jacobian/`
  - `coherence.py` — MPFST-style exponents (µ, γ, H) and coherence meter m_ℓ,
    including CSN tail fitting, PSD slopes and DFA, adapted from the MPFST
      v3/v9 analysis toolbox.
  - `avalanches.py` — dynamic coherence meter, latent meter \hat m_ℓ,
    soft two-tier valve V(t), avalanche segmentation and tail fits,
      following the MPFST Avalanche Addendum.
  - `preprocessing.py` — light EEG-style preprocessing: detrending,
    bandpass filters and analytic envelopes for canonical bands
    (δ, θ, α, β, γ).
  - `jacobian.py` — low-dimensional latent construction from band power,
    local Jacobian estimation dX/dt ≈ J X, and basic criticality metrics
    (spectrum, distance to the imaginary axis), in the spirit of recent
     work on critical bifurcations as maximal-information points.
  - `plotting.py` — helper plotting functions (matplotlib) used in the demo.
  - `utils.py` — small utilities (z-scoring, windowing, logging).

- `scripts/`
  - `run_full_pipeline.py` — CLI entry point for:
    - synthetic demo (`--demo`);
    - real EEG arrays provided as `.npy` (`--eeg-npy path.npy --fs 1000`).
  - `download_monkey_eeg_template.py` — commented template showing how to
    download / convert public EEG datasets (e.g. Sandhaeger et al. 2019
  “Monkey EEG links neuronal color and motion information across species
  and scales”).

- `tests/`
  - `test_all.py` — self-contained smoke tests on synthetic data; no internet
    or external datasets required.

- `data/`
  - `README.md` — notes on expected structure if you want to mirror a public
    dataset locally.

## Dependencies

The core pipeline depends only on commonly available scientific Python tools:

- `numpy`
- `scipy`
- `matplotlib`

Optional, but **recommended** for real EEG use:

- `mne` — for robust EEG/LFP I/O and preprocessing.
- `h5py`, `pymatreader` — for loading `.mat` files if your data are in MATLAB
  format.

The core package (`src/mpfst_eeg_jacobian`) and the synthetic tests import only
NumPy/SciPy/Matplotlib, so they run without the optional stack. CLI helpers
that touch real datasets (for example `scripts/run_eegmmidb_batch.py`) import
`mne` at module scope, and the Dryad conversion scripts call into `h5py` / `pymatreader`.
Install the extras before running those commands; skip them if you only need the
demo.

## Using your own data

The pipeline is intentionally conservative: instead of trying to guess the
structure of every public dataset, it standardizes on simple NumPy arrays.

1. Convert your data into an array of shape `(n_channels, n_samples)` and
   sampling rate `fs` in Hz.
2. Decide which subset of channels you treat as “PFC-like” (or use the
   average across a region of interest).
3. Call the high-level function:

```python
from mpfst_eeg_jacobian.pipeline import run_eeg_to_jacobian_avalanches

results = run_eeg_to_jacobian_avalanches(
    eeg=eeg_array,
    fs=fs,
    channel_indices=[0, 1, 2],     # or any subset
    band='beta',                   # or 'theta', 'alpha', 'gamma'
)
```

4. The `results` dict contains:
   - windowed exponents `{mu, gamma, H}`,
   - coherence meter time series `m_l` and `m_hat`,
   - valve trace `V`,
   - avalanche table with sizes/durations,
   - avalanche tail-fit summary,
   - Jacobian estimate and spectrum for the latent trajectory.

Inspect `scripts/run_full_pipeline.py` for a complete example.

### Reproducing the packaged run

The release artifacts are generated by two short commands:

```bash
# Synthetic sanity-check (fast)
python scripts/run_full_pipeline.py --demo --output-dir outputs/demo

# Example PhysioNet subject using the pre-converted arrays
python scripts/run_full_pipeline.py \
  --eeg-npy data/flip_example1_vlPFC_concat.npy \
  --fs 1000 \
  --channel 0 1 2 3 \
  --output-dir outputs/flip_example1
```

The NumPy arrays referenced above are small excerpts shipped in `data/` for
smoke testing. For the full ds006036 PhysioNet dataset, follow the download
instructions in `data/README_DATA.md` (or re-run `scripts/download_ds006036.sh`).

### Notebook walkthrough

A lightweight exploratory version of the pipeline lives in
`notebooks/01_demo_physionet.ipynb`. It loads `data/flip_example1_vlPFC_concat.npy`,
runs `run_eeg_to_jacobian_avalanches`, and renders the same plots bundled in the
release.

## Reproducing the shipped artifacts

Every derived folder inside `data/` can be regenerated from public sources by
running the documented scripts. The commands below are copy/paste equivalents of
what was used to create the committed CSV/JSON/PNG bundles.

### PhysioNet ds006036 (eegmmidb) sweep

1. Download the PhysioNet release (≈ 3 GB) into the expected mirrored path:

   ```bash
   bash scripts/download_ds006036.sh  # writes to physionet.org/files/eegmmidb/1.0.0
   ```

2. Run the MPFST pipeline on all subjects/runs with the publication defaults.

   ```bash
   python scripts/run_eegmmidb_batch.py \
     --root physionet.org/files/eegmmidb/1.0.0 \
     --out-dir data/eegmmidb_all_theta_abg_vq035 \
     --band beta \
     --latent-bands theta,alpha,beta,gamma \
     --window-sec 10 \
     --step-sec 1 \
     --gate-q1 0.20 \
     --gate-q2 0.60 \
     --valve-quantile 0.35
   ```

   This produces `S???_R??_results.json` files plus `summary_manifest.csv` inside
   `data/eegmmidb_all_theta_abg_vq035/`.

3. Pool avalanche tails per subject to recreate `subject_tail_metrics.csv`:

   ```bash
   python scripts/pool_avalanche_tails.py data/eegmmidb_all_theta_abg_vq035
   ```

### Dryad FLIP / vlPFC examples

1. Place `data.mat` and `vFLIP2_testdata.mat` from
   <https://doi.org/10.5061/dryad.9w0vt4bnp> under
   `data/doi_10_5061_dryad_9w0vt4bnp__v20251104/`.

2. Convert the MATLAB structs into NumPy arrays using the helper:

   ```bash
   python scripts/convert_dryad_lfp_to_npy.py \
     --mat data/doi_10_5061_dryad_9w0vt4bnp__v20251104/data.mat \
     --field data.example1_vlPFC_lfp \
     --mode concat \
     --out-npy data/flip_example1_vlPFC_concat.npy

   python scripts/convert_dryad_lfp_to_npy.py \
     --mat data/doi_10_5061_dryad_9w0vt4bnp__v20251104/data.mat \
     --field data.example2_7A_lfp \
     --mode concat \
     --out-npy data/flip_example2_7A_concat.npy

   python scripts/convert_dryad_lfp_to_npy.py \
     --mat data/doi_10_5061_dryad_9w0vt4bnp__v20251104/vFLIP2_testdata.mat \
     --field raw \
     --mode concat \
     --out-npy data/vflip2_raw.npy
   ```

3. Run the full pipeline on each converted excerpt to populate the folders that
   ship in `data/`:

   ```bash
   python scripts/run_full_pipeline.py \
     --eeg-npy data/flip_example1_vlPFC_concat.npy \
     --fs 1000 \
     --channel 0 1 2 3 \
     --output-dir data/flip_example1_vlPFC_mpfst

   python scripts/run_full_pipeline.py \
     --eeg-npy data/flip_example2_7A_concat.npy \
     --fs 1000 \
     --output-dir data/flip_example2_7A_mpfst

   python scripts/run_full_pipeline.py \
     --eeg-npy data/vflip2_raw.npy \
     --fs 1000 \
     --output-dir data/vflip2_raw_mpfst
   ```

The same instructions are mirrored (with a bit more narrative context) in
`data/README_DATA.md`.

## Public dataset notes (Miller lab–aligned)

As an example of public EEG closely aligned with Miller’s work, the kit is
designed to work cleanly with monkey EEG data from:

> Sandhaeger, F. et al. (2019). *Monkey EEG links neuronal color and motion
> information across species and scales.* eLife. Data + MATLAB code:
> OSF project `tuhsk`.

If you have access to the Dryad FLIP LFP release, use
`scripts/convert_dryad_lfp_to_npy.py` and the `configs/dryad_flip_default.yaml`
recipe. The data are not redistributed here, so you must accept Dryad's terms
before running that conversion script.

The exact file naming in that OSF project may change; therefore the repo
includes only a **template** script showing how to:

- download the OSF project (via `osfclient` or manual download),
- convert the MATLAB structures into `(n_channels, n_samples)` arrays,
- drop those into `data/monkey_eeg_color_motion/`,
- call the pipeline on those arrays.

This keeps the repo robust while still giving Miller or collaborators a clear,
minimal path to plug in their own recording formats.

## Alignment with MPFST and the critical Jacobian picture

- The coherence primitives (µ, γ, H → m_ℓ; gate thresholds m₁, m₂) follow the
  v3/v9 addenda and cross-domain empirical dossier.
- The avalanche valve and segmentation implement the two-tier gate as a
  dynamical object, turning coherence excursions into avalanches whose size
  statistics re-express the same fractional order inferred from µ, γ, H.
- The Jacobian module treats the band-power manifold as a low-dimensional
  dynamical system and estimates local linearizations. This matches the
  “unified criticality/bifurcation” framing where temporal scale-invariance and
  maximal information throughput are quantified via eigenmodes of those
  low-dimensional ODEs.

The goal is not to replace any lab-specific analysis, but to give a compact,
auditable bridge between MPFST’s coherence gating and contemporary Jacobian /
criticality work that can be run on public data and easily adapted to new
recordings.
