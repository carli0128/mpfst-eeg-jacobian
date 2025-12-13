# MPFST–EEG–Jacobian Avalanche Kit

This repository implements the canonical Manuscript-10 pathway on EEG-like data:

- occupant doping fields u₄…u₈ from band-limited envelopes,
- a fractional-memory illusions proxy d(t),
- synergy S(t) = ∑u_p + d,
- meltdownFrac = Heaviside[S(t) > 0.8 M\_th] in sliding windows,
- low-dimensional Jacobian metrics on the latent [u₄…u₈, d].

The legacy v9 coherence meter + valve avalanches remain available as **diagnostic
outputs**; they are no longer treated as the MPFST order parameter here.

It is organized so that a lab can:
1. Plug in their own EEG/LFP or multi-unit data from PFC tasks.
2. Compute canonical meltdownFrac and meltdown events from S(t).
3. Construct a low-dimensional latent trajectory (u₄…u₈ + d) and estimate local
   Jacobians / criticality metrics.
4. Optionally recompute the legacy coherence meter, gate tiers and avalanche
   statistics as a sanity check.

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

# 4. Run the end-to-end demo (canonical meltdownFrac by default)
python scripts/run_full_pipeline.py --demo
# Legacy diagnostics:
# python scripts/run_full_pipeline.py --demo --mode legacy
```

Because the project uses a `src/` layout, `pip install -e .` (or an equivalent
editable install) is required so `mpfst_eeg_jacobian` can be imported from the
tests, scripts and notebooks. Skipping that step will reproduce the
`ModuleNotFoundError` mentioned in the issues list.

Canonical defaults (see `configs/canonical_meltdown_default.yaml`):

- `--mth-method=quantile`, `--mth-q=0.999`
- `--meltdown-frac=0.8`
- `--meltdown-window-sec=5.0`, `--meltdown-step-sec=1.0`
- `--illusions-alpha=0.05`, `--illusions-lam=0.05`, `--illusions-sigma=0.1`
- `--jacobian-ridge=1e-4`

Legacy diagnostics keep the original knobs:
- `--latent-bands=theta,alpha,beta,gamma`
- `--gate-q1=0.20`, `--gate-q2=0.60`
- `--valve-quantile=0.35`

**Reviewer note:** the Dryad excerpts are only a few minutes long. To obtain
≥10 avalanches for a stable tail fit you may need to lower `--valve-quantile`
from 0.35 to ~0.30 or concatenate multiple runs before re-running the pipeline.
The raw vFLIP montage also benefits from light preprocessing (remove obvious
bad channels or high-pass at ≥0.5 Hz) before feeding it into the MPFST pipeline
to avoid NaN coherence meters.

These match `configs/eegmmidb_default.yaml`, so you can go from the CLI to the
batch config without translating hyperparameters.

If everything is installed correctly you should see:

- A meltdown threshold estimate M\_th, windowed meltdownFrac, and a list of
  meltdown events (contiguous S(t) > 0.8 M\_th) with size/peak/duration.
- A Jacobian estimate for the latent [u₄…u₈, d] with a spectrum close to
  marginal stability (max Re(λ) ≈ 0).
- Optional legacy diagnostics: coherence exponents (µ, γ, H), coherence meter
  m_ℓ, valve avalanches and their tail fit.

Figures and intermediate results are written to `outputs/`.

## Repository layout

- `src/mpfst_eeg_jacobian/`
  - `occupant_fields.py` — occupant proxies u₄…u₈ from band envelopes.
  - `illusions_field.py` — fractional-memory surrogate for d(t).
  - `meltdownfrac.py` — meltdownFrac indicator + windowed fraction.
  - `meltdown_events.py` — canonical meltdown event segmentation.
  - `coherence.py` — legacy MPFST-style exponents (µ, γ, H) and coherence meter.
  - `avalanches.py` — legacy coherence gate + valve avalanches.
  - `preprocessing.py` — light EEG-style preprocessing: detrending,
    bandpass filters and analytic envelopes for canonical bands
    (δ, θ, α, β, γ, high-γ).
  - `jacobian.py` — low-dimensional latent construction and Jacobian metrics.
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
3. Call the canonical high-level function:

```python
from mpfst_eeg_jacobian.pipeline import run_eeg_to_meltdownfrac

results = run_eeg_to_meltdownfrac(
    eeg=eeg_array,
    fs=fs,
    channel_indices=[0, 1, 2],     # or any subset
    threshold_frac=0.8,
)
```

4. The `results` dict contains:
   - occupant fields `U`, illusions field `d`, synergy `S`,
   - meltdown threshold `Mth` and windowed meltdownFrac,
   - meltdown event table with size/peak/duration and an optional tail fit,
   - Jacobian estimate and spectrum for the latent [u₄…u₈, d],
   - `diagnostics` with the legacy coherence/avalanche bundle if needed.

Inspect `scripts/run_full_pipeline.py` for a complete example. Legacy v9-style
coherence valve diagnostics are still available via `run_eeg_to_jacobian_avalanches`
or `--mode legacy` on the CLI.

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

A canonical exploratory version of the pipeline lives in
`notebooks/01_canonical_meltdown_demo.ipynb`. It loads public excerpts (or the
synthetic demo), runs `run_eeg_to_meltdownfrac`, and renders S(t) with thresholds,
meltdownFrac, event times and Jacobian diagnostics. The legacy PhysioNet demo
notebook remains as a reference for the old valve pipeline.

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

## Public dataset notes 

As an example of public EEG, the kit is
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

- Canonical: occupant fields + fractional illusions (Plane-9 proxy) drive
  synergy S(t). MeltdownFrac thresholds S(t) at 0.8 M\_th and segments
  contiguous meltdown events; Jacobian metrics are computed on [u₄…u₈, d].
- Legacy diagnostics: the coherence gate + valve avalanche module is retained
  for continuity with prior MPFST v9/Addendum analyses, but its outputs are
  treated as supporting diagnostics rather than the primary order parameter.

The goal is not to replace any lab-specific analysis, but to provide a compact,
auditable bridge between canonical MPFST meltdownFrac and Jacobian / criticality
work that can be run on public data and easily adapted to new recordings.
