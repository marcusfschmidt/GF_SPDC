# GF_SPDC
Algorithm to extract all Green's functions for a second-order nonlinear crystal of arbitrary material, length and pump characteristics.
The approach assumes three-wave mixing, but is originally developed in the context of four-wave mixing in optical fibers. One can mathematically show that the same algorithm can be used both in the context of distinguishable and indistinguishable TWM.

Requires further functionality to effectively calculate the Green's functions for very long pulses (above hundreds of ns). For very short pulses (order of fs), utilizing large amounts of CUDA cores and memory becomes necessary to extract the functions in a reasonable amount of time.

## Setup

This repository now uses `uv` and a local `.venv`.

```bash
uv sync
```

That creates `.venv`, installs the package in editable mode, and installs the pinned runtime and development dependencies from `pyproject.toml` and `uv.lock`.

To resync exactly from the lockfile:

```bash
uv sync --frozen
```

## Layout

- `src/gf_spdc/`: typed library implementation
- `GreenFunctionStitcher.py`: script entry point for stitched Green's-function extraction
- `loadGreens.py`: script entry point for loading and plotting saved Green's functions
- `src/gf_spdc/two_photon_absorption.py`: two-photon absorption overlap logic for indistinguishable photons
 - `src/gf_spdc/`: typed library implementation
 - `scripts/run_stitcher.py`: notebook-friendly helpers to run extraction and stitching (preferred)
 - `scripts/plot_greens.py`: plot-only loader for saved stitched outputs
 - `src/gf_spdc/two_photon_absorption.py`: two-photon absorption overlap logic for indistinguishable photons

Notebook usage example
----------------------

Open a Jupyter notebook and run the following cells:

1) Import helpers and build default parameters

```python
from scripts.run_stitcher import build_default_params, run_stitcher_from_params

params = build_default_params(type="typeII")
# tweak parameters in the notebook as needed
params.length = 0.002
```

2) Run the stitcher and save results

```python
gf, time_w, freq_w, times, saved = run_stitcher_from_params(params, pump_width=2.0, basis_width=0.5, save=True)
print("Saved:", saved)
```

The scripts are intentionally minimal so the notebook remains the primary driver for experiments.
