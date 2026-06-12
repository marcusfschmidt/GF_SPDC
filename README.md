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
- `SPDCNumerical_CPU.py`, `GreenFunctionsExtractorParallelized.py`, `type0_beta_5perMgO_LN.py`, `typeII_beta_5perMgO_LN.py`: compatibility wrappers
