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
- `scripts/run_stitcher.py`: notebook-friendly helpers to run extraction and stitching (preferred)
- `scripts/plot_greens.py`: plot-only loader for saved stitched outputs
- `src/gf_spdc/two_photon_absorption.py`: two-photon absorption overlap logic for indistinguishable photons

## Notebook usage

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
gf, time_w, freq_w, times, saved = run_stitcher_from_params(
    params,
    pump_width=2.0,
    basis_width=0.5,
    kmax=20,
    step_fraction=1.0,       # increase for narrow basis_width (e.g. 3–10 for basis_width=0.1)
    stitch_skip_threshold=0.95,
    save=True,
)
print("Saved:", saved)
```

The scripts are intentionally minimal so the notebook remains the primary driver for experiments.

## Stitching algorithm

The Green's function (GF) is built up iteratively by extracting overlapping tiles in the Hermite-Gaussian basis and stitching them together.  A full run consists of two passes — one in each temporal direction from the initial extraction — each driven by `iterative_stitch`.

### Initial extraction

A single tile is extracted centered at the group-velocity walk-off midpoint (`init_offset`).  This tile is used as the seed for both stitching passes.

### Outer loop — one iteration per tile

Each iteration of the outer `while True` loop in `iterative_stitch` places the probe point at

```
test_offset = center_time + direction × width/2
```

i.e. at the far edge of the current GF window in the stitching direction.

**Termination branches (checked before extracting a new tile):**

| Condition | Meaning | Action |
|---|---|---|
| `test_offset` outside time axis | GF already fills the window; further tiling has no physical meaning | `break` — return current GF |
| `overlap > 0.999` at `test_offset` | Current GF is already perfect at the edge | `break` — return current GF |

### Inner loop — stitch point finder

If neither termination condition fires, a new tile is extracted and the inner loop finds where to join it to the existing GF.  The proposed stitch time is the midpoint between the old and new extraction centres:

```
stitch_time = test_offset − stitch_half_width
```

The inner loop has three acceptance branches and one retry branch:

| Branch | Condition | Action |
|---|---|---|
| **Both overlaps high** | `new_overlap > stitch_skip_threshold` AND `old_overlap > stitch_skip_threshold` | Accept proposed stitch point immediately — both GFs are accurate here, a second extraction would be wasted |
| **Agreement within tolerance** | `abs(new_overlap − old_overlap) < 0.03` | Accept proposed stitch point |
| **Stitch point moved** (`stitch_move_bool=True`) | Second pass of inner loop after a failed first pass | Accept with `a_test = a_stitch_test` (test function from the first pass's proposed point); the second extraction used a more conservative centre offset |
| **Disagreement too large** | None of the above | Pull the extraction centre back by `stitch_half_width`, set `stitch_move_bool=True`, re-extract at the conservative offset (next inner iteration hits the "stitch point moved" branch) |

### Post-stitch validation and termination

After the inner loop exits, the old and new tiles are merged and the stitched GF is validated at `a_test`:

| Condition | Meaning | Action |
|---|---|---|
| `overlap_new < 0.025` | The field has faded — the new tile added no useful information | Return the **pre-stitch** GF (discard the last tile) |
| Otherwise | Stitching succeeded | Advance `width_offset_from_global_center` by `width_offset_from_new_center × step_fraction` and continue outer loop |

### Key tuning parameters

| Parameter | Default | Effect |
|---|---|---|
| `basis_width` | — | Gaussian 1/e half-width of each HG basis function.  Smaller values resolve finer GF structure but produce narrower tiles and more iterations. |
| `kmax` | 20 | Number of Hermite-Gaussian basis orders per extraction. |
| `step_fraction` | 1.0 | Multiplier on the auto-detected per-tile advance (≈ `basis_width × √(2·kmax)`).  Increase to 3–10 when using a narrow `basis_width` to avoid spending many iterations on a long window. |
| `stitch_skip_threshold` | 0.95 | Skip the second (probe) extraction when both GF overlaps at the proposed stitch point already exceed this value.  Set to 1.0 to disable. |
| `validation_threshold` | 0.95 | Overlap required for the initial extraction to be considered valid. |

