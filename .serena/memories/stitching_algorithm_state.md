# GF_SPDC Stitching Algorithm — Final State

## Status: WORKING — clean termination confirmed

## What was done
Restored `iterative_stitch` in `src/gf_spdc/stitcher.py` to match the **pre-wuu** commit (`363112e^`), i.e. the state of `GreenFunctionStitcher.py` at commit `f803bb2` (parent of wuu).

## Pre-wuu logic (now in effect)
- **Single offset**: `test_offset = center_time + stitch_time_helper * width/2` (no split into edge/inner)
- **Bounds check on `test_offset` directly**
- **Single `a_test`** created at `test_offset`, used for both initial overlap test and post-stitch validation
- **`stitch_move_bool` branch**: `a_test = a_stitch_test` (reuses already-computed test function; does NOT create a new shifted function)
- **Threshold**: `< 0.03`

## Confirmed behaviour (type=0, n=10, dt=0.6e-2, basis_width=0.5, kmax=20)
- Pass 1 (`low_high_index=0`): 1 iteration, exits "Offset exceeds time axis"
- Pass 2 (`low_high_index=1`): 4 iterations, exits "overlap approaching zero" (validate=0.0247 < 0.025)

## Key reference commits
- `363112e` — wuu commit (had infinite loop issue)
- `363112e^` = `f803bb2` — pre-wuu state (working, now restored)
- `c29c7c5` — "Rollback stitching logic" (stripped wuu changes)

## Entry point
`scripts/run_stitcher.py` → `build_default_params(type="0", n=10, dt=0.6e-2)` → `run_stitcher_from_params(..., basis_width=0.5, kmax=20)`
