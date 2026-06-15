# %%
from __future__ import annotations


"""Notebook-friendly runner helpers for Green function stitching.

This module intentionally avoids an argparse-based CLI. In our workflow we
invoke these functions from Jupyter notebooks, so the public API exposes two
helpers:

- `build_default_params(...)` — build a `StitcherParameters` instance with
  reasonable defaults suitable for quick experiments.
- `run_stitcher_from_params(...)` — run extraction + stitching using a
  `StitcherParameters` object; returns the stitch output and optionally saves
  it through the stitcher API.

The module keeps the run logic minimal so notebooks remain readable and
reproducible.
"""

from typing import Any

import numpy as np

from gf_spdc.mgo_lithium_niobate_type0_beta import MgOLithiumNiobateType0
from gf_spdc.mgo_lithium_niobate_type2_beta import MgOLithiumNiobateType2
from gf_spdc.stitcher import GreenFunctionStitcher, StitcherParameters


def build_default_params(
    type: str = "typeII",
    length: float = 4000e-6,
    gamma: float = 1e-5,
    n: int = 12,
    dt: float = 1e-2,
    dz: float = 0.2e-3,
) -> StitcherParameters:
    """Return a `StitcherParameters` dataclass with sensible defaults.

    Use this in notebooks and tweak fields explicitly when needed.
    """
    # pump and signal wavelengths
    lambda_p = 532e-9
    lambda_s = 1064e-9
    c = 299792458e-12
    om_p = 2 * np.pi * c / lambda_p
    om_s = 2 * np.pi * c / lambda_s
    om_i = om_p - om_s
    lambda_i = 2 * np.pi * c / om_i

    if type == "typeII":
        beta: Any = MgOLithiumNiobateType2(lambda_s, lambda_i, lambda_p)
    else:
        qpm_period = 5.916450343734758e-6
        beta = MgOLithiumNiobateType0(
            lambda_s,
            lambda_i,
            lambda_p,
            ordinary_axis_bool=True,
            temperature=36,
            qpm_period=qpm_period,
        )

    return StitcherParameters(
        n=n,
        dt=dt,
        dz=dz,
        length=length,
        beta=beta,
        gamma=gamma,
        lambda_p=lambda_p,
        omega_s=-(om_p - om_s),
        omega_i=-(om_p - om_i),
        alpha_s=0.0,
        alpha_i=0.0,
        print_bool=False,
        rtol=1e-4,
        nsteps=10000,
    )


def run_stitcher_from_params(
    params: StitcherParameters,
    pump_width: float = 2.0,
    basis_width: float = 1,
    kmax: int = 20,
    validation_threshold: float = 0.49,
    save: bool = True,
    filename: str | None = None,
):
    """Run extraction and stitching using `params`.

    Returns the tuple returned by `run_full_stitch` and the filename if saved.
    """
    stitcher = GreenFunctionStitcher(params, pump_width, kmax, debug_bool=False)

    green_functions, time_width_array, freq_width_array, stitch_times = (
        stitcher.run_full_stitch(basis_width, validation_threshold)
    )

    saved_name = None
    if save:
        saved_name = stitcher.save_stitch_output(
            filename,
            green_functions,
            time_width_array,
            freq_width_array,
            stitch_times,
            params,
        )
    return green_functions, time_width_array, freq_width_array, stitch_times, saved_name


# %%
if __name__ == "__main__":
    params = build_default_params(type="0", n=10, dt=0.6e-2)
    run_stitcher_from_params(params, basis_width=0.5, kmax=20)
# %%
