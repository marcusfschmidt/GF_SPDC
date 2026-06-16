# %%
from __future__ import annotations


"""Notebook-friendly runner helpers for two-photon absorption overlaps.

This module mirrors `scripts/run_stitcher.py`: it exposes small helper
functions instead of an argparse-based CLI so notebooks can load a stitched
Green's function file and compute the indistinguishable-photon 2PA overlap
from `gf_spdc.two_photon_absorption`.
"""

from pathlib import Path
from typing import Any

import numpy as np

from gf_spdc.loader import fft2_shifted
from gf_spdc.two_photon_absorption import (
    IndistinguishableTPAInputs,
    TPAContributionBreakdown,
    calculate_indistinguishable_tpa_overlap,
)


DEFAULT_FILENAME = "stitchedGreens_type0_gamma 100_T0p 2.0_L 0.004.npy"


def load_stitched_green_functions(filename: str | Path | None = None) -> tuple:
    """Load stitched Green's functions saved by `GreenFunctionStitcher`.

    The saved file is the object array written by `save_stitch_output(...)`.
    """
    if filename is None:
        filename = DEFAULT_FILENAME
    return tuple(np.load(filename, allow_pickle=True))


def build_tpa_inputs(
    filename: str | Path | None = None,
    *,
    transition_linewidth: float,
    omega_fg: float = 0.0,
) -> IndistinguishableTPAInputs:
    """Build `IndistinguishableTPAInputs` from a stitched Green's file."""
    green_functions, _, _, _, _, omega, _, parameter_array = (
        load_stitched_green_functions(filename)
    )
    g = fft2_shifted(np.asarray(green_functions[0]))
    f = fft2_shifted(np.asarray(green_functions[1]))
    omega = np.asarray(omega, dtype=float)
    domega = float(omega[1] - omega[0]) if omega.size > 1 else 1.0
    return IndistinguishableTPAInputs(
        g=g,
        f=f,
        omega=omega,
        domega=domega,
        transition_linewidth=transition_linewidth,
        omega_fg=omega_fg,
    )


def run_tpa_from_file(
    filename: str | Path | None = None,
    *,
    transition_linewidth: float,
    omega_fg: float = 0.0,
) -> TPAContributionBreakdown:
    """Load stitched Green's functions and compute the 2PA overlap."""
    inputs = build_tpa_inputs(
        filename, transition_linewidth=transition_linewidth, omega_fg=omega_fg
    )
    result = calculate_indistinguishable_tpa_overlap(inputs)

    print("coherent:", result.coherent)
    print("incoherent_type1:", result.incoherent_type1)
    print("incoherent_type2:", result.incoherent_type2)
    print("incoherent_total:", result.incoherent_total)
    print("total:", result.total)
    print("g2:", result.g2)
    return result


# %%
if __name__ == "__main__":
    run_tpa_from_file()

# %%
