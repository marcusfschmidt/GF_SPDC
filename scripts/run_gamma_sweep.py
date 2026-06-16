# %%
from __future__ import annotations


"""Gamma sweep runner for stitched Green's functions and 2PA metrics.

For each gamma value this script:
- builds solver parameters,
- extracts and stitches Green's functions,
- computes photon number and Schmidt number K,
- evaluates the 2PA contribution breakdown,
- saves a compact summary payload for later analysis.

Distinguishable 2PA is the supported path. If indistinguishable is requested,
the script warns and falls back to the distinguishable logic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import warnings

import numpy as np

from gf_spdc.extractor import GreenFunctionsExtractor
from gf_spdc.loader import fft2_shifted
from gf_spdc.mgo_lithium_niobate_type0_beta import MgOLithiumNiobateType0
from gf_spdc.mgo_lithium_niobate_type2_beta import MgOLithiumNiobateType2
from gf_spdc.stitcher import GreenFunctionStitcher, StitcherParameters
from gf_spdc.two_photon_absorption import (
    DistinguishableTPAInputs,
    TPAContributionBreakdown,
    TPAHFunction,
    calculate_distinguishable_tpa_overlap,
)


@dataclass(frozen=True)
class GammaSweepResult:
    gamma: float
    photon_number: float
    schmidt_number: float
    coherent: float
    incoherent_type1: float
    incoherent_type2: float
    incoherent_total: float
    rate: float
    h_function: TPAHFunction
    breakdown: TPAContributionBreakdown
    saved_filename: str


def build_default_params(
    type: str = "typeII",
    length: float = 4000e-6,
    gamma: float = 1e-5,
    n: int = 12,
    dt: float = 1e-2,
    dz: float = 0.2e-3,
) -> StitcherParameters:
    """Return a `StitcherParameters` dataclass with sensible defaults."""
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


def _extract_gamma(parameter_array: object) -> float:
    if hasattr(parameter_array, "gamma"):
        return float(getattr(parameter_array, "gamma"))
    if isinstance(parameter_array, np.ndarray):
        if parameter_array.ndim == 0:
            parameter_array = cast(Any, parameter_array).item()
        else:
            parameter_array = cast(Any, parameter_array).tolist()
    if isinstance(parameter_array, (list, tuple)) and len(parameter_array) > 5:
        return float(cast(Any, parameter_array[5]))
    raise ValueError("Could not extract gamma from parameter payload.")


def _extract_schmidt_number_from_stitched(
    green_functions: tuple[np.ndarray, ...],
) -> float:
    singular_values = np.linalg.svd(np.asarray(green_functions[0]), compute_uv=False)
    denom = float(np.sum(singular_values**4))
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError(
            "Schmidt number is undefined for the supplied Green's function."
        )
    return float(1.0 / denom)


def _build_tpa_inputs(
    green_functions: tuple[np.ndarray, ...],
    omega: np.ndarray,
    gamma: float,
    omega_fg: float,
    indistinguishable_bool: bool,
) -> DistinguishableTPAInputs:
    g_is = fft2_shifted(np.asarray(green_functions[0]))
    g_ii = fft2_shifted(np.asarray(green_functions[1])) if len(green_functions) > 1 else g_is
    g_si = fft2_shifted(np.asarray(green_functions[2])) if len(green_functions) > 2 else g_is
    g_ss = fft2_shifted(np.asarray(green_functions[3])) if len(green_functions) > 3 else g_ii
    omega = np.asarray(omega, dtype=float)
    domega = float(omega[1] - omega[0]) if omega.size > 1 else 1.0

    if indistinguishable_bool:
        warnings.warn(
            "Indistinguishable 2PA requested; falling back to distinguishable logic.",
            stacklevel=2,
        )

    return DistinguishableTPAInputs(
        g_is=g_is,
        g_ii=g_ii,
        g_si=g_si,
        g_ss=g_ss,
        omega=omega,
        domega=domega,
        gamma=gamma,
        omega_fg=omega_fg,
    )


def _calculate_2pa(
    tpa_inputs: DistinguishableTPAInputs,
    indistinguishable_bool: bool,
) -> tuple[TPAContributionBreakdown, TPAHFunction]:
    if indistinguishable_bool:
        warnings.warn(
            "Indistinguishable 2PA requested; using distinguishable fallback logic.",
            stacklevel=2,
        )
    return calculate_distinguishable_tpa_overlap(tpa_inputs)


def run_gamma_sweep(
    gammas: np.ndarray,
    *,
    type: str = "typeII",
    length: float = 4000e-6,
    n: int = 12,
    dt: float = 1e-2,
    dz: float = 0.2e-3,
    pump_width: float = 2.0,
    basis_width: float = 1.0,
    kmax: int = 20,
    validation_threshold: float = 0.49,
    step_fraction: float = 1.0,
    stitch_skip_threshold: float = 0.95,
    rate_factor: float = 1.0,
    omega_fg: float = 0.0,
    indistinguishable_bool: bool = False,
    output_dir: str | Path = "gamma_sweep_outputs",
) -> list[GammaSweepResult]:
    results: list[GammaSweepResult] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for gamma in np.asarray(gammas, dtype=float):
        params = build_default_params(
            type=type, length=length, gamma=float(gamma), n=n, dt=dt, dz=dz
        )
        stitcher = GreenFunctionStitcher(params, pump_width, kmax, debug_bool=False)
        green_functions, time_width_array, freq_width_array, stitch_times = (
            stitcher.run_full_stitch(
                basis_width,
                validation_threshold,
                step_fraction=step_fraction,
                stitch_skip_threshold=stitch_skip_threshold,
            )
        )

        saved_filename = stitcher.save_stitch_output(
            None,
            green_functions,
            time_width_array,
            freq_width_array,
            stitch_times,
            params,
        )

        photon_number = GreenFunctionsExtractor.photon_number_from_green_function(
            np.asarray(green_functions[0]),
            stitcher.dt,
        )
        schmidt_number = _extract_schmidt_number_from_stitched(green_functions)
        tpa_inputs = _build_tpa_inputs(
            green_functions,
            stitcher.omega,
            float(gamma),
            omega_fg,
            indistinguishable_bool,
        )

        breakdown, h_function = _calculate_2pa(tpa_inputs, indistinguishable_bool)

        rate = float(rate_factor * (breakdown.coherent + breakdown.incoherent_total))

        result = GammaSweepResult(
            gamma=float(gamma),
            photon_number=photon_number,
            schmidt_number=schmidt_number,
            coherent=breakdown.coherent,
            incoherent_type1=breakdown.incoherent_type1,
            incoherent_type2=breakdown.incoherent_type2,
            incoherent_total=breakdown.incoherent_total,
            rate=rate,
            h_function=h_function,
            breakdown=breakdown,
            saved_filename=saved_filename,
        )
        results.append(result)

    summary = np.array(
        [
            (
                item.gamma,
                item.photon_number,
                item.schmidt_number,
                item.coherent,
                item.incoherent_type1,
                item.incoherent_type2,
                item.incoherent_total,
                item.rate,
                item.h_function.coherent,
                item.h_function.incoherent_type1,
                item.h_function.incoherent_type2,
                item.h_function.incoherent_total,
                item.saved_filename,
            )
            for item in results
        ],
        dtype=object,
    )
    np.save(output_dir / "gamma_sweep_summary.npy", summary)
    return results


# %%
if __name__ == "__main__":
    print(run_gamma_sweep(np.array([1e-5], dtype=float)))

# %%
