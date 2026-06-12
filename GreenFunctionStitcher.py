from __future__ import annotations

import sys
import time
from typing import Sequence

import numpy as np

import type0_beta_5perMgO_LN as beta_function_type0
import typeII_beta_5perMgO_LN as beta_function_type_ii
from gf_spdc.stitcher import ExtractionResult, GreenFunctionStitcher as _GreenFunctionStitcher
from gf_spdc.solver import BetaModel, TaylorBeta


class GreenFunctionStitcher(_GreenFunctionStitcher):
    def __init__(self, parameterArray: Sequence[object], T0p: float, kmax: int, debugBool: bool) -> None:
        super().__init__(parameterArray, T0p, kmax, debugBool)


__all__ = ["ExtractionResult", "GreenFunctionStitcher"]


def main() -> None:
    ###################################################
    ###### CME parameters ######
    # Number of grid points.
    n = 13
    # Time step and spatial step. The spatial step will be adjusted slightly depending on crystal length.
    dt = 0.5e-2
    dz = 0.2e-3
    # Relative tolerance and number of steps for the adaptive step size.
    rtol = 1e-4
    nsteps = 10000
    # Print the progress of the solver.
    print_bool = False

    # Pump and signal wavelengths.
    lambda_p = 532e-9
    lambda_s = 1064e-9
    # Calculate the idler wavelength from energy conservation.
    c = 299792458e-12
    om_p = 2 * np.pi * c / lambda_p
    om_s = 2 * np.pi * c / lambda_s
    om_i = om_p - om_s
    lambda_i = 2 * np.pi * c / om_i

    # Attenuation coefficients (dA ~ -alpha*A). For QPM on, they must be identical.
    alpha_s = 0.0
    alpha_i = alpha_s

    # Crystal length.
    length = 4000e-6

    # Define the beta function for type II phase matching.
    beta_type_ii = beta_function_type_ii.typeII(lambda_s, lambda_i, lambda_p)

    # Define the beta function for type 0 phase matching.
    qpm_period = 5.916450343734758e-6
    beta = beta_function_type0.type0(
        lambda_s,
        lambda_i,
        lambda_p,
        ordinaryAxisBool=True,
        temperature=36,
        QPMPeriod=qpm_period,
    )

    # phys: The type-II beta object remains here as a scratch variable for manual experiments.
    _ = beta_type_ii

    # Nonlinear coefficient.
    gamma = 1e-5
    # Pump pulse duration in ps.
    t0_p = 5.4

    # Define numerical frequencies.
    omega_s = -(om_p - om_s)
    omega_i = -(om_p - om_i)
    parameter_array: list[object] = [
        n,
        dt,
        dz,
        length,
        beta,
        gamma,
        lambda_p,
        omega_s,
        omega_i,
        alpha_s,
        alpha_i,
        print_bool,
        rtol,
        nsteps,
    ]

    #### Green's function parameters ####
    # Define the number of basis functions to be used in the Green's function extraction.
    kmax = 50

    # phys: Basis-function width used for stitching experiments.
    t0 = 1 / 10

    stitcher = GreenFunctionStitcher(parameter_array, t0_p, kmax, debugBool=True)
    initial_center_time = stitcher.find_initial_center_time(t0) * 0
    stitcher.gf.make_pump(stitcher.gf.solver_object.make_gaussian_input(t0_p))

    print("Extracting initial Green's functions...")
    start_time = time.time()
    extraction_result = stitcher.extract_green_functions(t0, -initial_center_time, check_bool=False)
    print(f"Green's function extraction took {time.time() - start_time} seconds.")

    validation_threshold = 0.98 / 2
    a_test_init = stitcher.make_test_function(extraction_result.init_offset[0], t0)
    validation_bool, init_overlap = stitcher.validate_propagation(
        extraction_result.green_functions[0],
        a_test_init,
        validation_threshold,
        plot_bool=False,
    )
    print(f"Calculated initial validation overlap: {init_overlap}")
    if not validation_bool:
        print(
            f"Validation failed with threshold of {validation_threshold}. "
            "Consider increasing the number of basis functions or the time resolution."
        )
        sys.exit()

    green_functions, time_width_array, freq_width_array, stitch_times_1 = stitcher.iterative_stitch(
        extraction_result.green_functions,
        extraction_result.init_offset,
        extraction_result.width_offset_from_global_center,
        extraction_result.width,
        validation_threshold,
        extraction_result.time_indices,
        extraction_result.freq_indices,
        0,
    )
    print("######################################### Changing direction #########################################")
    green_functions, time_width_array, freq_width_array, stitch_times_2 = stitcher.iterative_stitch(
        green_functions,
        extraction_result.init_offset,
        extraction_result.width_offset_from_global_center,
        extraction_result.width,
        validation_threshold,
        time_width_array,
        freq_width_array,
        1,
    )
    stitch_times = stitch_times_1 + stitch_times_2
    time_width_array_output = stitcher.output_width_array(time_width_array)
    freq_width_array_output = stitcher.output_width_array(freq_width_array)

    save_array = np.array(
        [
            green_functions,
            time_width_array_output,
            freq_width_array_output,
            stitch_times,
            stitcher.t,
            stitcher.omega,
            stitcher.lambda_axis,
            parameter_array,
        ],
        dtype=object,
    )
    type_string = "type0" if beta.QPMbool else "typeII"
    save_string = f"stitchedGreens_{type_string}_gamma {gamma}_T0p {t0_p}_L {length}"
    np.save(save_string, save_array)


if __name__ == "__main__":
    main()
