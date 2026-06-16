from __future__ import annotations

import gc
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence, cast, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore[import-untyped]

from .extractor import GreenFunctionsExtractor
from .mgo_lithium_niobate_type0_beta import MgOLithiumNiobateType0
from .mgo_lithium_niobate_type2_beta import MgOLithiumNiobateType2
from .solver import CoupledModes, SolverParameterTuple


ComplexArray = NDArray[Any]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
WidthTuple = tuple[int, int, int, int]
BETA_MODEL_TYPES = (MgOLithiumNiobateType2, MgOLithiumNiobateType0)


# Strongly-typed holder for parameters passed to the stitcher and solver.
@dataclass(slots=True)
class StitcherParameters:
    n: int
    dt: float
    dz: float
    length: float
    beta: Any
    gamma: float
    lambda_p: float
    omega_s: float
    omega_i: float
    alpha_s: float
    alpha_i: float
    print_bool: bool
    rtol: float
    nsteps: int


WidthCollection = WidthTuple | tuple[WidthTuple, WidthTuple]
BETA_MODEL_TYPES = (MgOLithiumNiobateType2, MgOLithiumNiobateType0)


@dataclass(slots=True)
class ExtractionResult:
    green_functions: tuple[ComplexArray, ...]
    overlaps: NDArray[np.float64] | None
    schmidt_numbers: NDArray[np.float64] | None
    init_offset: FloatArray
    width_offset_from_global_center: FloatArray
    width: float
    time_indices: WidthCollection
    freq_indices: WidthCollection


class GreenFunctionStitcher:
    def __init__(
        self,
        parameter_array: Union[Sequence[Any], StitcherParameters],
        pump_width: float,
        kmax: int,
        debug_bool: bool,
    ) -> None:
        # Accept either the legacy sequence form or the new StitcherParameters dataclass.
        if isinstance(parameter_array, StitcherParameters):
            params_list = [
                parameter_array.n,
                parameter_array.dt,
                parameter_array.dz,
                parameter_array.length,
                parameter_array.beta,
                parameter_array.gamma,
                parameter_array.lambda_p,
                parameter_array.omega_s,
                parameter_array.omega_i,
                parameter_array.alpha_s,
                parameter_array.alpha_i,
                parameter_array.print_bool,
                parameter_array.rtol,
                parameter_array.nsteps,
            ]
        else:
            params_list = list(parameter_array)

        self.parameter_array = cast(SolverParameterTuple, tuple(params_list))
        self.T0p = pump_width
        self.kmax = kmax
        self.debug_bool = debug_bool
        self.gf = GreenFunctionsExtractor(kmax, debug_bool)
        self.gf.make_solver_parameters(
            self.parameter_array, CoupledModes(*self.parameter_array)
        )
        self.t = self.gf.t
        self.omega = self.gf.omega
        self.dt = self.gf.dt
        self.domega = self.gf.domega
        self.lambda_axis = self.gf.lambda_axis
        beta = self.gf.solver_object.beta
        if not isinstance(beta, BETA_MODEL_TYPES):
            raise TypeError(
                "GreenFunctionStitcher requires a beta model object with phase-matching metadata."
            )
        beta_model = cast(Any, beta)
        self.indistinguishable_bool = bool(beta_model.indistinguishableBool)
        self.current_basis_width: float | None = None

    def extract_green_functions(
        self,
        t0: float,
        basis_offset: float,
        check_bool: bool = False,
    ) -> ExtractionResult:
        self.current_basis_width = t0
        self.gf.make_pump(self.gf.solver_object.make_gaussian_input(self.T0p))
        self.gf.make_basis_functions(t0, basis_offset)
        green_tuple, overlaps, schmidt_numbers = self.gf.run_extractor(
            self.indistinguishable_bool, check_bool
        )

        time_indices_idler, freq_indices_idler = self.find_width(green_tuple[0])
        t1x_idler, t2x_idler, _, _ = time_indices_idler
        time_indices: WidthCollection = time_indices_idler
        freq_indices: WidthCollection = freq_indices_idler

        offset_x1_idler = -(self.t[t1x_idler] - (-self.gf.init_offset[0]))
        offset_x2_idler = -(self.t[t2x_idler] - (-self.gf.init_offset[0]))
        width = float(np.abs(offset_x1_idler - offset_x2_idler))
        width_offset_from_global_center = (
            np.array([offset_x1_idler, offset_x2_idler], dtype=float) / 2
        )

        if not self.indistinguishable_bool:
            time_indices_signal, freq_indices_signal = self.find_width(green_tuple[2])
            time_indices = (time_indices_idler, time_indices_signal)
            freq_indices = (freq_indices_idler, freq_indices_signal)

        return ExtractionResult(
            green_functions=green_tuple,
            overlaps=overlaps,
            schmidt_numbers=schmidt_numbers,
            init_offset=self.gf.init_offset,
            width_offset_from_global_center=width_offset_from_global_center,
            width=width,
            time_indices=time_indices,
            freq_indices=freq_indices,
        )

    def find_width(self, green_function: ComplexArray) -> tuple[WidthTuple, WidthTuple]:
        green_time = np.abs(green_function) ** 2
        green_frequency = np.abs(self.gf.fft(green_function)) ** 2
        return self.width_helper_function(green_time), self.width_helper_function(
            green_frequency
        )

    def width_helper_function(self, values: NDArray[np.float64]) -> WidthTuple:
        threshold = float(np.mean(values))
        non_zero_indices = np.where(values > threshold)
        x_coordinates = non_zero_indices[1]
        y_coordinates = non_zero_indices[0]
        if x_coordinates.size == 0 or y_coordinates.size == 0:
            return (0, values.shape[1] - 1, 0, values.shape[0] - 1)
        x1 = int(np.min(x_coordinates))
        x2 = int(np.max(x_coordinates))
        y1 = int(np.min(y_coordinates))
        y2 = int(np.max(y_coordinates))
        return (x1, x2, y1, y2)

    def test_overlap(
        self,
        green_function: ComplexArray,
        a_test: ComplexArray,
        basis_index: int,
        plot_bool: bool,
    ) -> float:
        pump = self.gf.Ap_0
        a_noise = np.zeros_like(a_test)
        if basis_index == 0:
            init_conditions = np.array([a_test, a_noise, pump])
        elif basis_index == 1:
            init_conditions = np.array([a_noise, a_test, pump])
        else:
            raise ValueError(f"Unsupported basis index {basis_index}")

        in_index = int(basis_index)
        out_index = int(not basis_index)

        cnlse = CoupledModes(*self.parameter_array)
        input_field, output_field = self.gf.run_cnlse(init_conditions, cnlse)

        input_component = input_field[:, in_index]
        output_component = output_field[:, out_index]
        propagated_input = output_field[:, in_index]
        green_output = np.sum(green_function * input_component, axis=1) * self.dt
        overlap = self.gf.calc_overlap(green_output, output_component)

        if plot_bool:
            plt.figure()
            plt.plot(self.t, np.abs(input_component) ** 2)
            plt.show(block=False)
            plt.pause(0.001)

            plt.figure()
            plt.plot(self.t, np.abs(green_output) ** 2)
            plt.plot(self.t, np.abs(output_component) ** 2)
            plt.show(block=False)
            plt.pause(0.001)

        del input_component, output_component, propagated_input, green_output, cnlse
        return overlap

    def validate_propagation(
        self,
        green_function: ComplexArray,
        a_test: ComplexArray,
        threshold: float,
        basis_index: int = 0,
        plot_bool: bool = False,
    ) -> tuple[bool, float]:
        overlap = self.test_overlap(green_function, a_test, basis_index, plot_bool)
        return overlap >= threshold, overlap

    def make_test_function(
        self, offset: float, t0: float | None = None
    ) -> ComplexArray:
        basis_width = self.current_basis_width if t0 is None else t0
        if basis_width is None:
            raise RuntimeError("Basis width has not been set yet.")
        return self.gf.solver_object.make_hermite_gaussian_basis_functions(
            offset, basis_width, 0
        )

    def remove_zero_values(
        self,
        g_field: ComplexArray,
        f_field: ComplexArray,
        input_axis: FloatArray,
        index_array: WidthTuple,
    ) -> tuple[ComplexArray, ComplexArray, FloatArray, FloatArray]:
        x1, x2, y1, y2 = index_array
        return (
            g_field[y1:y2, x1:x2],
            f_field[y1:y2, x1:x2],
            input_axis[x1:x2],
            input_axis[y1:y2],
        )

    def stitch_green_functions(
        self,
        init_offset_old: FloatArray,
        init_offset_new: FloatArray,
        old_green_functions: tuple[ComplexArray, ...],
        new_green_functions: tuple[ComplexArray, ...],
    ) -> tuple[ComplexArray, ...]:
        if not self.indistinguishable_bool:
            gis_old, g_ii_old, g_si_old, g_ss_old = old_green_functions
            gis_new, g_ii_new, g_si_new, g_ss_new = new_green_functions
            t1_idler, t1_signal = init_offset_old
            t2_idler, t2_signal = init_offset_new
            g_is, g_ii = self.stitch_helper_function(
                gis_old, g_ii_old, gis_new, g_ii_new, t1_idler, t2_idler
            )
            g_si, g_ss = self.stitch_helper_function(
                g_si_old, g_ss_old, g_si_new, g_ss_new, t1_signal, t2_signal
            )
            return (g_is, g_ii, g_si, g_ss)

        g_old, f_old = old_green_functions
        g_new, f_new = new_green_functions
        t1, _ = init_offset_old
        t2, _ = init_offset_new
        g_field, f_field = self.stitch_helper_function(
            g_old, f_old, g_new, f_new, t1, t2
        )
        return (g_field, f_field)

    def stitch_helper_function(
        self,
        g1: ComplexArray,
        f1: ComplexArray,
        g2: ComplexArray,
        f2: ComplexArray,
        t1: float,
        t2: float,
    ) -> tuple[ComplexArray, ComplexArray]:
        abs_diff1 = np.abs(self.t + t1)
        abs_diff2 = np.abs(self.t + t2)
        condition = (abs_diff1 < abs_diff2)[np.newaxis, :]
        return np.where(condition, g1, g2), np.where(condition, f1, f2)

    def compare_width_arrays_helper_function(
        self, array1: WidthTuple, array2: WidthTuple
    ) -> WidthTuple:
        return (
            min(array1[0], array2[0]),
            max(array1[1], array2[1]),
            min(array1[2], array2[2]),
            max(array1[3], array2[3]),
        )

    def compare_width_arrays(
        self,
        time_array_old: WidthCollection,
        time_array_new: WidthCollection,
        freq_array_old: WidthCollection,
        freq_array_new: WidthCollection,
    ) -> tuple[WidthCollection, WidthCollection]:
        if self.indistinguishable_bool:
            time_old = cast(WidthTuple, time_array_old)
            time_new = cast(WidthTuple, time_array_new)
            freq_old = cast(WidthTuple, freq_array_old)
            freq_new = cast(WidthTuple, freq_array_new)
            return (
                self.compare_width_arrays_helper_function(time_old, time_new),
                self.compare_width_arrays_helper_function(freq_old, freq_new),
            )

        time_old_pair = cast(tuple[WidthTuple, WidthTuple], time_array_old)
        time_new_pair = cast(tuple[WidthTuple, WidthTuple], time_array_new)
        freq_old_pair = cast(tuple[WidthTuple, WidthTuple], freq_array_old)
        freq_new_pair = cast(tuple[WidthTuple, WidthTuple], freq_array_new)
        time_idler = self.compare_width_arrays_helper_function(
            time_old_pair[0], time_new_pair[0]
        )
        time_signal = self.compare_width_arrays_helper_function(
            time_old_pair[1], time_new_pair[1]
        )
        freq_idler = self.compare_width_arrays_helper_function(
            freq_old_pair[0], freq_new_pair[0]
        )
        freq_signal = self.compare_width_arrays_helper_function(
            freq_old_pair[1], freq_new_pair[1]
        )
        return (time_idler, time_signal), (freq_idler, freq_signal)

    def add_padding_to_width(
        self, array: WidthTuple, padding_factor: float = 0.25
    ) -> NDArray[np.float64]:
        output = np.array(array, dtype=float)
        output[0] = array[0] - padding_factor * (array[1] - array[0])
        output[1] = array[1] + padding_factor * (array[1] - array[0])
        output[2] = array[2] - padding_factor * (array[3] - array[2])
        output[3] = array[3] + padding_factor * (array[3] - array[2])
        return output

    def output_width_array(self, widths: WidthCollection) -> WidthTuple:
        if self.indistinguishable_bool:
            return cast(WidthTuple, widths)
        width_pair = cast(tuple[WidthTuple, WidthTuple], widths)
        x1a, x2a, y1a, y2a = width_pair[0]
        x1b, x2b, y1b, y2b = width_pair[1]
        return (min(x1a, x1b), max(x2a, x2b), min(y1a, y1b), max(y2a, y2b))

    def stitch_time_helper(self, low_high_index: int) -> int:
        return 1 if low_high_index == 0 else -1

    def iterative_stitch(
        self,
        green_functions: tuple[ComplexArray, ...],
        center_time_initial: FloatArray,
        width_offset_initial: FloatArray,
        width: float,
        validation_threshold: float,
        time_width_array: WidthCollection,
        freq_width_array: WidthCollection,
        low_high_index: int = 0,
        signal_idler_test_index: int = 0,
        phase_label: str = "",
        step_fraction: float = 1.0,
        stitch_skip_threshold: float = 0.95,
        print_bool: bool = False,
    ) -> tuple[tuple[ComplexArray, ...], WidthCollection, WidthCollection, list[float]]:
        if self.current_basis_width is None:
            raise RuntimeError("Basis width has not been set yet.")

        test_index = (
            0
            if self.indistinguishable_bool
            else (2 if signal_idler_test_index == 1 else 0)
        )
        width_offset_from_global_center = np.copy(width_offset_initial)
        center_time = np.copy(center_time_initial)

        header = "Iteratively stitching Green's functions"
        if phase_label:
            header = f"{header} ({phase_label})"
        if print_bool:
            tqdm.write(f"{header}...")
        stitch_times: list[float] = []
        iteration = 0

        while True:
            iteration += 1
            gc.collect()
            loop_start = perf_counter()
            test_offset = (
                center_time[signal_idler_test_index]
                + self.stitch_time_helper(low_high_index) * width / 2
            )

            if test_offset < self.t[0] or test_offset > self.t[-1]:
                if print_bool:
                    tqdm.write(
                        "Offset exceeds time axis, stitching complete. "
                        "Inspect the output to determine if the time window should be extended."
                    )
                break

            a_test = self.make_test_function(test_offset, self.current_basis_width)
            _, overlap = self.validate_propagation(
                green_functions[test_index],
                a_test,
                validation_threshold,
                signal_idler_test_index,
                plot_bool=False,
            )
            iteration_header = f"iter {iteration}: edge={overlap:.6f}"
            self.gf.debug_print(f"Test overlap at edge of region: {overlap}")
            if overlap > 0.999:
                if print_bool:
                    tqdm.write(f"{iteration_header} | done (edge overlap > 99.9%)")
                break

            stitch_move_bool = False
            while True:
                if print_bool:
                    tqdm.write(f"{iteration_header} | extracting new Green's functions")
                new_result = self.extract_green_functions(
                    self.current_basis_width,
                    float(width_offset_from_global_center[low_high_index]),
                    check_bool=False,
                )
                center_time_new = new_result.init_offset
                width_offset_from_new_center = (
                    new_result.width_offset_from_global_center
                )
                width_new = new_result.width
                stitch_half_width = (
                    center_time_new[signal_idler_test_index]
                    - center_time[signal_idler_test_index]
                ) / 2
                stitch_time = test_offset - stitch_half_width

                if stitch_move_bool:
                    a_test = a_stitch_test
                    if print_bool:
                        tqdm.write(f"{iteration_header} | stitch point moved, retrying")
                    break

                a_stitch_test: ComplexArray = self.make_test_function(
                    stitch_time, self.current_basis_width
                )
                _, stitch_overlap_new = self.validate_propagation(
                    new_result.green_functions[test_index],
                    a_stitch_test,
                    validation_threshold,
                    signal_idler_test_index,
                    plot_bool=False,
                )
                _, stitch_overlap_old = self.validate_propagation(
                    green_functions[test_index],
                    a_stitch_test,
                    validation_threshold,
                    signal_idler_test_index,
                    plot_bool=False,
                )

                if print_bool:
                    tqdm.write(
                        f"{iteration_header} | stitch new={stitch_overlap_new:.6f} old={stitch_overlap_old:.6f} diff={abs(stitch_overlap_new - stitch_overlap_old):.6f}"
                    )

                if (
                    stitch_overlap_new > stitch_skip_threshold
                    and stitch_overlap_old > stitch_skip_threshold
                ):
                    if print_bool:
                        tqdm.write(
                            f"{iteration_header} | stitch point accepted (both overlaps high)"
                        )
                    break

                if abs(stitch_overlap_new - stitch_overlap_old) < 0.03:
                    if print_bool:
                        tqdm.write(f"{iteration_header} | stitch point accepted")
                    break

                width_offset_from_global_center -= stitch_half_width
                stitch_move_bool = True

            stitch_times.append(-stitch_time)
            time_width_out, freq_width_out = self.compare_width_arrays(
                time_width_array,
                new_result.time_indices,
                freq_width_array,
                new_result.freq_indices,
            )

            updated_green_functions = self.stitch_green_functions(
                center_time,
                center_time_new,
                green_functions,
                new_result.green_functions,
            )

            _, overlap_new = self.validate_propagation(
                updated_green_functions[test_index],
                a_test,
                validation_threshold,
                signal_idler_test_index,
                plot_bool=False,
            )
            loop_time = perf_counter() - loop_start
            if print_bool:
                tqdm.write(
                    f"{iteration_header} | validate={overlap_new:.6f} elapsed={loop_time:.2f}s"
                )
            self.gf.debug_print(
                f"Validation overlap at center of new Green's function: {overlap_new}"
            )

            if overlap_new < 0.025:
                if print_bool:
                    tqdm.write(f"{iteration_header} | done (overlap approaching zero)")
                    tqdm.write("#################################################\n")
                return green_functions, time_width_array, freq_width_array, stitch_times

            width_offset_from_global_center += (
                width_offset_from_new_center * step_fraction
            )
            width = width_new
            green_functions = updated_green_functions
            center_time = center_time_new
            time_width_array = time_width_out
            freq_width_array = freq_width_out

        return green_functions, time_width_array, freq_width_array, stitch_times

    def run_full_stitch(
        self,
        t0: float,
        validation_threshold: float = 0.95,
        kmax_pilot: int | None = None,
        sv_threshold: float = 0.01,
        step_fraction: float = 1.0,
        stitch_skip_threshold: float = 0.95,
        print_bool: bool = False,
    ) -> tuple[tuple[ComplexArray, ...], WidthCollection, WidthCollection, list[float]]:
        """Perform full extraction and stitching in both directions and return results.

        Parameters
        ----------
        step_fraction:
            Multiplier on the auto-detected per-step advance.  The default
            (1.0) steps by the detected GF support width.  For narrow basis
            functions (small ``basis_width``) this can be very small; increase
            ``step_fraction`` (e.g. 3–10) to take larger strides and reduce the
            total number of iterations.
        stitch_skip_threshold:
            When *both* the new and old GF overlaps at the proposed stitch
            point exceed this value, accept the stitch immediately without
            triggering a second extraction.  Set to 1.0 to disable (original
            behaviour) or 0.95 to skip the probe whenever both GFs are already
            high-fidelity there.
        """
        initial_center_time = 0
        self.gf.make_pump(self.gf.solver_object.make_gaussian_input(self.T0p))

        if print_bool:
            tqdm.write("Initial extraction: building Green's functions...")

        extraction_result = self.extract_green_functions(
            t0, -initial_center_time, check_bool=False
        )
        a_test_init = self.make_test_function(
            float(extraction_result.init_offset[0]), t0
        )
        validation_bool, init_overlap = self.validate_propagation(
            extraction_result.green_functions[0],
            a_test_init,
            validation_threshold,
            plot_bool=False,
        )
        # if not validation_bool:
        #     raise RuntimeError(
        #         "Validation failed with the requested threshold. "
        #         "Consider increasing the number of basis functions or the time resolution. "
        #         f"Initial overlap: {init_overlap:.6f}"
        #     )
        if print_bool:
            tqdm.write(
                "Initial extraction complete. Starting iterative stitching (pass 1)..."
            )

        green_functions, time_width_array, freq_width_array, stitch_times_1 = (
            self.iterative_stitch(
                extraction_result.green_functions,
                extraction_result.init_offset,
                extraction_result.width_offset_from_global_center,
                extraction_result.width,
                validation_threshold,
                extraction_result.time_indices,
                extraction_result.freq_indices,
                0,
                phase_label="pass 1",
                step_fraction=step_fraction,
                stitch_skip_threshold=stitch_skip_threshold,
                print_bool=print_bool,
            )
        )

        if print_bool:
            tqdm.write("Starting iterative stitching (pass 2)...")

        green_functions, time_width_array, freq_width_array, stitch_times_2 = (
            self.iterative_stitch(
                green_functions,
                extraction_result.init_offset,
                extraction_result.width_offset_from_global_center,
                extraction_result.width,
                validation_threshold,
                time_width_array,
                freq_width_array,
                1,
                phase_label="pass 2",
                step_fraction=step_fraction,
                stitch_skip_threshold=stitch_skip_threshold,
                print_bool=print_bool,
            )
        )

        stitch_times = stitch_times_1 + stitch_times_2
        return green_functions, time_width_array, freq_width_array, stitch_times

    def save_stitch_output(
        self,
        filename: str | None,
        green_functions: tuple[ComplexArray, ...],
        time_width_array: WidthCollection,
        freq_width_array: WidthCollection,
        stitch_times: list[float],
        parameter_array: Union[Sequence[Any], StitcherParameters],
    ) -> str:
        """Save stitch results to a .npy file and return the filename used.

        If `filename` is None a sensible default name is created from the
        parameter array and object state.
        """
        time_width_array_output = self.output_width_array(time_width_array)
        freq_width_array_output = self.output_width_array(freq_width_array)

        if filename is None:
            beta = self.gf.solver_object.beta
            type_string = "type0" if getattr(beta, "QPMbool", False) else "typeII"
            gamma: Any
            length: Any
            if isinstance(parameter_array, StitcherParameters):
                gamma = parameter_array.gamma
                length = parameter_array.length
            else:
                gamma = parameter_array[5] if len(parameter_array) > 5 else ""
                length = parameter_array[3] if len(parameter_array) > 3 else ""
            filename = f"stitchedGreens_{type_string}_gamma {gamma}_T0p {self.T0p}_L {length}.npy"

        save_array = np.array(
            [
                green_functions,
                time_width_array_output,
                freq_width_array_output,
                stitch_times,
                self.t,
                self.omega,
                self.lambda_axis,
                parameter_array,
            ],
            dtype=object,
        )

        np.save(filename, save_array)
        return filename

    extractGreenFunctions = extract_green_functions
    findWidth = find_width
    widthHelperFunction = width_helper_function
    testOverlap = test_overlap
    validatePropagation = validate_propagation
    makeTestFunction = make_test_function
    removeZeroValues = remove_zero_values
    stitchGreenFunctions = stitch_green_functions
    stitchHelperFunction = stitch_helper_function
    compareWidthArraysHelperFunction = compare_width_arrays_helper_function
    compareWidthArrays = compare_width_arrays
    addPaddingToWidth = add_padding_to_width
    outputWidthArray = output_width_array
    stitchTimeHelper = stitch_time_helper
    iterativeStitch = iterative_stitch
