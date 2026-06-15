from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from typing import Any, Sequence

import numpy as np
from numpy.linalg import svd
from numpy.typing import NDArray

from .solver import CoupledModes


ComplexArray = NDArray[Any]
FloatArray = NDArray[np.float64]
RealArray = NDArray[np.float64]


@dataclass(slots=True)
class ExtractTask:
    basis_order: int
    basis_index: int
    basis_slice: ComplexArray
    a_noise: ComplexArray
    pump: ComplexArray
    out_index: int
    in_index: int


@dataclass(slots=True)
class OverlapTask:
    g_cross: ComplexArray
    g_self: ComplexArray
    input_field: ComplexArray
    output_field: ComplexArray
    propagated_input: ComplexArray


def _parallel_green(
    args: tuple[ComplexArray, FloatArray, ComplexArray],
) -> ComplexArray:
    v_modes, rho_values, u_modes = args
    return np.asarray(
        (v_modes * rho_values[:, np.newaxis]).T @ np.conjugate(u_modes), dtype=complex
    )


def _make_pool():
    start_methods = get_all_start_methods()
    context_name = "forkserver" if "forkserver" in start_methods else "spawn"
    return get_context(context_name).Pool()


def _parallel_extract(
    args: tuple[ExtractTask, Sequence[Any]],
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    task, parameter_array = args
    cnlse = CoupledModes(*parameter_array)
    if task.basis_index == 0:
        init_conditions = np.array([task.basis_slice, task.a_noise, task.pump])
    elif task.basis_index == 1:
        init_conditions = np.array([task.a_noise, task.basis_slice, task.pump])
    else:
        raise ValueError(f"Unsupported basis index {task.basis_index}")

    cnlse.set_initial_conditions(init_conditions)
    _, _, _, _, _, field_time = cnlse.run()
    input_field = field_time[0, :, :]
    output_field = field_time[-1, :, :]
    # print(f"Finished k = {task.basis_order}")
    return (
        input_field[:, task.in_index],
        output_field[:, task.out_index],
        output_field[:, task.in_index],
    )


def _parallel_overlap(args: tuple[OverlapTask, float]) -> tuple[float, float]:
    task, dt = args
    green_output = np.sum(task.g_cross * task.input_field, axis=1) * dt
    green_output = green_output / np.sqrt(np.sum(np.abs(green_output) ** 2) * dt)
    output_field = task.output_field / np.sqrt(
        np.sum(np.abs(task.output_field) ** 2) * dt
    )
    overlap_cross = float(
        np.abs(np.sum(np.conjugate(green_output) * output_field) * dt) ** 2
    )

    input_conjugate = np.conjugate(task.input_field)
    green_output = np.sum(task.g_self * input_conjugate, axis=1) * dt
    green_output = green_output / np.sqrt(np.sum(np.abs(green_output) ** 2) * dt)
    propagated_input = task.propagated_input / np.sqrt(
        np.sum(np.abs(task.propagated_input) ** 2) * dt
    )
    overlap_self = float(
        np.abs(np.sum(np.conjugate(green_output) * propagated_input) * dt) ** 2
    )
    return overlap_cross, overlap_self


class GreenFunctionsExtractor:
    def __init__(self, kmax: int, debug_bool: bool = True) -> None:
        self.kmax = kmax
        self.debug_bool = debug_bool

    def fft(self, field: ComplexArray) -> ComplexArray:
        field = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(field, axes=0), axis=0), axes=0
        )
        field = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(field, axes=1), axis=1), axes=1
        )
        return field

    def ifft(self, field: ComplexArray) -> ComplexArray:
        field = np.fft.ifftshift(
            np.fft.ifft(np.fft.fftshift(field, axes=0), axis=0), axes=0
        )
        field = np.fft.ifftshift(
            np.fft.ifft(np.fft.fftshift(field, axes=1), axis=1), axes=1
        )
        return field

    def debug_print(self, *args: object, **kwargs: Any) -> None:
        if self.debug_bool:
            print(*args, **kwargs)

    def make_solver_parameters(
        self, parameters_array: Sequence[Any], solver_object: CoupledModes
    ) -> None:
        self.parameters_array = list(parameters_array)
        self.solver_object = solver_object
        self.time_len = self.solver_object.N
        self.t = self.solver_object.t
        self.omega = self.solver_object.omega
        self.lambda_axis = self.solver_object.lambdaRealS
        self.dt = self.solver_object.dt
        self.domega = self.solver_object.domega
        self.time_shift_array = self.solver_object.timeShiftArray

    def make_pump(self, field: ComplexArray) -> None:
        self.Ap_0 = field

    def make_basis_functions(
        self, t0: float, basis_function_offset: float = 0.0
    ) -> None:
        self.A_basis = np.zeros((self.kmax, self.time_len, 2), dtype=complex)
        self.T0 = t0
        self.basis_function_offset = basis_function_offset

        beta_s_pl = self.solver_object.timeShiftArray[0] / 2
        beta_i_pl = self.solver_object.timeShiftArray[1] / 2
        self.init_offset = np.array(
            [-beta_s_pl + basis_function_offset, -beta_i_pl + basis_function_offset],
            dtype=float,
        )

        for basis_order in range(self.kmax):
            self.A_basis[basis_order, :, 0] = (
                self.solver_object.make_hermite_gaussian_basis_functions(
                    self.init_offset[0],
                    t0,
                    basis_order,
                )
            )
            self.A_basis[basis_order, :, 1] = (
                self.solver_object.make_hermite_gaussian_basis_functions(
                    self.init_offset[1],
                    t0,
                    basis_order,
                )
            )

    def run_cnlse(
        self, init_conditions: NDArray[np.complex128], cnlse: CoupledModes
    ) -> tuple[ComplexArray, ComplexArray]:
        cnlse.set_initial_conditions(init_conditions)
        _, _, _, _, _, field_time = cnlse.run()
        return field_time[0, :, :], field_time[-1, :, :]

    def extract_schmidt_modes(
        self,
        basis_index: int,
    ) -> tuple[
        ComplexArray,
        FloatArray,
        ComplexArray,
        ComplexArray,
        FloatArray,
        ComplexArray,
        ComplexArray,
        float,
        list[ComplexArray],
    ]:
        pump = self.Ap_0
        a_noise = np.zeros_like(pump)

        cross_time = self.time_shift_array[int(not basis_index)] / 2
        self_time = self.time_shift_array[basis_index] / 2

        in_index = int(basis_index)
        out_index = int(not basis_index)
        tasks = [
            ExtractTask(
                basis_order=basis_order,
                basis_index=basis_index,
                basis_slice=self.A_basis[basis_order, :, basis_index],
                a_noise=a_noise,
                pump=pump,
                out_index=out_index,
                in_index=in_index,
            )
            for basis_order in range(self.kmax)
        ]

        with _make_pool() as pool:
            results = pool.map(
                _parallel_extract, [(task, self.parameters_array) for task in tasks]
            )

        b_cross = np.zeros((self.kmax, self.time_len), dtype=complex)
        b_self = np.zeros((self.kmax, self.time_len), dtype=complex)
        for basis_order in range(self.kmax):
            b_cross[basis_order, :] = (
                self.solver_object.make_hermite_gaussian_basis_functions(
                    cross_time + self.basis_function_offset,
                    self.T0,
                    basis_order,
                    fft_bool=False,
                )
            )
            b_self[basis_order, :] = (
                self.solver_object.make_hermite_gaussian_basis_functions(
                    self_time + self.basis_function_offset,
                    self.T0,
                    basis_order,
                    fft_bool=False,
                )
            )

        field_time_array = np.zeros((3, self.kmax, self.time_len), dtype=complex)
        (
            field_time_array[0, :, :],
            field_time_array[1, :, :],
            field_time_array[2, :, :],
        ) = zip(*results)

        g_cross = np.dot(field_time_array[1], np.conjugate(b_cross.T)) * self.dt
        g_self = np.dot(field_time_array[2], np.conjugate(b_self.T)) * self.dt
        self.debug_print("")

        u_cross, rho_cross, v_cross_conjugate = svd(g_cross)
        v_cross = np.conjugate(v_cross_conjugate).T

        u_self, rho_self, v_self_conjugate = svd(g_self)
        v_self = np.conjugate(v_self_conjugate).T

        u_cross = np.conjugate(u_cross)
        u_self = np.conjugate(u_self)
        v_cross = np.conjugate(v_cross)
        v_self = np.conjugate(v_self)

        phi = np.matmul(u_cross.T, b_self)
        phi_self = np.matmul(u_self.T, b_self)
        psi = np.matmul(v_cross.T, b_cross)
        psi_self = np.matmul(v_self.T, b_self)
        output_basis = [b_cross, b_self]
        self.field_time_array = field_time_array
        return (
            phi,
            rho_cross,
            psi,
            phi_self,
            rho_self,
            psi_self,
            field_time_array,
            float(cross_time),
            output_basis,
        )

    def extract_green_functions(
        self,
        args: tuple[Any, ...],
        indistinguishable_bool: bool,
    ) -> tuple[ComplexArray, ...]:
        self.debug_print("Extracting Green's functions...", end="\n")
        if indistinguishable_bool:
            us, vi, uss, vss, rho_cross, rho_self = args
            args_list = [(vi, rho_cross, us), (vss, rho_self, uss)]
        else:
            us, vs, ui, vi, uss, vss, uii, vii, rho_cross, rho_self = args
            args_list = [
                (vi, rho_cross, us),
                (vii, rho_self, uii),
                (vs, rho_cross, ui),
                (vss, rho_self, uss),
            ]

        with _make_pool() as pool:
            results = pool.map(_parallel_green, args_list)
        self.debug_print("Green's functions extracted.")
        return tuple(results)

    def photon_number(self, cross_green_time: ComplexArray) -> float:
        n_t = np.sum(np.abs(cross_green_time) ** 2, axis=1)
        return float(np.sum(n_t) * self.dt)

    def calc_overlap(
        self, green_output: ComplexArray, output_field: ComplexArray
    ) -> float:
        green_output = green_output / np.sqrt(
            np.sum(np.abs(green_output) ** 2) * self.dt
        )
        output_field = output_field / np.sqrt(
            np.sum(np.abs(output_field) ** 2) * self.dt
        )
        overlap = np.sum(np.conjugate(green_output) * output_field) * self.dt
        return float(np.abs(overlap) ** 2)

    def calculate_green_overlap(
        self,
        g_cross: ComplexArray,
        g_self: ComplexArray,
        field_array: ComplexArray,
    ) -> RealArray:
        overlap_array = np.zeros((field_array.shape[1], 2), dtype=float)
        tasks = [
            OverlapTask(
                g_cross,
                g_self,
                field_array[0, index, :],
                field_array[1, index, :],
                field_array[2, index, :],
            )
            for index in range(len(overlap_array))
        ]
        with _make_pool() as pool:
            results = pool.map(_parallel_overlap, [(task, self.dt) for task in tasks])
        overlap_array[:, :] = np.array(results, dtype=float)
        return overlap_array

    def check_schmidt_numbers(self, rho: FloatArray, nu: FloatArray) -> FloatArray:
        return rho**2 - nu**2

    def run_extractor(
        self,
        indistinguishable_bool: bool,
        check_bool: bool,
    ) -> tuple[
        tuple[ComplexArray, ...], RealArray | None, FloatArray | RealArray | None
    ]:
        overlaps: RealArray | None = None
        schmidt_numbers: FloatArray | RealArray | None = None
        g_tuple: tuple[ComplexArray, ...] = ()

        self.debug_print("\nPropagating signal...", end="\n")
        u_s, idler_rho, v_i, uss, self_rhos, vss, is_time_array, ti, signal_basis = (
            self.extract_schmidt_modes(0)
        )
        self.t_signal_propagated = 2 * ti

        if not indistinguishable_bool:
            self.debug_print("Propagating idler...", end="\n")
            (
                u_i,
                signal_rho,
                v_s,
                uii,
                self_rho_i,
                vii,
                si_time_array,
                ts,
                idler_basis,
            ) = self.extract_schmidt_modes(1)
            self.t_idler_propagated = 2 * ts

            args_tuple_full: tuple[Any, ...] = (
                u_s,
                v_s,
                u_i,
                v_i,
                uss,
                vss,
                uii,
                vii,
                signal_rho,
                self_rhos,
            )
            result_full = self.extract_green_functions(
                args_tuple_full, indistinguishable_bool
            )
            g_is = result_full[0]
            g_ii = result_full[1]
            g_si = result_full[2]
            g_ss = result_full[3]
            g_is = np.roll(g_is, round((2 * ts) / self.dt), axis=1)
            g_ss = np.roll(g_ss, round((2 * ts) / self.dt), axis=1)
            g_ii = np.roll(g_ii, round((2 * ti) / self.dt), axis=1)
            g_si = np.roll(g_si, round((2 * ti) / self.dt), axis=1)
            g_tuple = (g_is, g_ii, g_si, g_ss)

            if check_bool:
                print(f"Photon number from G_is: {self.photon_number(g_is)}")
                print(f"Photon number from G_si: {self.photon_number(g_si)}")
                print(
                    f"Absolute difference: {np.abs(self.photon_number(g_is) - self.photon_number(g_si))}"
                )
                print("Calculating overlaps and Schmidt numbers...")

                overlaps_signal = self.calculate_green_overlap(
                    g_si, g_ii, si_time_array
                )
                overlaps_idler = self.calculate_green_overlap(g_is, g_ss, is_time_array)
                overlaps = np.concatenate((overlaps_signal, overlaps_idler), axis=1)

                schmidt_signal = self.check_schmidt_numbers(self_rhos, idler_rho)
                schmidt_idler = self.check_schmidt_numbers(self_rho_i, signal_rho)
                schmidt_numbers = np.vstack((schmidt_signal, schmidt_idler)).T
                print("Finished!")
        else:
            args_tuple_simple: tuple[Any, ...] = (
                u_s,
                v_i,
                uss,
                vss,
                idler_rho,
                self_rhos,
            )
            result_simple = self.extract_green_functions(
                args_tuple_simple, indistinguishable_bool
            )
            g_field = result_simple[0]
            f_field = result_simple[1]
            g_field = np.roll(g_field, round((2 * ti) / self.dt), axis=1)
            f_field = np.roll(f_field, round((2 * ti) / self.dt), axis=1)
            g_tuple = (g_field, f_field)

            if check_bool:
                print(f"Photon number from G: {self.photon_number(g_field)}")
                print("Calculating overlaps and Schmidt numbers...")
                overlaps = self.calculate_green_overlap(g_field, f_field, is_time_array)
                schmidt_numbers = self.check_schmidt_numbers(self_rhos, idler_rho)
                print("Finished!")

        return g_tuple, overlaps, schmidt_numbers

    makeSolverParameters = make_solver_parameters
    makePump = make_pump
    makeBasisFunctions = make_basis_functions
    runCNLSE = run_cnlse
    debugPrint = debug_print
    extractSchmidtModes = extract_schmidt_modes
    extractGreenFunctions = extract_green_functions
    calculateGreenOverlap = calculate_green_overlap
    checkSchmidtNumbers = check_schmidt_numbers
    runExtractor = run_extractor
    photonNumber = photon_number
    calcOverlap = calc_overlap
