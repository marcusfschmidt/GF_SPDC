from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
import sys
from time import perf_counter
from typing import Any, Sequence, cast

import numpy as np
from numpy.linalg import svd
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore[import-untyped]

from .solver import CoupledModes, SolverParameterTuple


ComplexArray = NDArray[Any]
FloatArray = NDArray[np.float64]
RealArray = NDArray[np.float64]


@dataclass(slots=True)
class ExtractTask:
    basis_order: int
    basis_index: int


def _parallel_green(
    args: tuple[ComplexArray, FloatArray, ComplexArray] | tuple[ComplexArray, FloatArray, ComplexArray, int],
) -> ComplexArray:
    if len(args) == 3:
        v_modes, rho_values, u_modes = args
        input_shift = 0
    else:
        v_modes, rho_values, u_modes, input_shift = args
    if input_shift != 0:
        u_modes = np.roll(u_modes, input_shift, axis=1)
    return np.asarray(
        (v_modes * rho_values[:, np.newaxis]).T @ u_modes.conj(), dtype=complex
    )


def _make_pool(initializer=None, initargs: tuple[Any, ...] = ()):
    start_methods = get_all_start_methods()
    if sys.platform == "win32":
        context_name = "spawn"
    elif "fork" in start_methods:
        context_name = "fork"
    elif "forkserver" in start_methods:
        context_name = "forkserver"
    else:
        context_name = "spawn"
    return get_context(context_name).Pool(initializer=initializer, initargs=initargs)


_WORKER_SOLVER: CoupledModes | None = None
_WORKER_BASIS: ComplexArray | None = None
_WORKER_NOISE: ComplexArray | None = None
_WORKER_PUMP: ComplexArray | None = None


def _init_worker_solver(
    parameter_array: Sequence[Any],
    time_offset: float,
    basis: ComplexArray,
    pump: ComplexArray,
) -> None:
    global _WORKER_SOLVER, _WORKER_BASIS, _WORKER_NOISE, _WORKER_PUMP
    _WORKER_SOLVER = CoupledModes(*parameter_array, time_offset=time_offset)
    _WORKER_BASIS = basis
    _WORKER_PUMP = pump
    _WORKER_NOISE = np.zeros_like(pump)


def _parallel_extract(
    args: ExtractTask,
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    task = args
    global _WORKER_SOLVER, _WORKER_BASIS, _WORKER_NOISE, _WORKER_PUMP
    if (
        _WORKER_SOLVER is None
        or _WORKER_BASIS is None
        or _WORKER_NOISE is None
        or _WORKER_PUMP is None
    ):
        raise RuntimeError("Worker solver was not initialized.")
    cnlse = _WORKER_SOLVER
    basis_slice = _WORKER_BASIS[task.basis_order, :, task.basis_index]
    if task.basis_index == 0:
        init_conditions = np.array([basis_slice, _WORKER_NOISE, _WORKER_PUMP])
    elif task.basis_index == 1:
        init_conditions = np.array([_WORKER_NOISE, basis_slice, _WORKER_PUMP])
    else:
        raise ValueError(f"Unsupported basis index {task.basis_index}")

    cnlse.set_initial_conditions(init_conditions)
    input_field, output_field = cnlse.run_final_only()
    in_index = int(task.basis_index)
    out_index = int(not task.basis_index)
    return (
        input_field[:, in_index],
        output_field[:, out_index],
        output_field[:, in_index],
    )


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
        self.parameters_array = cast(SolverParameterTuple, tuple(parameters_array))
        self.solver_object = solver_object
        self.time_offset = float(getattr(solver_object, "time_offset", 0.0))
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
        pool: Any | None = None,
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
        extraction_start = perf_counter()

        cross_time = self.time_shift_array[int(not basis_index)] / 2
        self_time = self.time_shift_array[basis_index] / 2

        task_build_start = perf_counter()
        tasks = [
            ExtractTask(
                basis_order=basis_order,
                basis_index=basis_index,
            )
            for basis_order in range(self.kmax)
        ]
        task_build_time = perf_counter() - task_build_start

        worker_start = perf_counter()
        if pool is None:
            with _make_pool(
                _init_worker_solver,
                (self.parameters_array, self.time_offset, self.A_basis, self.Ap_0),
            ) as local_pool:
                results = list(
                    tqdm(
                        local_pool.imap(
                            _parallel_extract,
                            tasks,
                        ),
                        total=len(tasks),
                        desc="Extracting modes",
                    )
                )
        else:
            results = list(
                tqdm(
                    pool.imap(
                        _parallel_extract,
                        tasks,
                    ),
                    total=len(tasks),
                    desc="Extracting modes",
                )
            )
        worker_time = perf_counter() - worker_start

        basis_start = perf_counter()
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
        basis_time = perf_counter() - basis_start

        projection_start = perf_counter()
        field_time_array = np.zeros((3, self.kmax, self.time_len), dtype=complex)
        (
            field_time_array[0, :, :],
            field_time_array[1, :, :],
            field_time_array[2, :, :],
        ) = zip(*results)

        b_cross_T = b_cross.T.conj()
        b_self_T = b_self.T.conj()

        g_cross = np.dot(field_time_array[1], b_cross_T) * self.dt
        g_self = np.dot(field_time_array[2], b_self_T) * self.dt

        u_cross, rho_cross, v_cross_conjugate = svd(g_cross)
        v_cross = v_cross_conjugate.conj().T

        u_self, rho_self, v_self_conjugate = svd(g_self)
        v_self = v_self_conjugate.conj().T

        u_cross = u_cross.conj()
        u_self = u_self.conj()
        v_cross = v_cross.conj()
        v_self = v_self.conj()

        phi = np.matmul(u_cross.T, b_self)
        phi_self = np.matmul(u_self.T, b_self)
        psi = np.matmul(v_cross.T, b_cross)
        psi_self = np.matmul(v_self.T, b_self)
        output_basis = [b_cross, b_self]
        self.field_time_array = field_time_array
        projection_time = perf_counter() - projection_start
        total_time = perf_counter() - extraction_start
        tqdm.write(
            (
                f"Extraction timing: task setup {task_build_time:.2f}s, "
                f"workers {worker_time:.2f}s, basis {basis_time:.2f}s, "
                f"projection/SVD {projection_time:.2f}s, total {total_time:.2f}s"
            )
        )
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
        input_shifts: tuple[int, ...],
    ) -> tuple[ComplexArray, ...]:
        self.debug_print("Extracting Green's functions...", end="\n")
        if indistinguishable_bool:
            us, vi, uss, vss, rho_cross, rho_self = args
            args_list = [
                (vi, rho_cross, us, input_shifts[0]),
                (vss, rho_self, uss, input_shifts[1]),
            ]
        else:
            us, vs, ui, vi, uss, vss, uii, vii, rho_cross, rho_self = args
            args_list = [
                (vi, rho_cross, us, input_shifts[0]),
                (vii, rho_self, uii, input_shifts[1]),
                (vs, rho_cross, ui, input_shifts[2]),
                (vss, rho_self, uss, input_shifts[3]),
            ]

        results = [_parallel_green(arg_set) for arg_set in args_list]
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
        overlap = np.sum(green_output.conj() * output_field) * self.dt
        return float(np.abs(overlap) ** 2)

    def calculate_green_overlap(
        self,
        g_cross: ComplexArray,
        g_self: ComplexArray,
        field_array: ComplexArray,
    ) -> RealArray:
        input_fields = field_array[0]
        output_fields = field_array[1]
        propagated_inputs = field_array[2]

        green_cross = (g_cross @ input_fields.T).T * self.dt
        green_self = (g_self @ input_fields.conj().T).T * self.dt

        green_cross /= np.sqrt(
            np.sum(np.abs(green_cross) ** 2, axis=1, keepdims=True) * self.dt
        )
        green_self /= np.sqrt(
            np.sum(np.abs(green_self) ** 2, axis=1, keepdims=True) * self.dt
        )
        output_fields_normalized = output_fields / np.sqrt(
            np.sum(np.abs(output_fields) ** 2, axis=1, keepdims=True) * self.dt
        )
        propagated_inputs_normalized = propagated_inputs / np.sqrt(
            np.sum(np.abs(propagated_inputs) ** 2, axis=1, keepdims=True) * self.dt
        )

        overlap_cross = np.abs(
            np.sum(green_cross.conj() * output_fields_normalized, axis=1) * self.dt
        ) ** 2
        overlap_self = np.abs(
            np.sum(green_self.conj() * propagated_inputs_normalized, axis=1) * self.dt
        ) ** 2

        return np.column_stack((overlap_cross, overlap_self)).astype(float, copy=False)

    def check_schmidt_numbers(self, rho: FloatArray, nu: FloatArray) -> FloatArray:
        return rho**2 - nu**2

    def run_extractor(
        self,
        indistinguishable_bool: bool,
        check_bool: bool,
    ) -> tuple[
        tuple[ComplexArray, ...], RealArray | None, FloatArray | RealArray | None
    ]:
        run_start = perf_counter()
        overlaps: RealArray | None = None
        schmidt_numbers: FloatArray | RealArray | None = None
        g_tuple: tuple[ComplexArray, ...] = ()

        with _make_pool(
            _init_worker_solver,
            (self.parameters_array, self.time_offset, self.A_basis, self.Ap_0),
        ) as pool:
            self.debug_print("\nPropagating signal...", end="\n")
            signal_start = perf_counter()
            u_s, idler_rho, v_i, uss, self_rhos, vss, is_time_array, ti, signal_basis = (
                self.extract_schmidt_modes(0, pool)
            )
            signal_time = perf_counter() - signal_start
            self.t_signal_propagated = 2 * ti

            if not indistinguishable_bool:
                self.debug_print("Propagating idler...", end="\n")
                idler_start = perf_counter()
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
                ) = self.extract_schmidt_modes(1, pool)
                idler_time = perf_counter() - idler_start
                self.t_idler_propagated = 2 * ts

                assemble_start = perf_counter()
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
                input_shifts = (
                    round((2 * ts) / self.dt),
                    round((2 * ti) / self.dt),
                    round((2 * ti) / self.dt),
                    round((2 * ts) / self.dt),
                )
                green_start = perf_counter()
                result_full = self.extract_green_functions(
                    args_tuple_full, indistinguishable_bool, input_shifts
                )
                green_time = perf_counter() - green_start
                g_is = result_full[0]
                g_ii = result_full[1]
                g_si = result_full[2]
                g_ss = result_full[3]
                g_tuple = (g_is, g_ii, g_si, g_ss)
                assemble_time = perf_counter() - assemble_start

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
                tqdm.write(
                    (
                        f"Run extractor: signal {signal_time:.2f}s, idler {idler_time:.2f}s, "
                        f"green {green_time:.2f}s, assemble {assemble_time:.2f}s, "
                        f"total {perf_counter() - run_start:.2f}s"
                    )
                )
            else:
                assemble_start = perf_counter()
                args_tuple_simple: tuple[Any, ...] = (
                    u_s,
                    v_i,
                    uss,
                    vss,
                    idler_rho,
                    self_rhos,
                )
                input_shifts = (
                    round((2 * ti) / self.dt),
                    round((2 * ti) / self.dt),
                )
                green_start = perf_counter()
                result_simple = self.extract_green_functions(
                    args_tuple_simple, indistinguishable_bool, input_shifts
                )
                green_time = perf_counter() - green_start
                g_field = result_simple[0]
                f_field = result_simple[1]
                g_tuple = (g_field, f_field)
                assemble_time = perf_counter() - assemble_start

                if check_bool:
                    print(f"Photon number from G: {self.photon_number(g_field)}")
                    print("Calculating overlaps and Schmidt numbers...")
                    overlaps = self.calculate_green_overlap(g_field, f_field, is_time_array)
                    schmidt_numbers = self.check_schmidt_numbers(self_rhos, idler_rho)
                    print("Finished!")
                tqdm.write(
                    (
                        f"Run extractor: signal {signal_time:.2f}s, green {green_time:.2f}s, "
                        f"assemble {assemble_time:.2f}s, "
                        f"total {perf_counter() - run_start:.2f}s"
                    )
                )

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
