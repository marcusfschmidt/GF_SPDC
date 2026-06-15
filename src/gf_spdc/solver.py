from __future__ import annotations

import time
from math import factorial
from typing import Any, Protocol, Sequence, cast

import numpy as np
import scipy as sp  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy import special  # type: ignore[import-untyped]
from scipy.integrate import ode  # type: ignore[import-untyped]
from scipy.interpolate import InterpolatedUnivariateSpline  # type: ignore[import-untyped]

from .mgo_lithium_niobate_type0_beta import MgOLithiumNiobateType0
from .mgo_lithium_niobate_type2_beta import MgOLithiumNiobateType2


ComplexArray = NDArray[Any]
FloatArray = NDArray[np.float64]


class BetaModel(Protocol):
    QPMbool: bool
    QPMPeriod: float
    indistinguishableBool: bool
    kp: FloatArray
    ks: FloatArray
    ki: FloatArray
    om: FloatArray


TaylorBeta = Sequence[Sequence[float]]
SolverParameterTuple = tuple[
    int,
    float,
    float,
    float,
    BetaModel | Sequence[Sequence[float]],
    float,
    float,
    float,
    float,
    float,
    float,
    bool,
    float,
    int,
]

BETA_MODEL_TYPES = (MgOLithiumNiobateType2, MgOLithiumNiobateType0)


class CoupledModes:
    def __init__(
        self,
        n: int,
        dt: float,
        dz: float,
        length: float,
        beta: BetaModel | Sequence[Sequence[float]],
        gamma: float,
        lambda_p: float,
        omega_s: float = 0.0,
        omega_i: float = 0.0,
        alpha_s: float = 0.0,
        alpha_i: float = 0.0,
        print_bool: bool = False,
        rtol: float = 1e-3,
        nsteps: int = 10000,
        *,
        integration_method: str = "rk4",
    ) -> None:
        self.N = 2 ** int(n)
        self.dt = dt
        self.dz = dz
        self.L = length
        self.beta = beta
        self.gamma = gamma
        self.lambda_p = lambda_p
        self.alpha_s = alpha_s
        self.alpha_i = alpha_i
        self.initial_conditions_flag = False
        self.print_bool = print_bool
        self.integration_method = integration_method.lower()
        if self.integration_method not in {"rk4", "adaptive"}:
            raise ValueError("integration_method must be 'rk4' or 'adaptive'.")

        self.QPMPeriod = 0.0
        self.QPM_bool = (
            bool(cast(BetaModel, beta).QPMbool)
            if isinstance(beta, BETA_MODEL_TYPES)
            else False
        )
        if self.QPM_bool:
            self.QPMPeriod = float(getattr(cast(BetaModel, beta), "QPMPeriod", 0.0))

        self.solver = ode(self.ode_nl)
        self.solver.set_integrator("dopri5", rtol=rtol, nsteps=nsteps)

        self.c = 299792458e-12
        self.hbar = 1.054571817e-34 * 1e12

        self.t = np.arange(-self.N / 2, self.N / 2, dtype=float) * self.dt
        self.domega = 2 * np.pi / (self.N * self.dt)
        self.omega = np.arange(-self.N / 2, self.N / 2, dtype=float) * self.domega

        self.fp = self.c / self.lambda_p
        self.omega_p = 2 * np.pi * self.fp

        self.omega_real = self.omega + self.omega_p
        self.omega_s = self.omega_p + omega_s
        self.omega_i = self.omega_p + omega_i
        self.omegaRealS = self.omega + self.omega_s
        self.omegaRealI = self.omega + self.omega_i

        self.lambdaReal = self.c / ((self.omega + self.omega_p) / (2 * np.pi)) * 1e9
        self.lambdaRealS = self.c / ((self.omega + self.omega_s) / (2 * np.pi)) * 1e9
        self.lambdaRealI = self.c / ((self.omega + self.omega_i) / (2 * np.pi)) * 1e9
        self.lamReal = self.lambdaReal

        if self.omega_real[0] < 0:
            raise ValueError(
                "Center wavelength too high for the omega-array to be non-negative. "
                "Try increasing the frequency or increasing dt."
            )

        self.kp_fit: InterpolatedUnivariateSpline | None = None
        kp_full: FloatArray
        ks_full: FloatArray
        ki_full: FloatArray
        if isinstance(beta, BETA_MODEL_TYPES):
            beta_model = cast(BetaModel, beta)
            kp = np.asarray(beta_model.kp, dtype=float)
            ks = np.asarray(beta_model.ks, dtype=float)
            ki = np.asarray(beta_model.ki, dtype=float)
            omega_axis = np.asarray(beta_model.om, dtype=float)

            kp_fit = InterpolatedUnivariateSpline(omega_axis, kp)
            self.kp_fit = kp_fit
            kp_full = np.asarray(kp_fit(self.omega_real), dtype=float)
            beta1 = float(cast(Any, kp_fit.derivative()(self.omega_p)))
            self.beta1 = beta1
            self.kpFull = kp_full

            ks_fit = InterpolatedUnivariateSpline(omega_axis, ks)
            ks_full = np.asarray(ks_fit(self.omegaRealS), dtype=float)
            self.ks = float(cast(Any, ks_fit.derivative()(self.omega_s)))
            self.ksFull = ks_full

            ki_fit = InterpolatedUnivariateSpline(omega_axis, ki)
            ki_full = np.asarray(ki_fit(self.omegaRealI), dtype=float)
            self.ki = float(cast(Any, ki_fit.derivative()(self.omega_i)))
            self.kiFull = ki_full

            self.k_reference = beta1
        else:
            beta_coefficients = cast(TaylorBeta, beta)
            self.k_reference = float(beta_coefficients[0][1])
            kp_full = self.make_simple_beta(
                beta_coefficients[0], expansion_point=self.omega_p
            )
            ks_full = self.make_simple_beta(
                beta_coefficients[1], expansion_point=self.omega_s
            )
            self.ks = float(beta_coefficients[1][1])
            ki_full = self.make_simple_beta(
                beta_coefficients[2], expansion_point=self.omega_i
            )
            self.ki = float(beta_coefficients[2][1])

        self.betap = self.transform_beta(
            np.asarray(kp_full, dtype=float),
            self.k_reference,
            self.omega_real,
            self.omega_p,
        )
        self.betas = self.transform_beta(
            np.asarray(ks_full, dtype=float),
            self.k_reference,
            self.omegaRealS,
            self.omega_s,
        )
        self.betai = self.transform_beta(
            np.asarray(ki_full, dtype=float),
            self.k_reference,
            self.omegaRealI,
            self.omega_i,
        )

        time_shift_s = (self.ks - self.k_reference) * self.L
        time_shift_i = (self.ki - self.k_reference) * self.L
        self.timeShiftArray = np.array([time_shift_s, time_shift_i], dtype=float)

    def fft(self, field: ComplexArray) -> ComplexArray:
        return sp.fft.fftshift(sp.fft.fft(sp.fft.ifftshift(field)))

    def ifft(self, field: ComplexArray) -> ComplexArray:
        return sp.fft.ifftshift(sp.fft.ifft(sp.fft.fftshift(field)))

    def taylor_sum(
        self, coefficients: Sequence[float], variable: FloatArray
    ) -> FloatArray:
        output = np.zeros_like(variable, dtype=float)
        for index, coefficient in enumerate(coefficients):
            output += coefficient / factorial(index) * variable**index
        return output

    def find_nearest(self, array: FloatArray, value: float) -> int:
        return int(np.abs(array - value).argmin())

    def make_simple_beta(
        self, coefficients: Sequence[float], expansion_point: float
    ) -> FloatArray:
        frequency = self.omega + self.omega_p
        return self.taylor_sum(coefficients, frequency - expansion_point)

    def transform_beta(
        self,
        beta: FloatArray,
        beta1: float,
        frequency: FloatArray,
        expansion_point: float,
    ) -> FloatArray:
        return beta - beta1 * (frequency - expansion_point)

    def normalize_input(self, field: ComplexArray) -> ComplexArray:
        norm = np.sum(field) * self.domega
        return field / norm

    def make_gaussian_input(self, t0: float, t_off: float = 0.0) -> ComplexArray:
        field = np.zeros_like(self.t, dtype=complex)
        field += (
            1
            / (t0 * np.sqrt(2 * np.pi))
            * np.exp(-4 * np.log(2) * ((self.t + t_off) / t0) ** 2)
        )
        return self.normalize_input(self.fft(field))

    def make_sech_input(self, p0: float, t_off: float, t0: float) -> ComplexArray:
        field = np.zeros_like(self.t, dtype=complex)

        def inv_acosh(x_axis: FloatArray) -> FloatArray:
            result = [
                1 / np.cosh(value) if np.abs(value) < 710.4 else 0 for value in x_axis
            ]
            return np.array(result, dtype=float)

        argument = (self.t + t_off) / t0
        field += (
            np.sqrt(p0)
            * inv_acosh(argument)
            * np.exp(-1j * (self.t + t_off) ** 2 / (2 * t0**2))
        )
        return self.normalize_input(self.fft(field))

    def make_cw_input(self, p0: float = 1.0) -> ComplexArray:
        field = np.zeros_like(self.t, dtype=complex)
        df = (self.omega[1] - self.omega[0]) / (2 * np.pi)
        center_index = int(self.N / 2)
        field[center_index] = np.sqrt(p0 / (2 * np.pi)) / df
        return self.normalize_input(field)

    def make_hermite_gaussian_basis_functions(
        self,
        t_off: float,
        t0: float,
        order: int,
        fft_bool: bool = True,
    ) -> ComplexArray:
        field = np.zeros_like(self.t, dtype=complex)
        field += self.hermite_gaussian_function(order, self.t + t_off, t0)
        norm = np.sum(np.abs(field) ** 2) * self.dt
        field = field / np.sqrt(norm)
        if fft_bool:
            field = self.fft(field)
        return field

    def hermite_gaussian_function(
        self, order: int, t_axis: FloatArray, width: float
    ) -> ComplexArray:
        return (
            1
            / np.exp(order)
            * np.exp(-(t_axis**2) / (2 * width**2))
            * special.eval_hermite(order, t_axis / width)
        )

    def add_noise(self, field: ComplexArray) -> ComplexArray:
        df = (self.omega[1] - self.omega[0]) / (2 * np.pi)
        random_values = np.random.random(field.size)
        field += np.exp(1j * random_values * 2 * np.pi) * np.sqrt(
            self.hbar * self.omega_real / df
        )
        return field

    def get_pump(self, z: float) -> ComplexArray:
        return self.Ap_0 * np.exp(1j * self.betap * z)

    def qpm(self, z: float) -> float:
        if self.QPMPeriod == 0:
            return 1.0
        return float(np.sign(np.sin(2 * np.pi / self.QPMPeriod * z)))

    def set_initial_conditions(self, input_fields: ComplexArray) -> None:
        self.initial_conditions_flag = True
        self.As_0, self.Ai_0, self.Ap_0 = input_fields
        initial_values = np.hstack(
            (
                np.real(self.As_0),
                np.imag(self.As_0),
                np.real(self.Ai_0),
                np.imag(self.Ai_0),
            )
        )
        self.solver.set_initial_value(initial_values, 0)

    def save_variables(
        self,
        step_index: int,
        field: NDArray[np.float64],
    ) -> tuple[float, ComplexArray, ComplexArray]:
        z = step_index * self.dz

        as_field = (field[0] + 1j * field[1]) * np.exp(1j * self.betas * z)
        ai_field = (field[2] + 1j * field[3]) * np.exp(1j * self.betai * z)

        field_time = np.zeros((self.N, 2), dtype=complex)
        field_spec = np.zeros_like(field_time)

        field_time[:, 0] = self.ifft(as_field)
        field_time[:, 1] = self.ifft(ai_field)
        field_spec[:, 0] = as_field
        field_spec[:, 1] = ai_field
        return z, field_spec, field_time

    def rk4_step(
        self, z: float, field_interaction: NDArray[np.float64], dz: float
    ) -> NDArray[np.float64]:
        k1 = self.ode_nl(z, field_interaction)
        k2 = self.ode_nl(z + 0.5 * dz, field_interaction + 0.5 * dz * k1)
        k3 = self.ode_nl(z + 0.5 * dz, field_interaction + 0.5 * dz * k2)
        k4 = self.ode_nl(z + dz, field_interaction + dz * k3)
        return field_interaction + dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def _prepare_run(self) -> tuple[int, NDArray[np.float64], float]:
        if not self.initial_conditions_flag:
            raise RuntimeError("No initial conditions given.")

        nz_float = self.L / self.dz
        n_save = int(np.round(nz_float))
        self.dz = self.L / n_save
        state = np.hstack(
            (
                np.real(self.As_0),
                np.imag(self.As_0),
                np.real(self.Ai_0),
                np.imag(self.Ai_0),
            )
        )
        return n_save, state, 0.0

    def _state_to_complex(
        self, state: NDArray[np.float64]
    ) -> tuple[ComplexArray, ComplexArray]:
        fields = np.reshape(state, (4, self.N))
        return fields[0] + 1j * fields[1], fields[2] + 1j * fields[3]

    def _complex_to_state(
        self, as_interaction: ComplexArray, ai_interaction: ComplexArray
    ) -> NDArray[np.float64]:
        return np.hstack(
            (
                np.real(as_interaction),
                np.imag(as_interaction),
                np.real(ai_interaction),
                np.imag(ai_interaction),
            )
        )

    def _prepare_rk4_stage_cache(
        self, n_save: int
    ) -> list[
        tuple[
            tuple[
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                float,
            ],
            tuple[
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                float,
            ],
            tuple[
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                float,
            ],
            tuple[
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                float,
            ],
        ]
    ]:
        z_values = np.arange(n_save, dtype=float) * self.dz
        half_step = z_values + 0.5 * self.dz
        full_step = z_values + self.dz

        signal_phase_scale = 1j * (self.betas + 1j * self.alpha_s)
        idler_phase_scale = 1j * (self.betai + 1j * self.alpha_i)

        def stage_data(
            z_axis: FloatArray,
        ) -> tuple[
            ComplexArray,
            ComplexArray,
            ComplexArray,
            ComplexArray,
            ComplexArray,
            FloatArray,
        ]:
            signal_forward = np.exp(np.outer(z_axis, signal_phase_scale))
            idler_forward = np.exp(np.outer(z_axis, idler_phase_scale))
            signal_backward = np.exp(np.outer(z_axis, -signal_phase_scale))
            idler_backward = np.exp(np.outer(z_axis, -idler_phase_scale))
            pump_time = np.array(
                [self.ifft(self.Ap_0 * np.exp(1j * self.betap * z)) for z in z_axis]
            )
            qpm_values = np.array([self.qpm(float(z)) for z in z_axis], dtype=float)
            return (
                signal_forward,
                idler_forward,
                signal_backward,
                idler_backward,
                pump_time,
                qpm_values,
            )

        start_stage = stage_data(z_values)
        half_stage = stage_data(half_step)
        end_stage = stage_data(full_step)
        return [
            (
                (
                    start_stage[0][index],
                    start_stage[1][index],
                    start_stage[2][index],
                    start_stage[3][index],
                    start_stage[4][index],
                    float(start_stage[5][index]),
                ),
                (
                    half_stage[0][index],
                    half_stage[1][index],
                    half_stage[2][index],
                    half_stage[3][index],
                    half_stage[4][index],
                    float(half_stage[5][index]),
                ),
                (
                    half_stage[0][index],
                    half_stage[1][index],
                    half_stage[2][index],
                    half_stage[3][index],
                    half_stage[4][index],
                    float(half_stage[5][index]),
                ),
                (
                    end_stage[0][index],
                    end_stage[1][index],
                    end_stage[2][index],
                    end_stage[3][index],
                    end_stage[4][index],
                    float(end_stage[5][index]),
                ),
            )
            for index in range(n_save)
        ]

    def _ode_nl_complex(
        self,
        as_interaction: ComplexArray,
        ai_interaction: ComplexArray,
        signal_forward: ComplexArray,
        idler_forward: ComplexArray,
        signal_backward: ComplexArray,
        idler_backward: ComplexArray,
        pump_time: ComplexArray,
        qpm: float,
    ) -> tuple[ComplexArray, ComplexArray]:
        as_field = self.ifft(as_interaction * signal_forward)
        ai_field = self.ifft(ai_interaction * idler_forward)

        nas = 1j * self.gamma * np.conjugate(ai_field) * pump_time * qpm
        nai = 1j * self.gamma * np.conjugate(as_field) * pump_time * qpm

        das = signal_backward * self.fft(nas)
        dai = idler_backward * self.fft(nai)
        return das, dai

    def _rk4_step_complex(
        self,
        as_interaction: ComplexArray,
        ai_interaction: ComplexArray,
        stage_values: tuple[
            tuple[
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                float,
            ],
            tuple[
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                float,
            ],
            tuple[
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                float,
            ],
            tuple[
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                ComplexArray,
                float,
            ],
        ],
        dz: float,
    ) -> tuple[ComplexArray, ComplexArray]:
        k1_as, k1_ai = self._ode_nl_complex(
            as_interaction, ai_interaction, *stage_values[0]
        )
        k2_as, k2_ai = self._ode_nl_complex(
            as_interaction + 0.5 * dz * k1_as,
            ai_interaction + 0.5 * dz * k1_ai,
            *stage_values[1],
        )
        k3_as, k3_ai = self._ode_nl_complex(
            as_interaction + 0.5 * dz * k2_as,
            ai_interaction + 0.5 * dz * k2_ai,
            *stage_values[2],
        )
        k4_as, k4_ai = self._ode_nl_complex(
            as_interaction + dz * k3_as,
            ai_interaction + dz * k3_ai,
            *stage_values[3],
        )
        return (
            as_interaction + dz * (k1_as + 2 * k2_as + 2 * k3_as + k4_as) / 6.0,
            ai_interaction + dz * (k1_ai + 2 * k2_ai + 2 * k3_ai + k4_ai) / 6.0,
        )

    def ode_nl(
        self, z: float, field_interaction: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        field_interaction = np.reshape(field_interaction, (4, self.N))
        as_real, as_imag, ai_real, ai_imag = field_interaction
        as_interaction = as_real + 1j * as_imag
        ai_interaction = ai_real + 1j * ai_imag

        signal_exponential_factor = 1j * (self.betas + 1j * self.alpha_s) * z
        idler_exponential_factor = 1j * (self.betai + 1j * self.alpha_i) * z

        as_field = self.ifft(as_interaction * np.exp(signal_exponential_factor))
        ai_field = self.ifft(ai_interaction * np.exp(idler_exponential_factor))
        ap_field = self.ifft(self.get_pump(z))
        qpm = self.qpm(z)

        nas = 1j * self.gamma * np.conjugate(ai_field) * ap_field * qpm
        nai = 1j * self.gamma * np.conjugate(as_field) * ap_field * qpm

        das = np.exp(-signal_exponential_factor) * self.fft(nas)
        dai = np.exp(-idler_exponential_factor) * self.fft(nai)
        return np.hstack((np.real(das), np.imag(das), np.real(dai), np.imag(dai)))

    def run(
        self,
    ) -> tuple[FloatArray, FloatArray, float, FloatArray, ComplexArray, ComplexArray]:
        n_save, state, z_current = self._prepare_run()

        z_out = np.zeros(n_save + 1, dtype=float)
        field_spec = np.zeros((n_save + 1, self.N, 2), dtype=complex)
        field_time = np.zeros_like(field_spec)

        fields_0 = np.array(
            [
                np.real(self.As_0),
                np.imag(self.As_0),
                np.real(self.Ai_0),
                np.imag(self.Ai_0),
            ]
        )
        z_out[0], field_spec[0, :, :], field_time[0, :, :] = self.save_variables(
            0, fields_0
        )
        start_time = time.time()

        time_left = 0.0
        if self.integration_method == "rk4":
            as_interaction, ai_interaction = self._state_to_complex(state)
            rk4_stage_cache = self._prepare_rk4_stage_cache(n_save)
        for step_index in range(1, n_save + 1):
            step_start = time.time()
            if self.print_bool:
                print("", end="\r")
                print(
                    f"Step {step_index} of {n_save + 1}, approximate time left [s]: {np.round(time_left, 2)}",
                    end="",
                )

            if self.integration_method == "adaptive":
                self.solver.integrate(z_out[step_index - 1] + self.dz)
                state = self.solver.y
            else:
                as_interaction, ai_interaction = self._rk4_step_complex(
                    as_interaction,
                    ai_interaction,
                    rk4_stage_cache[step_index - 1],
                    self.dz,
                )
                state = self._complex_to_state(as_interaction, ai_interaction)
            z_current += self.dz

            fields = np.reshape(state, (4, self.N))
            (
                z_out[step_index],
                field_spec[step_index, :, :],
                field_time[step_index, :, :],
            ) = self.save_variables(
                step_index,
                fields,
            )
            elapsed = time.time() - step_start
            time_left = (n_save - step_index) * elapsed

        total_time = time.time() - start_time
        if self.print_bool:
            print(f"\nSimulation time [s]: {np.round(total_time, 2)}")

        return z_out, self.omega, self.omega_p, self.t, field_spec, field_time

    def run_final_only(self) -> tuple[ComplexArray, ComplexArray]:
        n_save, state, z_current = self._prepare_run()

        initial_state = np.array(
            [
                np.real(self.As_0),
                np.imag(self.As_0),
                np.real(self.Ai_0),
                np.imag(self.Ai_0),
            ]
        )
        _, _, input_field_time = self.save_variables(0, initial_state)

        if self.integration_method == "rk4":
            as_interaction, ai_interaction = self._state_to_complex(state)
            rk4_stage_cache = self._prepare_rk4_stage_cache(n_save)

        for _ in range(1, n_save + 1):
            if self.integration_method == "adaptive":
                self.solver.integrate(z_current + self.dz)
                state = self.solver.y
            else:
                step_index = int(np.round(z_current / self.dz))
                as_interaction, ai_interaction = self._rk4_step_complex(
                    as_interaction,
                    ai_interaction,
                    rk4_stage_cache[step_index],
                    self.dz,
                )
                state = self._complex_to_state(as_interaction, ai_interaction)
            z_current += self.dz

        fields = np.reshape(state, (4, self.N))
        _, _, output_field_time = self.save_variables(n_save, fields)
        return input_field_time, output_field_time
