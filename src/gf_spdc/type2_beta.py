from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class Type2:
    """Wave-vector model for type-II phase matching in 5% MgO:LiNbO3."""

    def __init__(self, lambda_s: float, lambda_i: float, lambda_p: float) -> None:
        # Speed of light in m/ps.
        c = 299792458e-12

        wavelength_array = np.array([lambda_p, lambda_s, lambda_i], dtype=float) * 1e6
        minimum_wavelength = float(np.min(wavelength_array))
        maximum_wavelength = float(np.max(wavelength_array))

        wavelength_grid = np.linspace(maximum_wavelength, minimum_wavelength, 5000)
        omega = 2 * np.pi * c / wavelength_grid * 1e6

        self.kp = self.k_type2(omega, False) * 1e6
        self.ks = self.k_type2(omega, True) * 1e6
        self.ki = self.k_type2(omega, False) * 1e6
        self.om = omega
        self.indistinguishableBool = False
        self.QPMbool = False

    def sellmeier(
        self,
        a: float,
        b1: float,
        b2: float,
        c1: float,
        c2: float,
        wavelength: ArrayLike,
    ) -> FloatArray:
        wavelength_array = np.asarray(wavelength, dtype=float)
        return np.sqrt(a + b1 / (wavelength_array**2 - b2) + c1 / (wavelength_array**2 - c2))

    def omega_func(self, nx: FloatArray, ny: FloatArray, nz: FloatArray) -> FloatArray:
        return np.arcsin(nz / ny * np.sqrt((ny**2 - nx**2) / (nz**2 - nx**2)))

    def theta_mm_func(self, theta: float, phi: float) -> float:
        return float(np.arctan(np.cos(phi) * np.tan(theta)))

    def theta1_func(self, theta: float, phi: float, theta_mm: float, omega: FloatArray) -> FloatArray:
        if theta != np.pi / 2:
            return np.arccos((np.cos(theta) / np.cos(theta_mm)) * np.cos(omega - theta_mm))
        return np.arccos(np.sin(omega) * np.cos(phi))

    def theta2_func(self, theta: float, phi: float, theta_mm: float, omega: FloatArray) -> FloatArray:
        if theta != np.pi / 2:
            return np.arccos((np.cos(theta) / np.cos(theta_mm)) * np.cos(omega + theta_mm))
        return np.arccos(-np.sin(omega) * np.cos(phi))

    def effective_n(self, nx: FloatArray, nz: FloatArray, theta: FloatArray) -> FloatArray:
        return nx * nz / np.sqrt(nz**2 * np.cos(theta / 2) ** 2 + nx**2 * np.sin(theta / 2) ** 2)

    def k_type2(self, omega: FloatArray, phase_matched: bool) -> FloatArray:
        # Speed of light in microns/ps.
        c = 299.792458
        wavelength = 2 * np.pi * c / omega

        ax = 3.29100
        b1x = 0.04140
        b2x = 0.03978
        c1x = 9.35522
        c2x = 31.45571

        ay = 3.45018
        b1y = 0.04341
        b2y = 0.04597
        c1y = 16.98825
        c2y = 39.43799

        az = 4.59423
        b1z = 0.06206
        b2z = 0.04763
        c1z = 110.80672
        c2z = 86.12171

        nx = self.sellmeier(ax, b1x, b2x, c1x, c2x, wavelength)
        ny = self.sellmeier(ay, b1y, b2y, c1y, c2y, wavelength)
        nz = self.sellmeier(az, b1z, b2z, c1z, c2z, wavelength)

        omega_value = self.omega_func(nx, ny, nz)
        theta = 90 * np.pi / 180
        phi = 23.58 * np.pi / 180
        theta_mm = self.theta_mm_func(theta, phi)

        theta1 = self.theta1_func(theta, phi, theta_mm, omega_value)
        theta2 = self.theta2_func(theta, phi, theta_mm, omega_value)
        total_theta = theta1 + theta2 if phase_matched else theta1 - theta2

        return self.effective_n(nx, nz, total_theta) * 2 * np.pi / wavelength
