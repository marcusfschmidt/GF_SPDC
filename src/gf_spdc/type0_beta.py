from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class Type0:
    """Wave-vector model for type-0 phase matching in 5% MgO:LiNbO3."""

    def __init__(
        self,
        lambda_s: float,
        lambda_i: float,
        lambda_p: float,
        ordinary_axis_bool: bool,
        temperature: float,
        qpm_period: float,
    ) -> None:
        # Speed of light in m/ps.
        c = 299792458e-12

        wavelength_array = np.array([lambda_p, lambda_s, lambda_i], dtype=float) * 1e6
        minimum_wavelength = float(np.min(wavelength_array))
        maximum_wavelength = float(np.max(wavelength_array))

        wavelength_grid = np.linspace(maximum_wavelength, minimum_wavelength, 5000)
        self.om = 2 * np.pi * c / wavelength_grid * 1e6

        f_term = (temperature - 24.5) * (temperature + 570.82)

        if ordinary_axis_bool:
            refractive_index = self.ordinary_axis(wavelength_grid, f_term)
        else:
            refractive_index = self.extraordinary_axis(wavelength_grid, f_term)

        beta = 2 * np.pi / wavelength_grid * refractive_index * 1e6

        self.kp = beta
        self.ki = beta
        self.ks = beta
        self.QPMPeriod = qpm_period
        self.indistinguishableBool = True
        self.QPMbool = True

    def ordinary_axis(self, wavelength: FloatArray, f_term: float) -> FloatArray:
        a1 = 5.653
        a2 = 0.1185
        a3 = 0.2091
        a4 = 89.61
        a5 = 10.85
        a6 = 1.97e-2
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.118e-6
        no2 = (
            a1
            + b1 * f_term
            + (a2 + b2 * f_term) / (wavelength**2 - (a3 + b3 * f_term) ** 2)
            + (a4 + b4 * f_term) / (wavelength**2 - a5**2)
            - a6 * wavelength**2
        )
        return np.sqrt(no2)

    def extraordinary_axis(self, wavelength: FloatArray, f_term: float) -> FloatArray:
        a1 = 5.756
        a2 = 0.0983
        a3 = 0.2020
        a4 = 189.32
        a5 = 12.52
        a6 = 1.32e-2
        b1 = 2.860e-6
        b2 = 4.7e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
        ne2 = (
            a1
            + b1 * f_term
            + (a2 + b2 * f_term) / (wavelength**2 - (a3 + b3 * f_term) ** 2)
            + (a4 + b4 * f_term) / (wavelength**2 - a5**2)
            - a6 * wavelength**2
        )
        return np.sqrt(ne2)
