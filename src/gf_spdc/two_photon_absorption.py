from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[Any]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IndistinguishableTPAInputs:
    """Inputs for indistinguishable-photon two-photon absorption overlaps.

    `g` is the cross Green's function, with `g_is = g_si = g`.
    `f` is the self Green's function, with `g_ii = g_ss = f`.
    `omega` must be real, ascending, and uniformly spaced by `domega`.
    """

    g: ComplexArray
    f: ComplexArray
    omega: FloatArray
    domega: float
    gamma: float
    omega_fg: float = 0.0


@dataclass(frozen=True)
class TPAContributionBreakdown:
    """Three-contribution breakdown for indistinguishable-photon 2PA."""

    coherent: float
    incoherent_type1: float
    incoherent_type2: float
    incoherent_total: float
    total: float
    g2_numerator_single_cross: complex
    g2_numerator_single_same: complex
    g2_numerator: complex
    g2_denominator_single: complex
    g2_denominator: complex
    g2: complex


def _validate_inputs(g: ComplexArray, f: ComplexArray, omega: FloatArray) -> None:
    if not np.isrealobj(omega):
        raise ValueError("omega must be a real-valued axis.")
    if omega.ndim != 1:
        raise ValueError("omega must be a one-dimensional axis.")
    if g.ndim != 2 or f.ndim != 2:
        raise ValueError("Green's functions must be rank-2 arrays.")
    if g.shape != f.shape:
        raise ValueError("g and f must have identical shapes.")
    if g.shape[0] != g.shape[1]:
        raise ValueError("Green's functions must be square matrices.")
    if g.shape[0] != len(omega):
        raise ValueError("Green's function dimensions must match the omega axis length.")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(omega)):
        raise ValueError("Green's functions and omega must contain only finite values.")


def _pair_matrix(left: ComplexArray, right: ComplexArray, domega: float, flip_lr: bool) -> ComplexArray:
    matrix = np.asarray((left @ np.transpose(right)) * domega, dtype=complex)
    return np.fliplr(matrix) if flip_lr else matrix


def _coherent_overlap_contribution(
    f1g1: ComplexArray,
    f1g2: ComplexArray,
    f2g1: ComplexArray,
    f2g2: ComplexArray,
    omega: FloatArray,
    domega: float,
    gamma: float,
    omega_fg: float,
) -> float:
    alpha = gamma - 1j * omega_fg
    f1 = _pair_matrix(f1g2, f1g1, domega, flip_lr=True)
    f2 = _pair_matrix(f2g2, f2g1, domega, flip_lr=True)

    omega_s = np.arange(-len(omega), len(omega), dtype=float) * domega
    h = np.zeros(2 * len(omega), dtype=complex)

    for ns in range(len(omega)):
        denominator = alpha + 1j * omega_s[ns]
        for index in range(ns):
            h[ns] += f1[ns - index, index] / denominator

    for ns in range(len(omega)):
        denominator = alpha + 1j * omega_s[ns + len(omega)]
        for index in range(ns, len(omega)):
            h[ns + len(omega)] += f1[ns - index, index] / denominator

    h *= domega
    output = 0j
    for omega_index in range(len(omega)):
        for omega_prime_index in range(len(omega)):
            output += h[omega_index + omega_prime_index] * f2[omega_index, omega_prime_index]
    output *= domega**2
    return float(np.real(output))


def _incoherent_overlap_contribution_type1(
    f1g1: ComplexArray,
    f1g2: ComplexArray,
    f2g1: ComplexArray,
    f2g2: ComplexArray,
    omega: FloatArray,
    domega: float,
    gamma: float,
    omega_fg: float,
) -> float:
    alpha = gamma - 1j * omega_fg
    f1 = _pair_matrix(f1g2, f1g1, domega, flip_lr=True)
    f2 = _pair_matrix(f2g2, f2g1, domega, flip_lr=True)

    omega_s = np.arange(-len(omega), len(omega), dtype=float) * domega
    h = np.zeros(2 * len(omega), dtype=complex)

    for ns in reversed(range(len(omega))):
        for index in range(len(omega) - ns, len(omega)):
            h[2 * len(omega) - ns - 1] += f1[index + ns - len(omega), index]

    for ns in range(len(omega)):
        for index in range(len(omega) - ns):
            h[len(omega) - ns] += f1[ns + index, index]

    h *= domega
    output = 0j
    for omega_index in range(len(omega)):
        for omega_prime_index in range(len(omega)):
            output += h[omega_index - omega_prime_index + len(omega)] * f2[omega_prime_index, omega_index] / (
                alpha + 1j * omega_s[omega_index + omega_prime_index]
            )
    output *= domega**2
    return float(np.real(output))


def _incoherent_overlap_contribution_type2(
    f1g1: ComplexArray,
    f1g2: ComplexArray,
    f2g1: ComplexArray,
    f2g2: ComplexArray,
    omega: FloatArray,
    domega: float,
    gamma: float,
    omega_fg: float,
) -> float:
    alpha = gamma - 1j * omega_fg
    f1 = _pair_matrix(f1g2, f1g1, domega, flip_lr=True)
    f2 = _pair_matrix(f2g2, f2g1, domega, flip_lr=True)

    h1 = np.zeros(2 * len(omega), dtype=complex)
    h2 = np.zeros(2 * len(omega), dtype=complex)

    for ns in range(len(omega)):
        for index in range(len(omega) - ns):
            h1[len(omega) - ns] += f1[ns + index, index] / (alpha / 2 + 1j * omega[index])

    for ns in reversed(range(len(omega))):
        for index in range(len(omega) - ns, len(omega)):
            h1[2 * len(omega) - ns - 1] += f1[index + ns - len(omega), index] / (alpha / 2 + 1j * omega[index])

    h1 *= domega
    inv_lambda_1 = 0j
    for omega_tilde_index in range(len(omega)):
        for omega_prime_index in range(len(omega)):
            inv_lambda_1 += h1[omega_tilde_index - omega_prime_index + len(omega)] * f2[omega_tilde_index, omega_prime_index]
    inv_lambda_1 *= domega**2

    for ns in range(len(omega)):
        for index in range(len(omega) - ns):
            h2[len(omega) - ns] += f1[ns + index, index]

    for ns in reversed(range(len(omega))):
        for index in range(len(omega) - ns, len(omega)):
            h2[2 * len(omega) - ns - 1] += f1[index + ns - len(omega), index]

    h2 *= domega
    inv_lambda_2 = 0j
    for omega_tilde_index in range(len(omega)):
        for omega_prime_index in range(len(omega)):
            inv_lambda_2 += h2[omega_tilde_index - omega_prime_index + len(omega)] * f2[omega_tilde_index, omega_prime_index] / (
                alpha / 2 + 1j * omega[omega_prime_index]
            )
    inv_lambda_2 *= domega**2

    if np.isclose(inv_lambda_1, 0.0) or np.isclose(inv_lambda_2, 0.0):
        raise ValueError("Incoherent type-2 contribution is singular for the supplied inputs.")

    lambda_total = 1 / inv_lambda_1 + 1 / inv_lambda_2
    output = 1 / lambda_total
    return float(np.real(output))


def _g2_numerator(
    f1g1: ComplexArray,
    f1g2: ComplexArray,
    f2g1: ComplexArray,
    f2g2: ComplexArray,
    domega: float,
) -> complex:
    f1 = _pair_matrix(f1g2, f1g1, domega, flip_lr=False)
    f2 = _pair_matrix(f2g2, f2g1, domega, flip_lr=False)
    return complex(np.sum(f1) * domega**2 * np.sum(f2) * domega**2)


def _g2_denominator(gjj: ComplexArray, domega: float) -> complex:
    f_matrix = np.asarray((np.conj(gjj) @ np.transpose(gjj)) * domega, dtype=complex)
    return complex(np.sum(f_matrix) * domega**2)


def calculate_indistinguishable_tpa_overlap(inputs: IndistinguishableTPAInputs) -> TPAContributionBreakdown:
    """Calculate the requested three 2PA contributions for indistinguishable photons."""

    _validate_inputs(inputs.g, inputs.f, inputs.omega)
    if not np.isfinite(inputs.domega) or inputs.domega <= 0:
        raise ValueError("domega must be finite and positive.")
    if not np.isfinite(inputs.gamma) or inputs.gamma <= 0:
        raise ValueError("gamma must be finite and positive.")
    if not np.isfinite(inputs.omega_fg):
        raise ValueError("omega_fg must be finite.")
    if len(inputs.omega) > 1 and not np.allclose(np.diff(inputs.omega), inputs.domega):
        raise ValueError("omega must be uniformly spaced with step domega.")

    g_conjugate = np.conj(inputs.g)
    f_conjugate = np.conj(inputs.f)

    coherent = _coherent_overlap_contribution(
        g_conjugate,
        f_conjugate,
        inputs.g,
        inputs.f,
        inputs.omega,
        inputs.domega,
        inputs.gamma,
        inputs.omega_fg,
    )

    incoherent_type1 = _incoherent_overlap_contribution_type1(
        g_conjugate,
        inputs.g,
        inputs.g,
        g_conjugate,
        inputs.omega,
        inputs.domega,
        inputs.gamma,
        inputs.omega_fg,
    )

    incoherent_type2 = _incoherent_overlap_contribution_type2(
        g_conjugate,
        inputs.g,
        inputs.g,
        g_conjugate,
        inputs.omega,
        inputs.domega,
        inputs.gamma,
        inputs.omega_fg,
    )
    incoherent_total = incoherent_type1 + incoherent_type2

    g2_numerator_single_cross = _g2_numerator(
        g_conjugate,
        f_conjugate,
        inputs.g,
        inputs.f,
        inputs.domega,
    )
    g2_numerator_single_same = _g2_numerator(
        g_conjugate,
        inputs.g,
        inputs.g,
        g_conjugate,
        inputs.domega,
    )
    g2_numerator = g2_numerator_single_cross + g2_numerator_single_same + g2_numerator_single_same

    g2_denominator_single = _g2_denominator(inputs.g, inputs.domega)
    g2_denominator = g2_denominator_single**2
    if np.isclose(g2_denominator, 0.0):
        raise ValueError("g2 denominator is zero or numerically singular for the supplied Green's function.")
    g2_value = g2_numerator / g2_denominator

    return TPAContributionBreakdown(
        coherent=coherent,
        incoherent_type1=incoherent_type1,
        incoherent_type2=incoherent_type2,
        incoherent_total=incoherent_total,
        total=coherent + incoherent_total,
        g2_numerator_single_cross=g2_numerator_single_cross,
        g2_numerator_single_same=g2_numerator_single_same,
        g2_numerator=g2_numerator,
        g2_denominator_single=g2_denominator_single,
        g2_denominator=g2_denominator,
        g2=g2_value,
    )


__all__ = [
    "IndistinguishableTPAInputs",
    "TPAContributionBreakdown",
    "calculate_indistinguishable_tpa_overlap",
]
