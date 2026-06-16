from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import warnings

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[Any]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


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


@dataclass(frozen=True)
class DistinguishableTPAInputs:
    """Inputs for distinguishable-photon two-photon absorption overlaps."""

    g_is: ComplexArray
    g_ii: ComplexArray
    g_si: ComplexArray
    g_ss: ComplexArray
    omega: FloatArray
    domega: float
    gamma: float
    omega_fg: float = 0.0


@dataclass(frozen=True)
class TPAHFunction:
    """Weighted h-vectors saved from the 2PA overlap contributions."""

    coherent: ComplexArray
    incoherent_type1: ComplexArray
    incoherent_type2: ComplexArray
    incoherent_total: ComplexArray


@dataclass(frozen=True)
class _NormalizedTPAInputs:
    g: ComplexArray
    f: ComplexArray
    g_scale: float
    f_scale: float
    omega: FloatArray
    domega: float
    gamma: float
    omega_fg: float


@dataclass(frozen=True)
class _OverlapIndexCache:
    diagonal_destinations: IntArray
    anti_diagonal_indices: IntArray
    forward_difference_indices: IntArray
    backward_difference_indices: IntArray


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
        raise ValueError(
            "Green's function dimensions must match the omega axis length."
        )
    if (
        not np.all(np.isfinite(g))
        or not np.all(np.isfinite(f))
        or not np.all(np.isfinite(omega))
    ):
        raise ValueError("Green's functions and omega must contain only finite values.")


def _normalize_green_function(field: ComplexArray) -> tuple[ComplexArray, float]:
    field = np.asarray(field, dtype=complex)
    finite_mask = np.isfinite(field)
    if not np.any(finite_mask):
        raise ValueError("Green's functions and omega must contain only finite values.")

    scale = float(np.max(np.abs(field[finite_mask])))
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("Green's functions and omega must contain only finite values.")

    return np.asarray(field / scale, dtype=complex), scale


def _normalize_tpa_inputs(
    g: ComplexArray,
    f: ComplexArray,
    omega: FloatArray,
    domega: float,
    gamma: float,
    omega_fg: float,
) -> _NormalizedTPAInputs:
    g_norm, g_scale = _normalize_green_function(g)
    f_norm, f_scale = _normalize_green_function(f)
    return _NormalizedTPAInputs(
        g=g_norm,
        f=f_norm,
        g_scale=g_scale,
        f_scale=f_scale,
        omega=np.asarray(omega, dtype=float),
        domega=domega,
        gamma=gamma,
        omega_fg=omega_fg,
    )


def _rescale_breakdown(
    breakdown: TPAContributionBreakdown,
    g_scale: float,
    f_scale: float,
) -> TPAContributionBreakdown:
    scale_coherent = f_scale * g_scale * f_scale * g_scale
    scale_type1 = f_scale * g_scale * g_scale * g_scale
    scale_type2 = f_scale * g_scale * g_scale * g_scale
    scale_total = scale_coherent + scale_type1 + scale_type2

    return TPAContributionBreakdown(
        coherent=breakdown.coherent * scale_coherent,
        incoherent_type1=breakdown.incoherent_type1 * scale_type1,
        incoherent_type2=breakdown.incoherent_type2 * scale_type2,
        incoherent_total=breakdown.incoherent_total * scale_type1 + breakdown.incoherent_type2 * scale_type2,
        total=breakdown.total * scale_total,
        g2_numerator_single_cross=breakdown.g2_numerator_single_cross * scale_coherent,
        g2_numerator_single_same=breakdown.g2_numerator_single_same * scale_type1,
        g2_numerator=breakdown.g2_numerator * scale_total,
        g2_denominator_single=breakdown.g2_denominator_single * (g_scale**4),
        g2_denominator=breakdown.g2_denominator * (g_scale**8),
        g2=breakdown.g2,
    )


def _pair_matrix(
    left: ComplexArray, right: ComplexArray, domega: float, flip_lr: bool
) -> ComplexArray:
    matrix = np.asarray((left @ np.transpose(right)) * domega, dtype=complex)
    return np.fliplr(matrix) if flip_lr else matrix


def _complex_bincount(
    indices: IntArray, values: ComplexArray, length: int
) -> ComplexArray:
    values_array = np.asarray(values, dtype=complex).ravel()
    return np.bincount(
        indices,
        weights=np.asarray(np.real(values_array), dtype=float),
        minlength=length,
    ) + 1j * np.bincount(
        indices,
        weights=np.asarray(np.imag(values_array), dtype=float),
        minlength=length,
    )


@lru_cache(maxsize=None)
def _overlap_indices(size: int) -> _OverlapIndexCache:
    rows_raw, cols_raw = np.indices((size, size), dtype=int)
    rows = np.asarray(rows_raw, dtype=int)
    cols = np.asarray(cols_raw, dtype=int)
    diagonal_destinations = np.asarray(
        np.where(cols > rows, size + cols - rows - 1, size + cols - rows).ravel(),
        dtype=int,
    )
    anti_diagonal_indices = np.asarray(rows + cols, dtype=int)
    forward_difference_indices = np.asarray(cols - rows + size, dtype=int)
    backward_difference_indices = np.asarray(rows - cols + size, dtype=int)
    return _OverlapIndexCache(
        diagonal_destinations,
        anti_diagonal_indices,
        forward_difference_indices,
        backward_difference_indices,
    )


@dataclass(frozen=True)
class _PreparedOverlapTerms:
    alpha: complex
    half_alpha_denominator: ComplexArray
    fg_pair: ComplexArray
    fg_pair_conjugate: ComplexArray
    gg_pair: ComplexArray
    gg_pair_conjugate: ComplexArray
    omega_s: FloatArray


def _prepare_overlap_terms(
    g: ComplexArray,
    f: ComplexArray,
    omega: FloatArray,
    domega: float,
    gamma: float,
    omega_fg: float,
) -> _PreparedOverlapTerms:
    alpha = gamma - 1j * omega_fg
    size = g.shape[0]
    omega_s = np.arange(-size, size, dtype=float) * domega
    half_alpha_denominator = alpha / 2 + 1j * omega

    fg_pair = _pair_matrix(f, g, domega, flip_lr=False)
    fg_pair_conjugate = np.conj(fg_pair)
    gg_pair = _pair_matrix(g, np.conj(g), domega, flip_lr=True)
    gg_pair_conjugate = np.conj(gg_pair)

    return _PreparedOverlapTerms(
        alpha=alpha,
        half_alpha_denominator=np.asarray(half_alpha_denominator, dtype=complex),
        fg_pair=fg_pair,
        fg_pair_conjugate=fg_pair_conjugate,
        gg_pair=gg_pair,
        gg_pair_conjugate=gg_pair_conjugate,
        omega_s=np.asarray(omega_s, dtype=float),
    )


def _coherent_overlap_contribution_prepared(
    prepared: _PreparedOverlapTerms,
    domega: float,
) -> float:
    size = prepared.fg_pair.shape[0]
    h = np.zeros(2 * size, dtype=complex)

    for ns in range(size):
        for index in range(size - ns):
            h[ns + size] += prepared.fg_pair_conjugate[ns - index, index] / (
                prepared.alpha + 1j * prepared.omega_s[ns + size]
            )

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h[ns] += prepared.fg_pair_conjugate[ns - index, index] / (
                prepared.alpha + 1j * prepared.omega_s[ns]
            )

    h *= domega
    output = 0j
    for omega_index in range(size):
        for omega_prime_index in range(size):
            output += h[omega_index + omega_prime_index] * prepared.fg_pair[
                omega_index, omega_prime_index
            ]
    output *= domega**2
    return float(np.real(output))


def _coherent_h_function_prepared(
    prepared: _PreparedOverlapTerms,
    domega: float,
) -> ComplexArray:
    size = prepared.fg_pair.shape[0]
    h = np.zeros(2 * size, dtype=complex)

    for ns in range(size):
        for index in range(size - ns):
            h[ns + size] += prepared.fg_pair_conjugate[ns - index, index] / (
                prepared.alpha + 1j * prepared.omega_s[ns + size]
            )

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h[ns] += prepared.fg_pair_conjugate[ns - index, index] / (
                prepared.alpha + 1j * prepared.omega_s[ns]
            )

    return np.asarray(h * domega, dtype=complex)


def _incoherent_overlap_contribution_type1_prepared(
    prepared: _PreparedOverlapTerms,
    domega: float,
) -> float:
    size = prepared.gg_pair.shape[0]
    h = np.zeros(2 * size, dtype=complex)

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h[2 * size - ns - 1] += prepared.gg_pair[index + ns - size, index]

    for ns in range(size):
        for index in range(size - ns):
            h[size - ns] += prepared.gg_pair[ns + index, index]

    h *= domega
    output = 0j
    for omega_index in range(size):
        for omega_prime_index in range(size):
            output += h[omega_index - omega_prime_index + size] * prepared.gg_pair_conjugate[
                omega_prime_index, omega_index
            ] / (
                prepared.alpha
                + 1j * prepared.omega_s[omega_index + omega_prime_index]
            )
    output *= domega**2
    return float(np.real(output))


def _incoherent_type1_h_function_prepared(
    prepared: _PreparedOverlapTerms,
    domega: float,
) -> ComplexArray:
    size = prepared.gg_pair.shape[0]
    h = np.zeros(2 * size, dtype=complex)

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h[2 * size - ns - 1] += prepared.gg_pair[index + ns - size, index]

    for ns in range(size):
        for index in range(size - ns):
            h[size - ns] += prepared.gg_pair[ns + index, index]

    h *= domega
    return np.asarray(h, dtype=complex)


def _incoherent_overlap_contribution_type2_prepared(
    prepared: _PreparedOverlapTerms,
    domega: float,
) -> float:
    size = prepared.gg_pair.shape[0]
    h1 = np.zeros(2 * size, dtype=complex)
    h2 = np.zeros(2 * size, dtype=complex)

    for ns in range(size):
        for index in range(size - ns):
            h1[size - ns] += prepared.gg_pair[ns + index, index] / (
                prepared.half_alpha_denominator[index]
            )

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h1[2 * size - ns - 1] += prepared.gg_pair[
                index + ns - size, index
            ] / prepared.half_alpha_denominator[index]

    h1 *= domega
    inv_lambda_1 = 0j
    for omega_tilde_index in range(size):
        for omega_prime_index in range(size):
            inv_lambda_1 += h1[omega_tilde_index - omega_prime_index + size] * prepared.gg_pair_conjugate[
                omega_tilde_index, omega_prime_index
            ]
    inv_lambda_1 *= domega**2

    for ns in range(size):
        for index in range(size - ns):
            h2[size - ns] += prepared.gg_pair[ns + index, index]

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h2[2 * size - ns - 1] += prepared.gg_pair[index + ns - size, index]

    h2 *= domega
    inv_lambda_2 = 0j
    for omega_tilde_index in range(size):
        for omega_prime_index in range(size):
            inv_lambda_2 += h2[omega_tilde_index - omega_prime_index + size] * prepared.gg_pair_conjugate[
                omega_tilde_index, omega_prime_index
            ] / prepared.half_alpha_denominator[omega_prime_index]
    inv_lambda_2 *= domega**2

    lambda_total = 1 / inv_lambda_1 + 1 / inv_lambda_2
    output = 1 / lambda_total
    return float(np.real(output))


def _incoherent_type2_h_function_prepared(
    prepared: _PreparedOverlapTerms,
    domega: float,
) -> ComplexArray:
    size = prepared.gg_pair.shape[0]
    h1 = np.zeros(2 * size, dtype=complex)

    for ns in range(size):
        for index in range(size - ns):
            h1[size - ns] += prepared.gg_pair[ns + index, index] / (
                prepared.half_alpha_denominator[index]
            )

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h1[2 * size - ns - 1] += prepared.gg_pair[
                index + ns - size, index
            ] / prepared.half_alpha_denominator[index]

    h1 *= domega
    return np.asarray(h1, dtype=complex)


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
    f1 = _pair_matrix(f1g2, f1g1, domega, flip_lr=False)
    f2 = _pair_matrix(f2g2, f2g1, domega, flip_lr=False)

    size = len(omega)
    omega_s = np.arange(-size, size, dtype=float) * domega
    h = np.zeros(2 * size, dtype=complex)

    for ns in range(size):
        for index in range(size - ns):
            h[ns + size] += f1[ns - index, index] / (
                alpha + 1j * omega_s[ns + size]
            )

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h[ns] += f1[ns - index, index] / (alpha + 1j * omega_s[ns])

    h *= domega
    output = 0j
    for omega_index in range(size):
        for omega_prime_index in range(size):
            output += h[omega_index + omega_prime_index] * f2[
                omega_index, omega_prime_index
            ]
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

    size = len(omega)
    omega_s = np.arange(-size, size, dtype=float) * domega
    h = np.zeros(2 * size, dtype=complex)

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h[2 * size - ns - 1] += f1[index + ns - size, index]

    for ns in range(size):
        for index in range(size - ns):
            h[size - ns] += f1[ns + index, index]

    h *= domega
    output = 0j
    for omega_index in range(size):
        for omega_prime_index in range(size):
            output += h[omega_index - omega_prime_index + size] * f2[
                omega_prime_index, omega_index
            ] / (alpha + 1j * omega_s[omega_index + omega_prime_index])
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

    size = len(omega)
    h1 = np.zeros(2 * size, dtype=complex)
    h2 = np.zeros(2 * size, dtype=complex)

    for ns in range(size):
        for index in range(size - ns):
            h1[size - ns] += f1[ns + index, index] / (alpha / 2 + 1j * omega[index])

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h1[2 * size - ns - 1] += f1[index + ns - size, index] / (
                alpha / 2 + 1j * omega[index]
            )

    h1 *= domega
    inv_lambda_1 = 0j
    for omega_tilde_index in range(size):
        for omega_prime_index in range(size):
            inv_lambda_1 += h1[omega_tilde_index - omega_prime_index + size] * f2[
                omega_tilde_index, omega_prime_index
            ]
    inv_lambda_1 *= domega**2

    for ns in range(size):
        for index in range(size - ns):
            h2[size - ns] += f1[ns + index, index]

    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            h2[2 * size - ns - 1] += f1[index + ns - size, index]

    h2 *= domega
    inv_lambda_2 = 0j
    for omega_tilde_index in range(size):
        for omega_prime_index in range(size):
            inv_lambda_2 += h2[omega_tilde_index - omega_prime_index + size] * f2[
                omega_tilde_index, omega_prime_index
            ] / (alpha / 2 + 1j * omega[omega_prime_index])
    inv_lambda_2 *= domega**2

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
    f1 = np.asarray((f1g2 @ np.transpose(f1g1)) * domega, dtype=complex)
    f2 = np.asarray((f2g2 @ np.transpose(f2g1)) * domega, dtype=complex)
    tal = np.sum(f1) * domega**2 * np.sum(f2) * domega**2
    return complex(tal)


def _g2_denominator(gjj: ComplexArray, domega: float) -> complex:
    f = np.asarray((np.conj(gjj) @ np.transpose(gjj)) * domega, dtype=complex)
    tal = np.sum(f) * domega**2
    return complex(tal)


def calculate_indistinguishable_tpa_overlap(
    inputs: IndistinguishableTPAInputs,
) -> TPAContributionBreakdown:
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

    normalized = _normalize_tpa_inputs(
        inputs.g,
        inputs.f,
        inputs.omega,
        inputs.domega,
        inputs.gamma,
        inputs.omega_fg,
    )
    g_conjugate = np.conj(normalized.g)
    f_conjugate = np.conj(normalized.f)
    prepared = _prepare_overlap_terms(
        normalized.g,
        normalized.f,
        normalized.omega,
        normalized.domega,
        normalized.gamma,
        normalized.omega_fg,
    )

    coherent = _coherent_overlap_contribution_prepared(
        prepared,
        normalized.domega,
    )

    incoherent_type1 = _incoherent_overlap_contribution_type1_prepared(
        prepared,
        normalized.domega,
    )

    incoherent_type2 = _incoherent_overlap_contribution_type2_prepared(
        prepared,
        normalized.domega,
    )
    incoherent_total = incoherent_type1 + incoherent_type2

    g2_numerator_single_cross = _g2_numerator(
        g_conjugate,
        f_conjugate,
        normalized.g,
        normalized.f,
        normalized.domega,
    )
    g2_numerator_single_same = _g2_numerator(
        g_conjugate,
        normalized.g,
        normalized.g,
        g_conjugate,
        normalized.domega,
    )
    g2_numerator = (
        g2_numerator_single_cross + g2_numerator_single_same + g2_numerator_single_same
    )

    g2_denominator_single = _g2_denominator(normalized.g, normalized.domega)
    g2_denominator = g2_denominator_single**2
    if np.isclose(g2_denominator, 0.0):
        raise ValueError(
            "g2 denominator is zero or numerically singular for the supplied Green's function."
        )
    g2_value = g2_numerator / g2_denominator

    g_scale = normalized.g_scale
    f_scale = normalized.f_scale
    coherent *= g_scale**2 * f_scale**2
    incoherent_type1 *= g_scale**4
    incoherent_type2 *= g_scale**4
    incoherent_total = incoherent_type1 + incoherent_type2
    g2_numerator_single_cross *= g_scale**2 * f_scale**2
    g2_numerator_single_same *= g_scale**4
    g2_numerator = (
        g2_numerator_single_cross + g2_numerator_single_same + g2_numerator_single_same
    )
    g2_denominator_single *= g_scale**2
    g2_denominator = g2_denominator_single**2
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


def calculate_distinguishable_tpa_overlap(
    inputs: DistinguishableTPAInputs,
) -> tuple[TPAContributionBreakdown, TPAHFunction]:
    """Calculate the 2PA breakdown for distinguishable photons.

    This uses the current indistinguishable implementation with the same
    structural inputs, since the project does not yet ship a separate
    distinguishable solver path.
    """
    warnings.warn(
        "Distinguishable 2PA is not implemented separately; using indistinguishable logic.",
        stacklevel=2,
    )
    shared = IndistinguishableTPAInputs(
        g=inputs.g_is,
        f=inputs.g_ii,
        omega=inputs.omega,
        domega=inputs.domega,
        gamma=inputs.gamma,
        omega_fg=inputs.omega_fg,
    )
    breakdown = calculate_indistinguishable_tpa_overlap(shared)
    prepared = _prepare_overlap_terms(
        inputs.g_is,
        inputs.g_ii,
        inputs.omega,
        inputs.domega,
        inputs.gamma,
        inputs.omega_fg,
    )
    h_function = TPAHFunction(
        coherent=_coherent_h_function_prepared(prepared, inputs.domega),
        incoherent_type1=_incoherent_type1_h_function_prepared(prepared, inputs.domega),
        incoherent_type2=_incoherent_type2_h_function_prepared(prepared, inputs.domega),
        incoherent_total=np.asarray(
            _incoherent_type1_h_function_prepared(prepared, inputs.domega)
            + _incoherent_type2_h_function_prepared(prepared, inputs.domega),
            dtype=complex,
        ),
    )
    return breakdown, h_function


__all__ = [
    "DistinguishableTPAInputs",
    "IndistinguishableTPAInputs",
    "TPAHFunction",
    "TPAContributionBreakdown",
    "calculate_distinguishable_tpa_overlap",
    "calculate_indistinguishable_tpa_overlap",
]
