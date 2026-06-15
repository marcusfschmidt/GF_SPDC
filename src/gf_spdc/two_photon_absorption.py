from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

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
class _OverlapIndexCache:
    coherent_destinations: IntArray
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
        raise ValueError("Green's function dimensions must match the omega axis length.")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(omega)):
        raise ValueError("Green's functions and omega must contain only finite values.")


def _pair_matrix(left: ComplexArray, right: ComplexArray, domega: float, flip_lr: bool) -> ComplexArray:
    matrix = np.asarray((left @ np.transpose(right)) * domega, dtype=complex)
    return np.fliplr(matrix) if flip_lr else matrix


def _complex_bincount(indices: IntArray, values: ComplexArray, length: int) -> ComplexArray:
    values_array = np.asarray(values, dtype=complex).ravel()
    return np.bincount(indices, weights=np.asarray(np.real(values_array), dtype=float), minlength=length) + 1j * np.bincount(
        indices,
        weights=np.asarray(np.imag(values_array), dtype=float),
        minlength=length,
    )


@lru_cache(maxsize=None)
def _overlap_indices(size: int) -> _OverlapIndexCache:
    rows_raw, cols_raw = np.indices((size, size), dtype=int)
    rows = np.asarray(rows_raw, dtype=int)
    cols = np.asarray(cols_raw, dtype=int)
    coherent_destinations = np.asarray(np.where(rows > 0, rows + cols, size + cols).ravel(), dtype=int)
    diagonal_destinations = np.asarray(np.where(cols > rows, size + cols - rows - 1, size + cols - rows).ravel(), dtype=int)
    anti_diagonal_indices = np.asarray(rows + cols, dtype=int)
    forward_difference_indices = np.asarray(cols - rows + size, dtype=int)
    backward_difference_indices = np.asarray(rows - cols + size, dtype=int)
    return _OverlapIndexCache(
        coherent_destinations,
        diagonal_destinations,
        anti_diagonal_indices,
        forward_difference_indices,
        backward_difference_indices,
    )


@dataclass(frozen=True)
class _PreparedOverlapTerms:
    alpha: complex
    omega_s: FloatArray
    coherent_denominator: ComplexArray
    half_alpha_denominator: ComplexArray
    fg_pair: ComplexArray
    fg_pair_conjugate: ComplexArray
    gg_pair: ComplexArray
    gg_pair_conjugate: ComplexArray
    coherent_projection_fg: ComplexArray
    anti_diagonal_projection_fg: ComplexArray
    diagonal_projection_gg: ComplexArray
    forward_projection_gg_conjugate: ComplexArray
    backward_projection_gg_conjugate: ComplexArray


def _prepare_overlap_terms(
    g: ComplexArray,
    f: ComplexArray,
    omega: FloatArray,
    domega: float,
    gamma: float,
    omega_fg: float,
) -> _PreparedOverlapTerms:
    size = g.shape[0]
    overlap_indices = _overlap_indices(size)
    alpha = gamma - 1j * omega_fg
    omega_s = np.arange(-size, size, dtype=float) * domega
    coherent_denominator = 1.0 / (alpha + 1j * omega_s)
    half_alpha_denominator = alpha / 2 + 1j * omega

    fg_pair = _pair_matrix(f, g, domega, flip_lr=True)
    fg_pair_conjugate = np.conj(fg_pair)
    gg_pair = _pair_matrix(g, np.conj(g), domega, flip_lr=True)
    gg_pair_conjugate = np.conj(gg_pair)

    coherent_projection_fg = _complex_bincount(
        overlap_indices.coherent_destinations,
        fg_pair_conjugate,
        2 * size,
    )
    anti_diagonal_projection_fg = _complex_bincount(
        np.asarray(overlap_indices.anti_diagonal_indices.ravel(), dtype=int),
        fg_pair,
        2 * size - 1,
    )
    diagonal_projection_gg = _complex_bincount(
        overlap_indices.diagonal_destinations,
        gg_pair,
        2 * size,
    )
    forward_projection_gg_conjugate = _complex_bincount(
        np.asarray(overlap_indices.forward_difference_indices.ravel(), dtype=int),
        gg_pair_conjugate,
        2 * size,
    )
    backward_projection_gg_conjugate = _complex_bincount(
        np.asarray(overlap_indices.backward_difference_indices.ravel(), dtype=int),
        gg_pair_conjugate,
        2 * size,
    )

    return _PreparedOverlapTerms(
        alpha=alpha,
        omega_s=omega_s,
        coherent_denominator=np.asarray(coherent_denominator, dtype=complex),
        half_alpha_denominator=np.asarray(half_alpha_denominator, dtype=complex),
        fg_pair=fg_pair,
        fg_pair_conjugate=fg_pair_conjugate,
        gg_pair=gg_pair,
        gg_pair_conjugate=gg_pair_conjugate,
        coherent_projection_fg=coherent_projection_fg,
        anti_diagonal_projection_fg=anti_diagonal_projection_fg,
        diagonal_projection_gg=diagonal_projection_gg,
        forward_projection_gg_conjugate=forward_projection_gg_conjugate,
        backward_projection_gg_conjugate=backward_projection_gg_conjugate,
    )


def _coherent_overlap_contribution_prepared(
    prepared: _PreparedOverlapTerms,
    domega: float,
) -> float:
    h = prepared.coherent_projection_fg * prepared.coherent_denominator
    h *= domega
    output = np.dot(h[: prepared.anti_diagonal_projection_fg.size], prepared.anti_diagonal_projection_fg) * domega**2
    return float(np.real(output))


def _incoherent_overlap_contribution_type1_prepared(
    prepared: _PreparedOverlapTerms,
    domega: float,
) -> float:
    overlap_indices = _overlap_indices(prepared.gg_pair.shape[0])
    anti_diagonal_denominator = prepared.alpha + 1j * prepared.omega_s[:-1]
    h = prepared.diagonal_projection_gg.copy()
    h *= domega
    weighted_forward_projection = _complex_bincount(
        np.asarray(overlap_indices.forward_difference_indices.ravel(), dtype=int),
        prepared.gg_pair_conjugate
        / anti_diagonal_denominator[overlap_indices.anti_diagonal_indices],
        2 * prepared.gg_pair.shape[0],
    )
    output = np.dot(h, weighted_forward_projection) * domega**2
    return float(np.real(output))


def _incoherent_overlap_contribution_type2_prepared(
    prepared: _PreparedOverlapTerms,
    domega: float,
) -> float:
    overlap_indices = _overlap_indices(prepared.gg_pair.shape[0])
    h1 = _complex_bincount(
        overlap_indices.diagonal_destinations,
        prepared.gg_pair / prepared.half_alpha_denominator[np.newaxis, :],
        2 * prepared.gg_pair.shape[0],
    )
    h1 *= domega
    inv_lambda_1 = np.dot(h1, prepared.backward_projection_gg_conjugate) * domega**2

    weighted_backward_projection = _complex_bincount(
        np.asarray(overlap_indices.backward_difference_indices.ravel(), dtype=int),
        prepared.gg_pair_conjugate / prepared.half_alpha_denominator[np.newaxis, :],
        2 * prepared.gg_pair.shape[0],
    )
    inv_lambda_2 = np.dot(prepared.diagonal_projection_gg * domega, weighted_backward_projection) * domega**2

    if np.isclose(inv_lambda_1, 0.0) or np.isclose(inv_lambda_2, 0.0):
        raise ValueError("Incoherent type-2 contribution is singular for the supplied inputs.")

    lambda_total = 1 / inv_lambda_1 + 1 / inv_lambda_2
    output = 1 / lambda_total
    return float(np.real(output))


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

    size = len(omega)
    overlap_indices = _overlap_indices(size)
    omega_s = np.arange(-size, size, dtype=float) * domega
    h = (
        _complex_bincount(overlap_indices.coherent_destinations, f1, 2 * size)
        / (alpha + 1j * omega_s)
    )
    h *= domega
    output = np.sum(h[overlap_indices.anti_diagonal_indices] * f2) * domega**2
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
    overlap_indices = _overlap_indices(size)
    omega_s = np.arange(-size, size, dtype=float) * domega
    h = _complex_bincount(overlap_indices.diagonal_destinations, f1, 2 * size)
    h *= domega
    output = np.sum(
        h[overlap_indices.forward_difference_indices]
        * f2
        / (alpha + 1j * omega_s[overlap_indices.anti_diagonal_indices])
    ) * domega**2
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
    overlap_indices = _overlap_indices(size)
    half_alpha_denominator = alpha / 2 + 1j * omega
    h1 = _complex_bincount(
        overlap_indices.diagonal_destinations,
        f1 / half_alpha_denominator[np.newaxis, :],
        2 * size,
    )
    h1 *= domega
    inv_lambda_1 = np.sum(h1[overlap_indices.backward_difference_indices] * f2) * domega**2

    h2 = _complex_bincount(overlap_indices.diagonal_destinations, f1, 2 * size)
    h2 *= domega
    inv_lambda_2 = np.sum(
        h2[overlap_indices.backward_difference_indices]
        * f2
        / half_alpha_denominator[np.newaxis, :]
    ) * domega**2

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
    f1_sum = np.sum(f1g2, axis=0) @ np.sum(f1g1, axis=0)
    f2_sum = np.sum(f2g2, axis=0) @ np.sum(f2g1, axis=0)
    return complex(f1_sum * f2_sum * domega**6)


def _g2_denominator(gjj: ComplexArray, domega: float) -> complex:
    gjj_sum = np.sum(gjj, axis=0)
    return complex((np.conj(gjj_sum) @ gjj_sum) * domega**3)


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
    prepared = _prepare_overlap_terms(
        inputs.g,
        inputs.f,
        inputs.omega,
        inputs.domega,
        inputs.gamma,
        inputs.omega_fg,
    )

    coherent = _coherent_overlap_contribution_prepared(
        prepared,
        inputs.domega,
    )

    incoherent_type1 = _incoherent_overlap_contribution_type1_prepared(
        prepared,
        inputs.domega,
    )

    incoherent_type2 = _incoherent_overlap_contribution_type2_prepared(
        prepared,
        inputs.domega,
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
