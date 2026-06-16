from __future__ import annotations

import numpy as np
import pytest

from gf_spdc.loader import fft2_shifted
from gf_spdc.two_photon_absorption import IndistinguishableTPAInputs, calculate_indistinguishable_tpa_overlap
from gf_spdc.two_photon_absorption import (
    _coherent_overlap_contribution,
    _coherent_h_function_prepared,
    _g2_denominator,
    _g2_numerator,
    _incoherent_type1_h_function_prepared,
    _incoherent_type2_h_function_prepared,
    _incoherent_overlap_contribution_type1,
    _incoherent_overlap_contribution_type2,
    _overlap_indices,
    _prepare_overlap_terms,
)


def _coherent_overlap_contribution_reference(
    f1g1: np.ndarray,
    f1g2: np.ndarray,
    f2g1: np.ndarray,
    f2g2: np.ndarray,
    omega: np.ndarray,
    domega: float,
    gamma: float,
    omega_fg: float,
) -> float:
    alpha = gamma - 1j * omega_fg
    f1 = np.asarray((f1g2 @ np.transpose(f1g1)) * domega, dtype=complex)
    f2 = np.asarray((f2g2 @ np.transpose(f2g1)) * domega, dtype=complex)

    omega_s = np.arange(-len(omega), len(omega), dtype=float) * domega
    h = np.zeros(2 * len(omega), dtype=complex)

    for ns in range(len(omega)):
        denominator = alpha + 1j * omega_s[ns + len(omega)]
        for index in range(len(omega) - ns):
            h[ns + len(omega)] += f1[ns - index, index] / denominator

    for ns in reversed(range(len(omega))):
        denominator = alpha + 1j * omega_s[ns]
        for index in range(len(omega) - ns, len(omega)):
            h[ns] += f1[ns - index, index] / denominator

    h *= domega
    output = 0j
    for omega_index in range(len(omega)):
        for omega_prime_index in range(len(omega)):
            output += h[omega_index + omega_prime_index] * f2[omega_index, omega_prime_index]
    output *= domega**2
    return float(np.real(output))


def _incoherent_overlap_contribution_type1_reference(
    f1g1: np.ndarray,
    f1g2: np.ndarray,
    f2g1: np.ndarray,
    f2g2: np.ndarray,
    omega: np.ndarray,
    domega: float,
    gamma: float,
    omega_fg: float,
) -> float:
    alpha = gamma - 1j * omega_fg
    f1 = np.asarray((f1g2 @ np.transpose(f1g1)) * domega, dtype=complex)
    f1 = np.fliplr(f1)
    f2 = np.asarray((f2g2 @ np.transpose(f2g1)) * domega, dtype=complex)
    f2 = np.fliplr(f2)

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


def _incoherent_overlap_contribution_type2_reference(
    f1g1: np.ndarray,
    f1g2: np.ndarray,
    f2g1: np.ndarray,
    f2g2: np.ndarray,
    omega: np.ndarray,
    domega: float,
    gamma: float,
    omega_fg: float,
) -> float:
    alpha = gamma - 1j * omega_fg
    f1 = np.asarray((f1g2 @ np.transpose(f1g1)) * domega, dtype=complex)
    f1 = np.fliplr(f1)
    f2 = np.asarray((f2g2 @ np.transpose(f2g1)) * domega, dtype=complex)
    f2 = np.fliplr(f2)

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

    lambda_total = 1 / inv_lambda_1 + 1 / inv_lambda_2
    return float(np.real(1 / lambda_total))


def make_inputs() -> IndistinguishableTPAInputs:
    g = np.array(
        [
            [1.0 + 0.0j, 0.1 - 0.2j],
            [0.05 + 0.3j, 0.8 - 0.1j],
        ],
        dtype=complex,
    )
    f = np.array(
        [
            [0.6 + 0.1j, -0.2 + 0.0j],
            [0.15 - 0.05j, 0.4 + 0.2j],
        ],
        dtype=complex,
    )
    omega = np.array([-0.5, 0.5], dtype=float)
    return IndistinguishableTPAInputs(
        g=g, f=f, omega=omega, domega=1.0, transition_linewidth=2.0
    )


def test_indistinguishable_overlap_uses_three_specified_contributions() -> None:
    inputs = make_inputs()
    result = calculate_indistinguishable_tpa_overlap(inputs)

    coherent_expected = _coherent_overlap_contribution(
        np.conj(inputs.g),
        np.conj(inputs.f),
        inputs.g,
        inputs.f,
        inputs.omega,
        inputs.domega,
        inputs.transition_linewidth,
        inputs.omega_fg,
    )

    incoherent_type1_expected = _incoherent_overlap_contribution_type1(
        np.conj(inputs.g),
        inputs.g,
        inputs.g,
        np.conj(inputs.g),
        inputs.omega,
        inputs.domega,
        inputs.transition_linewidth,
        inputs.omega_fg,
    )

    incoherent_type2_expected = _incoherent_overlap_contribution_type2(
        np.conj(inputs.g),
        inputs.g,
        inputs.g,
        np.conj(inputs.g),
        inputs.omega,
        inputs.domega,
        inputs.transition_linewidth,
        inputs.omega_fg,
    )

    np.testing.assert_allclose(result.coherent, coherent_expected)
    np.testing.assert_allclose(result.incoherent_type1, incoherent_type1_expected)
    np.testing.assert_allclose(result.incoherent_type2, incoherent_type2_expected)
    np.testing.assert_allclose(result.total, coherent_expected + incoherent_type1_expected + incoherent_type2_expected)


def test_indistinguishable_h_functions_store_weighted_h_vectors() -> None:
    inputs = make_inputs()
    prepared = _prepare_overlap_terms(
        inputs.g,
        inputs.f,
        inputs.omega,
        inputs.domega,
        inputs.transition_linewidth,
        inputs.omega_fg,
    )

    coherent_h = _coherent_h_function_prepared(prepared, inputs.domega)
    incoherent_type1_h = _incoherent_type1_h_function_prepared(prepared, inputs.domega)
    incoherent_type2_h = _incoherent_type2_h_function_prepared(prepared, inputs.domega)

    expected_coherent_h = np.zeros(2 * prepared.fg_pair.shape[0], dtype=complex)
    size = prepared.fg_pair.shape[0]
    for ns in range(size):
        for index in range(size - ns):
            expected_coherent_h[ns + size] += prepared.fg_pair_conjugate[ns - index, index] / (
                prepared.alpha + 1j * prepared.omega_s[ns + size]
            )
    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            expected_coherent_h[ns] += prepared.fg_pair_conjugate[ns - index, index] / (
                prepared.alpha + 1j * prepared.omega_s[ns]
            )
    expected_coherent_h *= inputs.domega

    np.testing.assert_allclose(
        coherent_h,
        expected_coherent_h,
    )

    expected_type1 = np.zeros(2 * size, dtype=complex)
    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            expected_type1[2 * size - ns - 1] += prepared.gg_pair[index + ns - size, index]
    for ns in range(size):
        for index in range(size - ns):
            expected_type1[size - ns] += prepared.gg_pair[ns + index, index]
    expected_type1 *= inputs.domega
    np.testing.assert_allclose(
        incoherent_type1_h,
        expected_type1,
    )

    expected_type2 = np.zeros(2 * size, dtype=complex)
    for ns in range(size):
        for index in range(size - ns):
            expected_type2[size - ns] += prepared.gg_pair[ns + index, index] / prepared.half_alpha_denominator[index]
    for ns in reversed(range(size)):
        for index in range(size - ns, size):
            expected_type2[2 * size - ns - 1] += prepared.gg_pair[index + ns - size, index] / prepared.half_alpha_denominator[index]
    expected_type2 *= inputs.domega
    np.testing.assert_allclose(incoherent_type2_h, expected_type2)


def test_fft2_shifted_is_the_expected_domain_transform() -> None:
    field = np.array(
        [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
        dtype=complex,
    )
    transformed = fft2_shifted(field)

    assert transformed.shape == field.shape
    assert transformed.dtype == complex
    assert not np.allclose(transformed, field)


def test_indistinguishable_g2_reduction_matches_expanded_sum() -> None:
    inputs = make_inputs()
    result = calculate_indistinguishable_tpa_overlap(inputs)

    single_cross = _g2_numerator(np.conj(inputs.g), np.conj(inputs.f), inputs.g, inputs.f, inputs.domega)
    single_same = _g2_numerator(np.conj(inputs.g), inputs.g, inputs.g, np.conj(inputs.g), inputs.domega)
    expanded = single_cross + single_same + single_same
    denominator = _g2_denominator(inputs.g, inputs.domega) ** 2

    np.testing.assert_allclose(result.g2_numerator, expanded)
    np.testing.assert_allclose(result.g2_denominator, denominator)
    np.testing.assert_allclose(result.g2, expanded / denominator)


def test_vectorized_overlap_contributions_match_reference_implementation() -> None:
    rng = np.random.default_rng(7)
    size = 5
    g = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    f = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    omega = np.linspace(-2.0, 2.0, size)
    domega = omega[1] - omega[0]
    gamma = 1.7
    omega_fg = -0.4

    np.testing.assert_allclose(
        _coherent_overlap_contribution(np.conj(g), np.conj(f), g, f, omega, domega, gamma, omega_fg),
        _coherent_overlap_contribution_reference(np.conj(g), np.conj(f), g, f, omega, domega, gamma, omega_fg),
    )
    np.testing.assert_allclose(
        _incoherent_overlap_contribution_type1(np.conj(g), g, g, np.conj(g), omega, domega, gamma, omega_fg),
        _incoherent_overlap_contribution_type1_reference(np.conj(g), g, g, np.conj(g), omega, domega, gamma, omega_fg),
    )
    np.testing.assert_allclose(
        _incoherent_overlap_contribution_type2(np.conj(g), g, g, np.conj(g), omega, domega, gamma, omega_fg),
        _incoherent_overlap_contribution_type2_reference(np.conj(g), g, g, np.conj(g), omega, domega, gamma, omega_fg),
    )


def test_tpa_input_validation_rejects_shape_mismatch() -> None:
    inputs = make_inputs()
    with pytest.raises(ValueError):
        calculate_indistinguishable_tpa_overlap(
            IndistinguishableTPAInputs(
                g=inputs.g,
                f=np.ones((3, 3), dtype=complex),
                omega=inputs.omega,
                domega=inputs.domega,
                transition_linewidth=inputs.transition_linewidth,
            )
        )


def test_tpa_input_validation_rejects_non_positive_domega() -> None:
    inputs = make_inputs()
    with pytest.raises(ValueError):
        calculate_indistinguishable_tpa_overlap(
            IndistinguishableTPAInputs(
                g=inputs.g,
                f=inputs.f,
                omega=inputs.omega,
                domega=0.0,
                transition_linewidth=inputs.transition_linewidth,
            )
        )


def test_tpa_input_validation_rejects_non_positive_gamma() -> None:
    inputs = make_inputs()
    with pytest.raises(ValueError):
        calculate_indistinguishable_tpa_overlap(
            IndistinguishableTPAInputs(
                g=inputs.g,
                f=inputs.f,
                omega=inputs.omega,
                domega=inputs.domega,
                transition_linewidth=0.0,
            )
        )


def test_tpa_input_validation_rejects_nonuniform_omega_grid() -> None:
    inputs = make_inputs()
    with pytest.raises(ValueError):
        calculate_indistinguishable_tpa_overlap(
            IndistinguishableTPAInputs(
                g=inputs.g,
                f=inputs.f,
                omega=np.array([-0.5, 0.7], dtype=float),
                domega=1.0,
                transition_linewidth=inputs.transition_linewidth,
            )
        )


def test_tpa_rejects_singular_g2_denominator() -> None:
    omega = np.array([-0.5, 0.5], dtype=float)
    zero_green = np.zeros((2, 2), dtype=complex)
    with pytest.raises(ValueError):
        calculate_indistinguishable_tpa_overlap(
            IndistinguishableTPAInputs(
                g=zero_green,
                f=np.eye(2, dtype=complex),
                omega=omega,
                domega=1.0,
                transition_linewidth=1.0,
            )
        )
