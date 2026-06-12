from __future__ import annotations

import numpy as np

from gf_spdc.extractor import GreenFunctionsExtractor, _parallel_green


def make_extractor(dt: float = 0.5) -> GreenFunctionsExtractor:
    extractor = GreenFunctionsExtractor(kmax=2, debug_bool=False)
    extractor.dt = dt
    return extractor


def test_calc_overlap_is_unity_for_identical_fields() -> None:
    extractor = make_extractor()
    field = np.array([1 + 1j, 2 - 1j, 0.5 + 0.25j], dtype=complex)
    overlap = extractor.calc_overlap(field.copy(), field.copy())
    np.testing.assert_approx_equal(overlap, 1.0, significant=12)


def test_check_schmidt_numbers_is_difference_of_squares() -> None:
    extractor = make_extractor()
    rho = np.array([3.0, 4.0])
    nu = np.array([1.0, 2.0])
    np.testing.assert_allclose(extractor.check_schmidt_numbers(rho, nu), np.array([8.0, 12.0]))


def test_calculate_green_overlap_returns_expected_shape_for_identity_kernel() -> None:
    extractor = make_extractor(dt=1.0)
    g_cross = np.eye(3, dtype=complex)
    g_self = np.eye(3, dtype=complex)
    field_array = np.zeros((3, 2, 3), dtype=complex)
    field_array[0, 0, :] = np.array([1.0, 0.0, 0.0])
    field_array[1, 0, :] = np.array([1.0, 0.0, 0.0])
    field_array[2, 0, :] = np.array([1.0, 0.0, 0.0])
    field_array[0, 1, :] = np.array([0.0, 1.0, 0.0])
    field_array[1, 1, :] = np.array([0.0, 1.0, 0.0])
    field_array[2, 1, :] = np.array([0.0, 1.0, 0.0])

    overlaps = extractor.calculate_green_overlap(g_cross, g_self, field_array)
    assert overlaps.shape == (2, 2)
    np.testing.assert_allclose(overlaps, np.ones((2, 2)))


def test_parallel_green_matches_explicit_mode_sum() -> None:
    v_modes = np.array([[1 + 1j, 2 - 1j], [0.5 + 0j, -1j]], dtype=complex)
    u_modes = np.array([[2 - 1j, 0.5 + 0.5j], [1 + 0j, -2j]], dtype=complex)
    rho_values = np.array([0.5, 1.5], dtype=float)

    result = _parallel_green((v_modes, rho_values, u_modes))
    expected = sum(rho_values[index] * np.outer(v_modes[index], np.conjugate(u_modes[index])) for index in range(2))
    np.testing.assert_allclose(result, expected)
