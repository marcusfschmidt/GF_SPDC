from __future__ import annotations

import numpy as np

from gf_spdc.extractor import GreenFunctionsExtractor


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
