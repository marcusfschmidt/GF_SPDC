from __future__ import annotations

import numpy as np

from gf_spdc.loader import add_padding_to_width, fft2_shifted, ifft2_shifted, remove_zero_values


def test_fft_round_trip() -> None:
    field = np.arange(16, dtype=float).reshape(4, 4) + 1j * np.eye(4)
    recovered = ifft2_shifted(fft2_shifted(field))
    np.testing.assert_allclose(recovered, field)


def test_add_padding_to_width_expands_bounds() -> None:
    padded = add_padding_to_width(np.array([10, 20, 30, 40]), 0.25)
    np.testing.assert_allclose(padded, np.array([7.5, 23.125, 27.5, 43.125]))


def test_remove_zero_values_slices_consistently() -> None:
    green = np.arange(36, dtype=float).reshape(6, 6).astype(complex)
    field = (green + 1).astype(complex)
    axis = np.linspace(-3, 2, 6)
    cropped_green, cropped_field, x_axis, y_axis = remove_zero_values(green, field, axis, (1, 4, 2, 5))

    np.testing.assert_array_equal(cropped_green, green[2:5, 1:4])
    np.testing.assert_array_equal(cropped_field, field[2:5, 1:4])
    np.testing.assert_array_equal(x_axis, axis[1:4])
    np.testing.assert_array_equal(y_axis, axis[2:5])
