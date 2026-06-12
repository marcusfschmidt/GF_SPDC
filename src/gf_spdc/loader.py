from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[Any]
FloatArray = NDArray[np.float64]


def fft2_shifted(field: ComplexArray) -> ComplexArray:
    field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes=0), axis=0), axes=0)
    field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes=1), axis=1), axes=1)
    return field


def ifft2_shifted(field: ComplexArray) -> ComplexArray:
    field = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field, axes=0), axis=0), axes=0)
    field = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field, axes=1), axis=1), axes=1)
    return field


def add_padding_to_width(widths: NDArray[np.int64] | NDArray[np.float64], padding_factor: float = 0.25) -> NDArray[np.float64]:
    x1, x2, y1, y2 = np.asarray(widths, dtype=float)
    output = np.empty(4, dtype=float)
    output[0] = x1 - padding_factor * (x2 - x1)
    output[1] = x2 + padding_factor * (x2 - x1)
    output[2] = y1 - padding_factor * (y2 - y1)
    output[3] = y2 + padding_factor * (y2 - y1)
    return output


def remove_zero_values(
    g_field: ComplexArray,
    f_field: ComplexArray,
    input_axis: FloatArray,
    index_array: NDArray[np.int64] | NDArray[np.float64] | tuple[int, int, int, int],
) -> tuple[ComplexArray, ComplexArray, FloatArray, FloatArray]:
    x1, x2, y1, y2 = [int(value) for value in index_array]
    g_output = g_field[y1:y2, x1:x2]
    f_output = f_field[y1:y2, x1:x2]
    return g_output, f_output, input_axis[x1:x2], input_axis[y1:y2]
