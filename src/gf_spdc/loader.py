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
    output = np.asarray(widths, dtype=float).copy()
    output[0] = output[0] - padding_factor * (output[1] - output[0])
    output[1] = output[1] + padding_factor * (output[1] - output[0])
    output[2] = output[2] - padding_factor * (output[3] - output[2])
    output[3] = output[3] + padding_factor * (output[3] - output[2])
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
