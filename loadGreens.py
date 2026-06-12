from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gf_spdc.loader import add_padding_to_width, fft2_shifted, remove_zero_values


def main() -> None:
    # filename = "stitchedGreens_type0_gamma 1e-05_T0p 2_L 0.004.npy"
    filename = "stitchedGreens_typeII_gamma 0.01_T0p 100_L 0.004.npy"

    green_array, time_width_array, freq_width_array, stitch_times, t, omega, lambda_axis, parameter_array = np.load(
        filename,
        allow_pickle=True,
    )
    _ = stitch_times, omega, parameter_array

    time_width_array = add_padding_to_width(time_width_array, 1)
    freq_width_array = add_padding_to_width(freq_width_array, 0.2)
    green_time, f_time, tx, ty = remove_zero_values(green_array[0], green_array[1], t, time_width_array)

    green_frequency = fft2_shifted(green_array[0])
    f_frequency = fft2_shifted(green_array[1])
    green_frequency, f_frequency, lx, ly = remove_zero_values(green_frequency, f_frequency, lambda_axis, freq_width_array)
    _ = f_time, f_frequency

    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    ax1.imshow(np.abs(green_time), origin="lower", extent=[tx[0], tx[-1], ty[0], ty[-1]])
    ax1.set_title("abs")
    ax2.imshow(np.real(green_time), origin="lower", extent=[tx[0], tx[-1], ty[0], ty[-1]])
    ax2.set_title("real")
    ax3.imshow(np.imag(green_time), origin="lower", extent=[tx[0], tx[-1], ty[0], ty[-1]])
    ax3.set_title("imag")
    ax1.set_xlabel("t [ps]")
    ax2.set_xlabel("t [ps]")
    ax3.set_xlabel("t [ps]")
    ax1.set_ylabel("t [ps]")
    figure.suptitle("G(t, t')")

    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    ax1.imshow(np.abs(green_frequency), origin="lower", extent=[lx[0], lx[-1], ly[0], ly[-1]])
    ax1.set_title("abs")
    ax2.imshow(np.real(green_frequency), origin="lower", extent=[lx[0], lx[-1], ly[0], ly[-1]])
    ax2.set_title("real")
    ax3.imshow(np.imag(green_frequency), origin="lower", extent=[lx[0], lx[-1], ly[0], ly[-1]])
    ax3.set_title("imag")
    ax1.set_xlabel("lambda [nm]")
    ax2.set_xlabel("lambda [nm]")
    ax3.set_xlabel("lambda [nm]")
    ax1.set_ylabel("lambda [nm]")
    figure.suptitle("G(lambda, lambda')")
    plt.show()


if __name__ == "__main__":
    main()
