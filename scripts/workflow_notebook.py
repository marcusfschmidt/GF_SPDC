# %%
from __future__ import annotations


from pathlib import Path

import numpy as np

from scripts.run_stitcher import build_default_params, run_stitcher_from_params
from scripts.run_gamma_sweep import run_gamma_sweep
from scripts.run_two_photon_absorption import run_tpa_from_file
from scripts.plot_greens import plot_stitched


# %%
# Generate stitched Green's functions
params = build_default_params(type="0", n=11, dt=0.7e-2)
params.gamma = 1e-5
green_functions, time_width_array, freq_width_array, stitch_times, saved_name = (
    run_stitcher_from_params(
        params,
        basis_width=0.2,
        kmax=50,
        step_fraction=5,
    )
)


# %%
# Plot stitched Green's functions
plot_stitched(saved_name)


# %%
# Run a gamma sweep for 2PA
gammas = np.logspace(-5, 2, num=15)  # 11 points from 1e-5 to 10, log spaced
gammas = [100]
gamma_results = run_gamma_sweep(
    gammas,
    type="typeII",
    n=11,
    basis_width=0.5,
    kmax=25,
    rate_factor=1.0,
)


# %%
# Optional: compute 2PA for a single stitched file
tpa_result = run_tpa_from_file(saved_name)

# %%
from matplotlib import pyplot as plt

photon_number_array = np.array([result.photon_number for result in gamma_results])
schmidt_number_array = np.array([result.schmidt_number for result in gamma_results])

coherent = np.array([result.coherent for result in gamma_results])
incoherent_total = np.array([result.incoherent_total for result in gamma_results])

factor = 5
tpa_rate = factor * (coherent + incoherent_total)

plt.plot(photon_number_array / schmidt_number_array, coherent, marker="o")
# log x, y
plt.xscale("log")
plt.yscale("log")
plt.show()

# %%
h = np.array([result.h_function for result in gamma_results])
plt.plot(np.abs(h[0].coherent) ** 2)
plt.show()
