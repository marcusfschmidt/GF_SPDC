# %%
from __future__ import annotations


from pathlib import Path


import numpy as np

from scripts.run_stitcher import build_default_params, run_stitcher_from_params
from scripts.run_gamma_sweep import (
    run_gamma_sweep,
    _extract_schmidt_number_from_stitched,
)
from scripts.run_two_photon_absorption import run_tpa_from_file
from scripts.plot_greens import plot_stitched


# %%
# Generate stitched Green's functions
params = build_default_params(n=11, type="0", dt=0.7e-2)
green_functions, time_width_array, freq_width_array, stitch_times, saved_name = (
    run_stitcher_from_params(
        params,
        basis_width=0.2,
        pump_width=2,
        kmax=25,
        step_fraction=1.5,
    )
)

# %%


def check_green_function_identities(Gii, Gis, Gsi, Gss, dt=1.0):
    """
    Checks Green-function canonical identities for the discrete propagators
    K = dt * G and reports both the full-grid and resolved-subspace errors.
    """

    # Discrete propagators act as K @ x, where K = dt * G.
    Kii = dt * Gii
    Kis = dt * Gis
    Ksi = dt * Gsi
    Kss = dt * Gss

    # --- Identity 1 ---
    C1 = Kii @ Kii.conj().T - Kis @ Kis.conj().T

    # --- Identity 2 ---
    C2 = Kii @ Ksi.conj().T - Kis @ Kss.conj().T

    N = Gii.shape[0]
    I = np.eye(N, dtype=complex)

    # A truncated extraction with kmax << N cannot satisfy the full-grid
    # identity. Compare C1 to the projector onto its numerically resolved
    # output subspace as well.
    eigvals, eigvecs = np.linalg.eigh(C1)
    eigvals = np.real_if_close(eigvals)
    projector_mask = eigvals > 0.5
    resolved_rank = int(np.count_nonzero(projector_mask))
    if resolved_rank > 0:
        resolved_vectors = eigvecs[:, projector_mask]
        projector = resolved_vectors @ resolved_vectors.conj().T
    else:
        projector = np.zeros_like(C1)

    # --- Errors ---
    err1_full = np.linalg.norm(C1 - I)
    err1_projector = np.linalg.norm(C1 - projector)
    err2 = np.linalg.norm(C2)

    return C1, C2, err1_full, err1_projector, err2, resolved_rank, eigvals


# %%
g1 = green_functions[0]
f1 = green_functions[1]
g2 = green_functions[2]
f2 = green_functions[3]
dt = params.dt

check_green_function_identities(f1, g1, g2, f2, dt=dt)


# %%
# Plot stitched Green's functions
plot_stitched(saved_name)

# %%
print(_extract_schmidt_number_from_stitched(green_functions))

# %%
# Run a gamma sweep for 2PA
gammas = np.logspace(4, 6, num=5)  # 11 points from 1e-5 to 10, log spaced
transition_linewidth = 10
gamma_results = run_gamma_sweep(
    gammas,
    transition_linewidth=transition_linewidth,
    type="typeII",
    n=11,
    basis_width=0.5,
    pump_width=2,
    kmax=25,
    rate_factor=3,
    length=4e-3,
    print_bool=True,
    stitch_print_bool=False,
)


# %%
# Optional: compute 2PA for a single stitched file
tpa_result = run_tpa_from_file(saved_name, transition_linewidth=transition_linewidth)

# %%
from matplotlib import pyplot as plt

photon_number_array = np.array([result.photon_number for result in gamma_results])
schmidt_number_array = np.array([result.schmidt_number for result in gamma_results])

coherent = np.array([result.coherent for result in gamma_results])
incoherent_total = np.array([result.incoherent_total for result in gamma_results])

factor = 5
tpa_rate = factor * (coherent + incoherent_total)
plt.plot(photon_number_array / schmidt_number_array, coherent, marker="o")
# plt.plot(0.25 * photon_number_array / schmidt_number_array, incoherent_total)
# log x, y
plt.xscale("log")
plt.yscale("log")
plt.show()

# %%
print(photon_number_array)
print(schmidt_number_array)
# %%
h = np.array([result.h_function for result in gamma_results])
plt.plot(np.abs(h[0].coherent) ** 2)
plt.show()
