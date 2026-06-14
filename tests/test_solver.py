from __future__ import annotations

import numpy as np

from gf_spdc.solver import CoupledModes


def _make_solver(integration_method: str) -> CoupledModes:
    beta = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    return CoupledModes(
        n=6,
        dt=0.5e-2,
        dz=0.2e-3,
        length=1e-3,
        beta=beta,
        gamma=0.1,
        lambda_p=532e-9,
        omega_s=0.0,
        omega_i=0.0,
        rtol=1e-4,
        nsteps=8000,
        integration_method=integration_method,
    )


def test_fixed_step_rk4_tracks_adaptive_solver_on_small_case() -> None:
    adaptive = _make_solver("adaptive")
    rk4 = _make_solver("rk4")

    pulse = adaptive.make_gaussian_input(2.0)
    init_conditions = np.array([pulse, pulse, pulse])

    adaptive.set_initial_conditions(init_conditions)
    rk4.set_initial_conditions(init_conditions)

    _, _, _, _, _, adaptive_field_time = adaptive.run()
    _, _, _, _, _, rk4_field_time = rk4.run()

    adaptive_final = adaptive_field_time[-1, :, 0]
    rk4_final = rk4_field_time[-1, :, 0]

    overlap = np.abs(np.sum(np.conjugate(adaptive_final) * rk4_final) * adaptive.dt) ** 2
    overlap /= (
        np.sum(np.abs(adaptive_final) ** 2) * adaptive.dt
        * np.sum(np.abs(rk4_final) ** 2) * adaptive.dt
    )
    assert overlap > 0.999999
