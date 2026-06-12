from gf_spdc.solver import BetaModel, CoupledModes as _CoupledModes, TaylorBeta


class CoupledModes(_CoupledModes):
    def __init__(
        self,
        n: int,
        dt: float,
        dz: float,
        L: float,
        beta: BetaModel | TaylorBeta,
        gamma: float,
        lambda_p: float,
        omega_s: float = 0.0,
        omega_i: float = 0.0,
        alpha_s: float = 0.0,
        alpha_i: float = 0.0,
        printBool: bool = False,
        rtol: float = 1e-3,
        nsteps: int = 10000,
    ) -> None:
        super().__init__(n, dt, dz, L, beta, gamma, lambda_p, omega_s, omega_i, alpha_s, alpha_i, printBool, rtol, nsteps)

__all__ = ["CoupledModes"]
