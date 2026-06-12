from gf_spdc.type2_beta import Type2


class typeII(Type2):
    def __init__(self, lambda_s: float, lambda_i: float, lambda_p: float) -> None:
        super().__init__(lambda_s, lambda_i, lambda_p)

__all__ = ["typeII"]
