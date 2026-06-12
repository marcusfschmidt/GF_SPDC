from gf_spdc.type0_beta import Type0


class type0(Type0):
    def __init__(
        self,
        lambda_s: float,
        lambda_i: float,
        lambda_p: float,
        ordinaryAxisBool: bool,
        temperature: float,
        QPMPeriod: float,
    ) -> None:
        super().__init__(lambda_s, lambda_i, lambda_p, ordinaryAxisBool, temperature, QPMPeriod)

__all__ = ["type0"]
