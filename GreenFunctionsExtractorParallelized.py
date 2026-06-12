from gf_spdc.extractor import GreenFunctionsExtractor as _GreenFunctionsExtractor


class GreenFunctionsExtractor(_GreenFunctionsExtractor):
    def __init__(self, kmax: int, debugBool: bool = True) -> None:
        super().__init__(kmax, debugBool)

__all__ = ["GreenFunctionsExtractor"]
