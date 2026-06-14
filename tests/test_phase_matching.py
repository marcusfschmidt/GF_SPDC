from __future__ import annotations

import numpy as np

from gf_spdc.mgo_lithium_niobate_type0_beta import MgOLithiumNiobateType0
from gf_spdc.mgo_lithium_niobate_type2_beta import MgOLithiumNiobateType2


def test_type0_sets_qpm_and_indistinguishable_flags() -> None:
    model = MgOLithiumNiobateType0(1064e-9, 1064e-9, 532e-9, True, 36, 5.9e-6)
    assert model.QPMbool is True
    assert model.indistinguishableBool is True
    assert model.kp.shape == model.ks.shape == model.ki.shape == model.om.shape


def test_mgo_lithium_niobate_type2_sets_distinguishable_flags() -> None:
    c = 299792458e-12
    lambda_p = 532e-9
    lambda_s = 1064e-9
    om_p = 2 * np.pi * c / lambda_p
    om_s = 2 * np.pi * c / lambda_s
    om_i = om_p - om_s
    lambda_i = 2 * np.pi * c / om_i

    model = MgOLithiumNiobateType2(lambda_s, lambda_i, lambda_p)
    assert model.QPMbool is False
    assert model.indistinguishableBool is False
    assert model.kp.shape == model.ks.shape == model.ki.shape == model.om.shape
