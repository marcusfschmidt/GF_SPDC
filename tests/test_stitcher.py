from __future__ import annotations

import numpy as np

from gf_spdc.stitcher import GreenFunctionStitcher


def make_stub_stitcher() -> GreenFunctionStitcher:
    stitcher = GreenFunctionStitcher.__new__(GreenFunctionStitcher)
    stitcher.t = np.array([-1.0, 0.0, 1.0])
    stitcher.indistinguishable_bool = True
    return stitcher


def test_width_helper_function_falls_back_for_uniform_array() -> None:
    stitcher = make_stub_stitcher()
    widths = stitcher.width_helper_function(np.ones((4, 5), dtype=float))
    assert widths == (0, 4, 0, 3)


def test_stitch_helper_function_selects_columns_by_input_time() -> None:
    stitcher = make_stub_stitcher()
    g1 = np.ones((3, 3), dtype=complex)
    g2 = 2 * np.ones((3, 3), dtype=complex)
    stitched_g, stitched_f = stitcher.stitch_helper_function(g1, g1, g2, g2, -1.0, 1.0)

    np.testing.assert_array_equal(stitched_g[:, :2], g2[:, :2])
    np.testing.assert_array_equal(stitched_g[:, 2], g1[:, 2])
    np.testing.assert_array_equal(stitched_f, stitched_g)
