from __future__ import annotations

import os
import re

import numpy as np
import pytest

from gf_spdc.stitcher import GreenFunctionStitcher
from scripts.run_stitcher import build_default_params, run_stitcher_from_params


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


@pytest.mark.skipif(
    os.environ.get("GF_SPDC_RUN_GOLDEN") != "1",
    reason="Set GF_SPDC_RUN_GOLDEN=1 to run the slow stitcher golden test.",
)
def test_default_stitcher_run_matches_golden_behavior(capsys: pytest.CaptureFixture[str]) -> None:
    params = build_default_params()

    green_functions, time_width_array, freq_width_array, stitch_times, saved_name = run_stitcher_from_params(
        params,
        basis_width=0.5,
        kmax=50,
        save=False,
    )
    captured = capsys.readouterr().out

    assert saved_name is None
    assert stitch_times == []
    assert len(green_functions) == 4
    assert len(time_width_array) == 2
    assert len(freq_width_array) == 2
    for green_function in green_functions:
        assert green_function.ndim == 2
        assert green_function.shape[0] == green_function.shape[1]
        assert green_function.shape[0] > 0

    assert "Initial extraction: building Green's functions..." in captured
    assert captured.count("Extraction timing:") == 2
    assert "Run extractor:" in captured
    assert "Initial extraction complete. Starting iterative stitching (pass 1)..." in captured
    assert "Iteratively stitching Green's functions (pass 1)..." in captured
    assert "Starting iterative stitching (pass 2)..." in captured
    assert "Iteratively stitching Green's functions (pass 2)..." in captured
    assert captured.count("| done (edge overlap > 99.9%)") == 2

    extraction_timings = re.findall(
        r"Extraction timing: task setup ([0-9.]+)s, workers ([0-9.]+)s, basis ([0-9.]+)s, projection/SVD ([0-9.]+)s, total\s+([0-9.]+)s",
        captured,
    )
    assert len(extraction_timings) == 2
    for task_setup_s, worker_s, basis_s, projection_s, total_s in extraction_timings:
        task_setup = float(task_setup_s)
        worker = float(worker_s)
        basis = float(basis_s)
        projection = float(projection_s)
        total = float(total_s)

        assert task_setup >= 0.0
        assert worker > 0.0
        assert basis >= 0.0
        assert projection >= 0.0
        assert total >= worker
        assert worker > basis
        assert worker > projection
        assert total < 180.0

    run_match = re.search(
        r"Run extractor: signal ([0-9.]+)s, idler ([0-9.]+)s, green ([0-9.]+)s, assemble ([0-9.]+)s, total ([0-9.]+)s",
        captured,
    )
    assert run_match is not None
    signal, idler, green, assemble, total = (float(value) for value in run_match.groups())
    assert signal > 0.0
    assert idler > 0.0
    assert green > 0.0
    assert assemble > 0.0
    assert total >= max(signal, idler, green, assemble)
    assert total < 180.0
