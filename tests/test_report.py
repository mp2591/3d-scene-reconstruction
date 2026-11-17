from pathlib import Path

from scene_recon.report import RunResult, _best_value, _format_count, _format_duration
from scene_recon.run_files import RunFiles


def _result(name: str, metrics: dict) -> RunResult:
    return RunResult(name=name, status="evaluated", files=RunFiles(Path(name)), metrics=metrics)


def test_best_value_respects_whether_higher_is_better():
    results = [
        _result("a", {1: {"psnr": 20.0, "lpips": 0.30}}),
        _result("b", {1: {"psnr": 22.0, "lpips": 0.20}}),
    ]

    assert _best_value(results, 1, "psnr") == 22.0
    assert _best_value(results, 1, "lpips") == 0.20


def test_no_best_value_with_a_single_run_or_an_unknown_metric():
    single = [_result("a", {1: {"psnr": 20.0}})]
    unknown = [_result("a", {1: {"sharpness": 1.0}}), _result("b", {1: {"sharpness": 2.0}})]

    assert _best_value(single, 1, "psnr") is None
    assert _best_value(unknown, 1, "sharpness") is None


def test_durations_and_parameter_counts_are_formatted_for_reading():
    assert _format_duration(1.5) == "1:30"
    assert _format_duration(26.02) == "26:01"
    assert _format_duration(None) == "–"
    assert _format_count(4_884) == "4,884"
    assert _format_count(21_300_000) == "21.30 M"
