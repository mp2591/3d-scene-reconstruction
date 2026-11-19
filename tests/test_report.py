from pathlib import Path

import numpy as np
import pytest

from scene_recon.comparison import Summary
from scene_recon.report import (
    GroupResult,
    RunResult,
    _best_value,
    _format_count,
    _format_duration,
    _format_summary,
    _mean_curves,
)
from scene_recon.run_files import RunFiles


def _run(key: str, metrics: dict, seed: int = 0) -> RunResult:
    return RunResult(key=key, seed=seed, status="evaluated", files=RunFiles(Path(key)), metrics=metrics)


def _group(name: str, *per_seed_metrics: dict) -> GroupResult:
    """A run with one result per seed."""
    replicates = [_run(f"{name}/seed_{seed}", metrics, seed) for seed, metrics in enumerate(per_seed_metrics)]
    return GroupResult(name, replicates)


def test_best_value_respects_whether_higher_is_better():
    results = [
        _group("a", {1: {"psnr": 20.0, "lpips": 0.30}}),
        _group("b", {1: {"psnr": 22.0, "lpips": 0.20}}),
    ]

    assert _best_value(results, 1, "psnr") == 22.0
    assert _best_value(results, 1, "lpips") == 0.20


def test_best_value_compares_means_over_seeds():
    results = [
        _group("steady", {1: {"psnr": 21.0}}, {1: {"psnr": 21.0}}),
        _group("lucky_once", {1: {"psnr": 23.0}}, {1: {"psnr": 18.0}}),
    ]

    assert _best_value(results, 1, "psnr") == 21.0


def test_no_best_value_with_a_single_run_or_an_unknown_metric():
    single = [_group("a", {1: {"psnr": 20.0}})]
    unknown = [_group("a", {1: {"sharpness": 1.0}}), _group("b", {1: {"sharpness": 2.0}})]

    assert _best_value(single, 1, "psnr") is None
    assert _best_value(unknown, 1, "sharpness") is None


def test_scene_scores_are_averaged_over_seeds():
    group = GroupResult("a", [_run("a/seed_0", {1: {}}, 0), _run("a/seed_1", {1: {}}, 1)])
    group.replicates[0].per_scene = {1: {"car_1": {"psnr": 20.0}, "car_2": {"psnr": 24.0}}}
    group.replicates[1].per_scene = {1: {"car_1": {"psnr": 22.0}}}  # car_2 missing for this seed

    assert group.scene_scores(1, "psnr") == {"car_1": 21.0}


def test_quality_cells_show_spread_only_with_several_seeds():
    assert _format_summary(Summary(mean=21.234, std=0.0, count=1), "psnr", num_seeds=1) == "21.23"
    assert _format_summary(Summary(mean=21.234, std=0.31, count=3), "psnr", num_seeds=3) == "21.23 ± 0.31"
    assert _format_summary(Summary(mean=0.9, std=0.01, count=2), "ssim", num_seeds=3) == "0.900 ± 0.010 (n=2)"


def test_training_curves_are_averaged_at_shared_steps():
    seed_0 = {"loss": (np.array([10, 20, 30]), np.array([1.0, 0.5, 0.25]))}
    seed_1 = {"loss": (np.array([10, 20]), np.array([3.0, 1.5]))}  # stopped earlier

    steps, values = _mean_curves([seed_0, seed_1])["loss"]

    assert steps.tolist() == [10, 20]
    assert values.tolist() == pytest.approx([2.0, 1.0])


def test_durations_and_parameter_counts_are_formatted_for_reading():
    assert _format_duration(1.5) == "1:30"
    assert _format_duration(26.02) == "26:01"
    assert _format_duration(None) == "–"
    assert _format_count(4_884) == "4,884"
    assert _format_count(21_300_000) == "21.30 M"
