from pathlib import Path

import numpy as np
import pytest

from scene_recon.comparison import Summary
from scene_recon.report import (
    GroupResult,
    RunResult,
    _best_value,
    _comparison_section,
    _format_count,
    _format_duration,
    _format_summary,
    _mean_curves,
    _paired_comparisons,
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

    assert group.scene_scores(1, "psnr", seeds={0, 1}) == {"car_1": 21.0}
    assert group.scene_scores(1, "psnr", seeds={0}) == {"car_1": 20.0, "car_2": 24.0}


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



# ---------------------------------------------------------------- paired comparisons (R3)


def _seeded_group(name: str, scores_per_seed: dict[int, float | None], seeds: list[int]) -> GroupResult:
    """A run over ``seeds``: ten scenes with score ``base + offset`` for each finished seed (None = not finished)."""
    replicates = []
    for seed in seeds:
        base = scores_per_seed.get(seed)
        run = RunResult(key=f"{name}/seed_{seed}", seed=seed, status="trained", files=RunFiles(Path(name)))
        if base is not None:
            per_scene = {f"scene_{index}": {"psnr": base + 0.37 * index} for index in range(10)}
            run.per_scene = {1: per_scene}
            run.metrics = {1: {"psnr": base + 0.37 * 4.5}}
            run.status = "evaluated"
        replicates.append(run)
    return GroupResult(name, replicates)


def test_an_identical_model_with_an_unfinished_seed_gets_no_verdict():
    """The review's example: identical scores on the shared seed, the candidate's second seed not done."""
    reference = _seeded_group("reference", {0: 20.0, 1: 40.0}, seeds=[0, 1])
    candidate = _seeded_group("candidate", {0: 20.0, 1: None}, seeds=[0, 1])

    (comparison,) = _paired_comparisons([reference, candidate])

    assert comparison.difference.mean == pytest.approx(0.0)
    assert comparison.difference.verdict(higher_is_better=True) == "no clear difference"
    assert comparison.seeds == [0] and comparison.partial


def test_a_seed_finished_on_one_side_only_does_not_change_the_comparison():
    reference = _seeded_group("reference", {0: 20.0, 1: None}, seeds=[0, 1])
    candidate_before = _seeded_group("candidate", {0: 21.0, 1: None}, seeds=[0, 1])
    candidate_after = _seeded_group("candidate", {0: 21.0, 1: 35.0}, seeds=[0, 1])

    (before,) = _paired_comparisons([reference, candidate_before])
    (after,) = _paired_comparisons([reference, candidate_after])

    assert after.difference == before.difference and after.seeds == [0]


def test_complete_seed_sets_are_not_marked_partial():
    reference = _seeded_group("reference", {0: 20.0, 1: 22.0}, seeds=[0, 1])
    candidate = _seeded_group("candidate", {0: 21.0, 1: 23.0}, seeds=[0, 1])

    (comparison,) = _paired_comparisons([reference, candidate])

    assert comparison.seeds == [0, 1] and not comparison.partial
    assert comparison.difference.mean == pytest.approx(1.0)


def test_the_first_run_stays_the_reference_even_without_results():
    reference = _seeded_group("baseline", {0: None}, seeds=[0])
    others = [_seeded_group("second", {0: 21.0}, seeds=[0]), _seeded_group("third", {0: 22.0}, seeds=[0])]
    results = [reference, *others]

    assert _paired_comparisons(results) == []
    section = "\n".join(_comparison_section(results, [], ["psnr"], multiple_seeds=False))
    assert "Paired comparison with baseline" in section and "Not available yet" in section
