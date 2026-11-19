import math

import pytest

from scene_recon.comparison import average_over_seeds, paired_difference, summarize


def test_summary_uses_the_sample_standard_deviation():
    summary = summarize([20.0, 22.0, 24.0])

    assert summary.mean == pytest.approx(22.0)
    assert summary.std == pytest.approx(2.0)
    assert summary.count == 3


def test_summary_of_a_single_seed_has_no_spread():
    assert summarize([21.5]).std == 0.0
    assert summarize([]) is None


def test_seed_average_keeps_only_scenes_every_seed_has():
    averaged = average_over_seeds([{"a": 1.0, "b": 3.0}, {"a": 3.0}])

    assert averaged == {"a": 2.0}


def test_paired_difference_matches_the_student_t_interval():
    reference = {"s1": 20.0, "s2": 21.0, "s3": 22.0, "s4": 23.0}
    candidate = {"s1": 20.5, "s2": 21.7, "s3": 22.3, "s4": 23.5}  # differences 0.5, 0.7, 0.3, 0.5

    difference = paired_difference(candidate, reference)

    standard_error = 0.16329931618554522 / math.sqrt(4)  # sample std of the differences / sqrt(n)
    half_width = 3.182446305284263 * standard_error  # t quantile for 97.5% with 3 degrees of freedom
    assert difference.mean == pytest.approx(0.5)
    assert difference.low == pytest.approx(0.5 - half_width)
    assert difference.high == pytest.approx(0.5 + half_width)
    assert difference.num_scenes == 4


def test_verdict_depends_on_the_interval_and_on_the_metric_direction():
    reference = {f"s{index}": 20.0 + index for index in range(10)}
    clearly_higher = {scene: value + 1.0 + 0.01 * index for index, (scene, value) in enumerate(reference.items())}
    noisy = {scene: value + (1.0 if index % 2 else -1.0) for index, (scene, value) in enumerate(reference.items())}

    assert paired_difference(clearly_higher, reference).verdict(higher_is_better=True) == "better"
    assert paired_difference(clearly_higher, reference).verdict(higher_is_better=False) == "worse"
    assert paired_difference(noisy, reference).verdict(higher_is_better=True) == "no clear difference"


def test_paired_difference_needs_two_shared_scenes():
    assert paired_difference({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0}) is None
