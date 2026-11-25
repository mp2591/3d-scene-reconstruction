from pathlib import Path

import numpy as np
import pytest

from scene_recon.comparison import Summary
from scene_recon.report import (
    SOFTWARE_DIFFERENT,
    GroupResult,
    RunResult,
    _best_value,
    _compare_software,
    _comparison_section,
    _differences_section,
    _format_count,
    _format_duration,
    _format_summary,
    _mean_curves,
    _paired_comparisons,
    _read_software,
    _short_hash,
)
from scene_recon.run_files import RunFiles


def _run(key: str, metrics: dict, seed: int = 0) -> RunResult:
    return RunResult(key=key, seed=seed, status="evaluated", files=RunFiles(Path(key)), metrics=metrics, eligible=True)


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
            run.eligible = True
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


# ---------------------------------------------------------------- starting weights (review B(b))


def _started_from(group: GroupResult, sha256: str) -> GroupResult:
    group.replicates[0].starting_weights = {"encoder": {"source": "weights/hiera.pth", "sha256": sha256}}
    return group


def test_the_report_warns_when_runs_started_from_different_versions_of_the_same_weights():
    runs = [_started_from(_group("hiera_mlp", {}), "1" * 64), _started_from(_group("hiera_nerf", {}), "2" * 64)]

    text = "\n".join(_differences_section(runs, {}, multiple_seeds=False))

    assert "runs that start from `weights/hiera.pth` started from different weights" in text
    assert "| pretrained weights recorded in the checkpoint | `encoder: hiera.pth 111111111111` |" in text


def test_the_report_lists_the_pretrained_weights_without_a_warning_when_they_agree():
    runs = [_started_from(_group("hiera_mlp", {}), "1" * 64), _started_from(_group("hiera_nerf", {}), "1" * 64)]
    differences = {"model.name": {"hiera_mlp": "pixel_aligned_nerf", "hiera_nerf": "hiera_nerf"}}

    text = "\n".join(_differences_section(runs, differences, multiple_seeds=False))

    assert "started from different weights" not in text
    assert "| pretrained weights recorded in the checkpoint | `encoder: hiera.pth 111111111111` | `encoder: hiera.pth" in text


def test_the_pretrained_weights_are_listed_even_when_every_setting_is_identical():
    """Review F47."""
    runs = [_started_from(_group("first", {}), "1" * 64), _started_from(_group("second", {}), "1" * 64)]

    text = "\n".join(_differences_section(runs, {}, multiple_seeds=False))

    assert "Nothing: all runs use identical settings." in text
    assert "| pretrained weights recorded in the checkpoint | `encoder: hiera.pth 111111111111` |" in text


def test_seeds_of_one_run_that_started_from_different_weights_are_labelled():
    """Review F47."""
    run = _group("hiera_mlp", {}, {})
    for replicate, sha256 in zip(run.replicates, ["1" * 64, "2" * 64]):
        replicate.starting_weights = {"encoder": {"source": "weights/hiera.pth", "sha256": sha256}}

    text = "\n".join(_differences_section([run], {}, multiple_seeds=True))

    assert "started from different weights" in text
    assert "seed 0: `encoder: hiera.pth 111111111111`; seed 1: `encoder: hiera.pth 222222222222`" in text


# ---------------------------------------------------------------- evaluation software (review F46)


def _software_record(**packages: str) -> dict:
    """A software record as ``environment.evaluation_software`` saves it."""
    return {
        "python": "3.10.6",
        "cuda": "13.0",
        "device_type": "cuda",
        "device": "NVIDIA GeForce RTX 4090",
        "packages": {"torch": "2.14.1", **packages},
    }


def _evaluated_with(name: str, lpips: str | None, **packages: str) -> GroupResult:
    """An evaluated run whose software record has this lpips version (None: no record)."""
    group = _group(name, {1: {"psnr": 20.0}})
    run = group.replicates[0]
    record = _software_record(lpips=lpips, **packages) if lpips else None
    run.evaluation_software, run.software_problems = _read_software(record)
    return group


def _evaluations(groups: list[GroupResult]) -> list[RunResult]:
    return [run for group in groups for run in group.replicates]


def test_the_report_lists_what_differs_between_the_runs_software():
    runs = [_evaluated_with("first", "0.1.4"), _evaluated_with("second", "0.2.0")]

    text = "\n".join(_differences_section(runs, {}, multiple_seeds=False))

    assert "the runs were evaluated with different software" in text
    assert "| evaluation software that differs | lpips 0.1.4 | lpips 0.2.0 |" in text  # only what differs


@pytest.mark.parametrize("name", ["device", "python", "cuda", "device-type"])
def test_a_package_named_like_a_record_field_is_compared_as_a_package(name: str):
    """Review F74: a package called ``device`` neither hides nor replaces the device."""
    runs = [_evaluated_with("first", "0.1.4"), _evaluated_with("second", "0.1.4", **{name: "0.1"})]
    first, second = _evaluations(runs)

    text = "\n".join(_differences_section(runs, {}, multiple_seeds=False))

    assert _compare_software([first, second]) == SOFTWARE_DIFFERENT
    assert _short_hash(first.evaluation_software) != _short_hash(second.evaluation_software)
    assert f"| evaluation software that differs | {name} absent | {name} 0.1 |" in text
    assert second.evaluation_software["device"] == "NVIDIA GeForce RTX 4090"


@pytest.mark.parametrize(
    "change",
    [
        lambda record: record["packages"].update({"": "1"}),  # review F78
        lambda record: record["packages"].update({"Torch": "1"}),  # not a normalised name
        lambda record: record["packages"].update(torch=""),
        lambda record: record["packages"].update(torch=["2.8"]),
        lambda record: record["packages"].update(torch="\n"),  # review F83
        lambda record: record["packages"].update(torch="2.8 beta"),
        lambda record: record.update(packages={}),
        lambda record: record.update(device=""),
        lambda record: record.update(python=3.1),
        lambda record: record.pop("device_type"),  # also a record from before the device type was saved
        lambda record: record.update(extra="1"),
    ],
    ids=[
        "empty_name",
        "unnormalised_name",
        "empty_version",
        "list_version",
        "blank_version",
        "spaced_version",
        "no_packages",
        "empty_device",
        "number",
        "missing",
        "extra",
    ],
)
def test_a_damaged_software_record_cannot_be_compared(change):
    record = _software_record(lpips="0.1.4")
    change(record)

    assert _read_software(record) == (None, ["damaged"])


def test_the_report_says_why_a_software_record_cannot_be_compared():
    """Review F76, F77 and F88: an unnamed device or an unreadable package list is unknown, never the same."""
    unreadable = {**_software_record(), "device": None, "packages": None}
    both = ["device model unknown", "package list unreadable"]
    assert _read_software(None) == (None, ["not recorded"])
    assert _read_software({**_software_record(), "device": None}) == (None, ["device model unknown"])
    assert _read_software({**_software_record(), "packages": None}) == (None, ["package list unreadable"])
    assert _read_software(unreadable) == (None, both)
    assert _read_software({**unreadable, "python": ""}) == (None, [*both, "damaged"])
    assert _read_software(_software_record()) == (_software_record(), [])

    runs = [_evaluated_with("first", "0.1.4"), _group("second", {1: {"psnr": 20.0}})]
    runs[1].replicates[0].software_problems = both
    text = "\n".join(_differences_section(runs, {}, multiple_seeds=False))

    assert "*device model unknown*: the model of the device it ran on cannot be named" in text
    assert "*package list unreadable*: the metadata of an installed package could not be read" in text
    # One readable record: nothing can differ from it.
    assert "| evaluation software that differs | – | device model unknown, package list unreadable |" in text


def test_a_results_file_with_a_key_given_twice_is_treated_as_damaged(tmp_path: Path):
    """Review F84: a JSON reader would keep only the last value, so the file is redone rather than trusted."""
    from scene_recon.run_files import read_json

    twice = tmp_path / "evaluation.json"
    twice.write_text('{"software": {"device": "A100", "device": "RTX 4090"}}', encoding="utf-8")
    once = tmp_path / "once.json"
    once.write_text('{"software": {"device": "RTX 4090"}}', encoding="utf-8")

    assert read_json(twice) is None
    assert read_json(once) == {"software": {"device": "RTX 4090"}}


def test_a_single_evaluation_whose_software_cannot_be_compared_is_flagged():
    """Review F79: also while the other runs have no results yet."""
    runs = [_evaluated_with("first", None), _group("second", {})]

    text = "\n".join(_differences_section(runs, {}, multiple_seeds=False))

    assert "the software of some evaluations cannot be compared" in text and "*not recorded*" in text


def test_runs_evaluated_with_the_same_software_get_no_software_row():
    runs = [_evaluated_with("first", "0.1.4"), _evaluated_with("second", "0.1.4")]

    text = "\n".join(_differences_section(runs, {}, multiple_seeds=False))

    assert "Nothing: all runs use identical settings." in text and "software" not in text


def test_a_missing_software_record_is_never_taken_for_the_same_software():
    """Review F65."""
    runs = [_evaluated_with("first", None), _evaluated_with("second", None)]

    text = "\n".join(_differences_section(runs, {}, multiple_seeds=False))

    assert "the software of some evaluations cannot be compared" in text
    assert "| evaluation software that differs | not recorded | not recorded |" in text
