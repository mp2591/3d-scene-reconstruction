"""The learning-rate pilot: choosing each encoder's rate, and the comparison continuing the chosen run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from scene_recon import environment
from scene_recon import suite as suite_module
from scene_recon.checkpoints import read_progress
from scene_recon.pilot import (
    CHOICE_FILE,
    TOLERANCE_DB,
    Choice,
    TunedRun,
    _reads_same_scenes,
    _same_data,
    _yaml_float,
    check_pilot,
    choose_and_continue,
    pair_pilot_with_comparison,
    write_rates_into_suite,
)
from scene_recon.registry import ConfigError
from scene_recon.run_files import RunFiles, read_json, write_json
from scene_recon.suite import load_suite, run_suite

from conftest import CONFIG_DIR

PILOT_STEPS = 2
COMPARISON_STEPS = 4

COMPARISON = """\
suite_name: comparison
stages: [train, evaluate]
seeds: [0]
common:
{common}
runs:
  # the run whose encoder rate the pilot chooses
  tiny:
    model:
      decoder: {{hidden_width: 16}}
    optimization:
      group_overrides:
        encoder: {{learning_rate: 1.0e-4, warmup_steps: 2}}   # placeholder until the pilot has run
  # a run the pilot does not tune
  untuned:
    model:
      decoder: {{hidden_width: 24}}
"""


@pytest.fixture(autouse=True)
def _named_cpu(monkeypatch):
    """The CPU these tests evaluate on has a model name, as the chooser requires; ARM Linux reports none."""
    monkeypatch.setattr(environment, "cpu_model", lambda: "Test CPU")


def _common(tmp_path: Path, num_steps: int, test_split: dict) -> dict:
    """Settings of both suites: the smoke test, shrunk to a few steps."""
    return {
        "base": str(CONFIG_DIR / "smoke_test.yaml"),
        "experiment": {"output_dir": str(tmp_path / "runs"), "device": "cpu"},
        "data": {"train": {"image_size": 16}, "val": {"image_size": 16}, "test": {"image_size": 16, **test_split}},
        "training": {
            "num_steps": num_steps,
            "batch_size": 2,
            "rays_per_scene": 32,
            "log_every": 2,
            "validate_every": 2,
            "checkpoint_every": 2,
        },
        "validation": {"num_scenes": 1, "num_target_views": 1},
        "evaluation": {"num_scenes": 2, "num_target_views": 2, "num_preview_scenes": 0},
    }


def _write_comparison(tmp_path: Path) -> Path:
    common = yaml.safe_dump(_common(tmp_path, COMPARISON_STEPS, {}), sort_keys=False)
    path = tmp_path / "comparison.yaml"
    path.write_text(COMPARISON.format(common="\n".join("  " + line for line in common.splitlines())), encoding="utf-8")
    return path


def _write_pilot(tmp_path: Path, rates=(1.0e-4, 3.0e-4), run_change: dict | None = None, **common_changes) -> Path:
    common = _common(tmp_path, PILOT_STEPS, {"seed": 1})  # scored on the validation split (seed 1)
    common.update(common_changes)
    runs = {
        f"tiny_lr{rate:g}": {
            "model": {"decoder": {"hidden_width": 16}},
            "optimization": {"group_overrides": {"encoder": {"learning_rate": rate, "warmup_steps": 2}}},
            **(run_change or {}),
        }
        for rate in rates
    }
    content = {"suite_name": "pilot", "stages": ["train", "evaluate"], "common": common, "runs": runs}
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def _set_scores(pilot_path: Path, psnr_by_rate: dict[float, float]) -> None:
    """Give each pilot run's evaluation the given PSNR in every protocol; the file stays current."""
    for config in load_suite(pilot_path).runs.values():
        rate = config.optimization.group_overrides["encoder"]["learning_rate"]
        path = RunFiles(config.run_dir).evaluation
        content = json.loads(path.read_text(encoding="utf-8"))
        for protocol in content["protocols"]:
            protocol["mean"]["psnr"] = psnr_by_rate[rate]
        write_json(content, path)


def _encoder_rate(comparison_path: Path) -> float:
    return load_suite(comparison_path).runs["tiny/seed_0"].optimization.group_overrides["encoder"]["learning_rate"]


@pytest.mark.parametrize(
    "scores, chosen",
    [
        ({1.0e-4: 20.0, 3.0e-4: 20.0 + TOLERANCE_DB / 2}, 1.0e-4),  # about as good: the lower rate
        ({1.0e-4: 20.0, 3.0e-4: 20.0 + TOLERANCE_DB * 2}, 3.0e-4),  # clearly better: the better rate
    ],
)
def test_the_lowest_rate_within_the_tolerance_is_chosen_and_continued(tmp_path: Path, monkeypatch, scores, chosen):
    comparison_path, pilot_path = _write_comparison(tmp_path), _write_pilot(tmp_path)
    assert run_suite(load_suite(pilot_path)) == {}
    _set_scores(pilot_path, scores)
    text_before = comparison_path.read_text(encoding="utf-8")

    choose_and_continue(pilot_path, comparison_path)

    # The rate is written into the suite file, which keeps its comments and other runs.
    assert _encoder_rate(comparison_path) == chosen
    text_after = comparison_path.read_text(encoding="utf-8")
    assert text_after.replace(f"learning_rate: {_yaml_float(chosen)}", "learning_rate: 1.0e-4") == text_before
    # The choice is saved with the scores it was made from.
    pilot = load_suite(pilot_path)
    saved = read_json(pilot.directory / CHOICE_FILE)["runs"]["tiny"]
    assert saved["chosen_rate"] == chosen and saved["chosen_pilot_run"] == f"tiny_lr{chosen:g}"
    assert saved["continued_from"]["pilot_run"] == f"tiny_lr{chosen:g}"
    assert saved["continued_from"]["step"] == PILOT_STEPS
    assert {candidate["rate"]: candidate["psnr"] for candidate in saved["candidates"]} == scores

    # The comparison continues the chosen pilot run instead of starting again.
    comparison = load_suite(comparison_path)
    winner = RunFiles(pilot.runs[f"tiny_lr{chosen:g}"].run_dir)
    continued = RunFiles(comparison.runs["tiny/seed_0"].run_dir)
    assert continued.checkpoint.read_bytes() == winner.checkpoint.read_bytes()
    resumed = []
    real_training = suite_module.run_training

    def recording_training(config, resume=False):
        resumed.append((config.experiment.run_name, resume))
        real_training(config, resume)

    monkeypatch.setattr(suite_module, "run_training", recording_training)
    assert run_suite(comparison) == {}
    assert resumed == [("tiny/seed_0", True), ("untuned/seed_0", False)]
    assert read_progress(continued.checkpoint).step == COMPARISON_STEPS

    # Choosing again changes nothing: the continued run keeps its checkpoint, and the record of the copy stays.
    trained = continued.checkpoint.read_bytes()
    choose_and_continue(pilot_path, comparison_path)
    assert continued.checkpoint.read_bytes() == trained
    assert comparison_path.read_text(encoding="utf-8") == text_after
    assert read_json(pilot.directory / CHOICE_FILE)["runs"]["tiny"]["continued_from"] == saved["continued_from"]


def test_no_rate_is_chosen_before_every_pilot_run_is_scored(tmp_path: Path):
    comparison_path, pilot_path = _write_comparison(tmp_path), _write_pilot(tmp_path)
    pilot = load_suite(pilot_path)
    run_suite(pilot)
    RunFiles(pilot.runs["tiny_lr0.0003"].run_dir).evaluation.unlink()
    text_before = comparison_path.read_text(encoding="utf-8")

    with pytest.raises(ConfigError, match="(?s)cannot be chosen yet.*tiny_lr0.0003' has no evaluation"):
        choose_and_continue(pilot_path, comparison_path)

    assert comparison_path.read_text(encoding="utf-8") == text_before


def test_a_pilot_run_trained_past_its_budget_or_with_other_settings_is_not_scored(tmp_path: Path):
    """Review F120-F121: a score counts only for the steps and settings its run declares."""
    comparison_path, pilot_path = _write_comparison(tmp_path), _write_pilot(tmp_path)
    pilot = load_suite(pilot_path)
    run_suite(pilot)
    # One run trained further than the pilot's budget, e.g. by hand.
    run_suite(load_suite(pilot_path, overrides=[f"training.num_steps={PILOT_STEPS + 2}"]))

    with pytest.raises(ConfigError, match=f"has trained {PILOT_STEPS + 2} steps, not its {PILOT_STEPS}"):
        choose_and_continue(pilot_path, comparison_path)


def test_a_comparison_run_trained_with_another_rate_is_not_replaced(tmp_path: Path):
    comparison_path, pilot_path = _write_comparison(tmp_path), _write_pilot(tmp_path)
    run_suite(load_suite(comparison_path))  # trained with the placeholder rate 1e-4
    run_suite(load_suite(pilot_path))
    _set_scores(pilot_path, {1.0e-4: 20.0, 3.0e-4: 25.0})

    with pytest.raises(ConfigError, match="trained with different settings"):
        choose_and_continue(pilot_path, comparison_path)


def test_a_run_saves_the_settings_of_its_latest_evaluation(tmp_path: Path):
    """Review F146: scored on fewer test scenes first (the launcher's step 4), then on all of them (step 5),
    a run's config.yaml repeats the second evaluation, not the first."""
    comparison_path = _write_comparison(tmp_path)
    run_suite(load_suite(comparison_path, overrides=["evaluation.num_scenes=1"]))
    run_suite(load_suite(comparison_path))

    for config in load_suite(comparison_path).runs.values():
        saved = yaml.safe_load(RunFiles(config.run_dir).config.read_text(encoding="utf-8"))
        assert saved["evaluation"]["num_scenes"] == config.evaluation.num_scenes == 2


def test_a_comparison_run_trained_past_its_budget_is_not_kept(tmp_path: Path):
    """Review F140: a run kept as finished would be compared after more steps than the others."""
    comparison_path, pilot_path = _write_comparison(tmp_path), _write_pilot(tmp_path)
    # Trained with the rate that will be chosen, but further than the comparison asks.
    run_suite(load_suite(comparison_path, overrides=[f"training.num_steps={COMPARISON_STEPS + 2}"]))
    run_suite(load_suite(pilot_path))
    _set_scores(pilot_path, {1.0e-4: 25.0, 3.0e-4: 20.0})

    with pytest.raises(ConfigError, match=f"has trained {COMPARISON_STEPS + 2} steps, more than the {COMPARISON_STEPS}"):
        choose_and_continue(pilot_path, comparison_path)


def test_scores_evaluated_with_unrecorded_software_are_not_compared(tmp_path: Path):
    """Review F137: two records that both say "unknown" do not show the same software."""
    comparison_path, pilot_path = _write_comparison(tmp_path), _write_pilot(tmp_path)
    pilot = load_suite(pilot_path)
    run_suite(pilot)
    for config in pilot.runs.values():
        path = RunFiles(config.run_dir).evaluation
        content = read_json(path)
        content["software"]["packages"] = None
        write_json(content, path)

    with pytest.raises(ConfigError, match="software of its evaluation cannot be compared"):
        choose_and_continue(pilot_path, comparison_path)


@pytest.mark.parametrize(
    "pilot_change, problem",
    [
        # scored on the comparison's test split (seed 2), not on its validation split
        ({"data": {"train": {"image_size": 16}, "val": {"image_size": 16}, "test": {"image_size": 16}}},
         "not on the comparison's validation split"),
        # trains another model than any comparison run
        ({"run_change": {"model": {"decoder": {"hidden_width": 32}}}}, "trains like 0 runs of the comparison"),
        # another learning-rate schedule: its runs would not be the beginning of the comparison's
        ({"optimization": {"decay_steps": 600}}, "trains like 0 runs of the comparison"),
        # longer than the comparison that would continue it
        ({"training": {**_common(Path("."), COMPARISON_STEPS + 2, {})["training"]}}, "more than the 4 of"),
        # a single rate is nothing to choose from
        ({"rates": (1.0e-4,)}, "needs at least two different rates"),
        # trained on the comparison's test scenes (review F148)
        ({"data": {"train": {"image_size": 16, "num_scenes": 4, "seed": 2}, "val": {"image_size": 16},
                   "test": {"image_size": 16, "seed": 1}}},
         "reads the comparison's test scenes in its data.train"),
        # validated during training on the comparison's test scenes (seed 2)
        ({"data": {"train": {"image_size": 16}, "val": {"image_size": 16, "seed": 2}, "test": {"image_size": 16, "seed": 1}}},
         "reads the comparison's test scenes in its data.val"),
        # a rate YAML reads as text (review F139): the copied checkpoint would not match the comparison
        ({"run_change": {"optimization": {"group_overrides": {"encoder": {"learning_rate": "1e-4", "warmup_steps": 2}}}}},
         "as text"),
    ],
)
def test_a_pilot_that_does_not_fit_the_comparison_is_refused_before_it_runs(tmp_path: Path, pilot_change, problem):
    comparison_path = _write_comparison(tmp_path)
    pilot_path = _write_pilot(tmp_path, **pilot_change)

    with pytest.raises(ConfigError, match=problem):
        pair_pilot_with_comparison(load_suite(pilot_path), load_suite(comparison_path))


def test_the_check_before_the_pilot_finds_what_would_stop_the_choice_after_it(tmp_path: Path):
    """Review F126: problems that would only show after hours of pilot training are found before it."""
    comparison_path = _write_comparison(tmp_path)
    # A comparison file whose rate line cannot be found, and a pilot that does not score PSNR.
    comparison_path.write_text(
        comparison_path.read_text(encoding="utf-8").replace(
            "        encoder: {learning_rate: 1.0e-4, warmup_steps: 2}   # placeholder until the pilot has run\n",
            "        encoder:\n          learning_rate: 1.0e-4\n          warmup_steps: 2\n",
        ),
        encoding="utf-8",
    )
    pilot_path = _write_pilot(tmp_path, evaluation={**_common(tmp_path, 1, {})["evaluation"], "metrics": ["ssim"]})

    with pytest.raises(ConfigError) as error:
        check_pilot(load_suite(pilot_path), load_suite(comparison_path), comparison_path)

    assert "is not evaluated with psnr" in str(error.value)
    assert "needs one line 'encoder: {learning_rate: ..., ...}'" in str(error.value)


def test_the_check_before_the_pilot_finds_a_rate_line_that_would_not_take_effect(tmp_path: Path):
    """Review F141: a later setting in the run's block overrides the line the chosen rate is written into."""
    comparison_path = _write_comparison(tmp_path)
    line = "        encoder: {learning_rate: 1.0e-4, warmup_steps: 2}   # placeholder until the pilot has run\n"
    override = "    optimization:\n      group_overrides:\n        encoder:\n          learning_rate: 1.0e-4\n          warmup_steps: 2\n"
    text = comparison_path.read_text(encoding="utf-8")
    comparison_path.write_text(text.replace(line, line + override), encoding="utf-8")
    pilot_path = _write_pilot(tmp_path)

    with pytest.raises(ConfigError, match="does not take effect for tiny"):
        check_pilot(load_suite(pilot_path), load_suite(comparison_path), comparison_path)


def test_a_split_directory_is_the_same_split_however_its_path_is_written():
    """Reviews F124 and F138: ./data/srn_cars/cars_test is the test split too, also with another near."""
    test = {"name": "srn", "split_dir": "data/srn_cars/cars_test", "near": 0.8}

    assert _same_data(test, {**test, "split_dir": "./data/srn_cars/cars_test"})
    assert _same_data(test, {**test, "split_dir": "data/srn_cars/../srn_cars/cars_test"})
    assert not _same_data(test, {**test, "split_dir": "data/srn_cars/cars_val"})
    assert not _same_data(test, {**test, "near": 0.9})
    # The same scenes, whatever else differs.
    assert _reads_same_scenes(test, {**test, "split_dir": "./data/srn_cars/cars_test", "near": 0.9})
    assert not _reads_same_scenes(test, {**test, "split_dir": "data/srn_cars/cars_val"})


@pytest.mark.parametrize("value", [1.0e-5, 3.0e-5, 1.0e-4, 2.5e-5, 2.0, 0.1, 1 / 3])
def test_rates_are_written_as_numbers_that_read_back_exactly(value: float):
    assert yaml.safe_load(f"rate: {_yaml_float(value)}") == {"rate": value}


def test_rates_can_be_written_into_the_shipped_comparison(tmp_path: Path):
    """The text edit must work on the real file, with its comments and commented-out baselines."""
    shutil.copytree(CONFIG_DIR, tmp_path / "configs")  # the suite's base files resolve relative to it
    path = tmp_path / "configs" / "suites" / "proposal_comparison.yaml"
    before = load_suite(path)
    # Rates other than those in the file, which holds the chosen ones once a pilot has run (review F134).
    names = ("convnextv2_mlp", "hiera_mlp", "hiera_nerf")
    current = {name: before.runs[f"{name}/seed_0"].optimization.group_overrides["encoder"]["learning_rate"] for name in names}
    chosen = {name: 3.0e-5 if rate != 3.0e-5 else 1.0e-4 for name, rate in current.items()}
    choices = [Choice(TunedRun(name, f"{name}/seed_0", ()), {}, rate) for name, rate in chosen.items()]

    write_rates_into_suite(path, choices)

    after = load_suite(path)
    for name, rate in chosen.items():
        assert after.runs[f"{name}/seed_0"].optimization.group_overrides["encoder"]["learning_rate"] == rate
        # Nothing else about the run changed.
        assert {**after.runs[f"{name}/seed_0"].raw, "optimization": None} == {
            **before.runs[f"{name}/seed_0"].raw,
            "optimization": None,
        }
    # Only the three rate lines changed; every comment and commented-out baseline stays.
    shipped = (CONFIG_DIR / "suites" / "proposal_comparison.yaml").read_text(encoding="utf-8").splitlines()
    written = path.read_text(encoding="utf-8").splitlines()
    changed = [(old, new) for old, new in zip(shipped, written) if old != new]
    assert len(written) == len(shipped) and len(changed) == 3
    assert all("encoder: {learning_rate:" in new for _, new in changed)
