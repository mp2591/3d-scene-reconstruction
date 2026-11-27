"""The time estimate (scene-recon estimate-time)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scene_recon import time_estimate
from scene_recon.build import build_dataset
from scene_recon.cli import main
from scene_recon.suite import load_suite
from scene_recon import provenance
from scene_recon.suite import run_suite
from scene_recon.time_estimate import (
    Estimate,
    Saving,
    Speed,
    _views,
    format_estimates,
    largest_budget_within,
    load_estimates,
    measure_saving,
    measure_speed,
    measure_speeds,
    plan_hours,
    checks_of_test_data_per_command,
    save_estimates,
)

from conftest import CONFIG_DIR

# A toy suite of two runs. Each trains 4 steps, is validated and saves a checkpoint every 2 steps
# (validation: 1 view of 1 scene with source view [0]), and is scored on 2 views of 2 scenes with
# source views [0] and [0, 6]. The test split has 4 scenes of 12 views.
STEPS = 4
ONE_VIEW, TWO_VIEWS = (0,), (0, 6)
NO_SAVING = Saving(0.0, 0.0, 0.0, 0.0)


def _write_suite(
    tmp_path: Path, stages: list[str] | None = None, seeds: list[int] | None = None, parallel_runs: int = 1
) -> Path:
    common = {
        "base": str(CONFIG_DIR / "smoke_test.yaml"),
        "experiment": {"output_dir": str(tmp_path / "runs"), "device": "cpu"},
        "data": {"train": {"image_size": 16}, "val": {"image_size": 16}, "test": {"image_size": 16}},
        "training": {
            "num_steps": STEPS,
            "batch_size": 2,
            "rays_per_scene": 32,
            "log_every": 2,
            "validate_every": 2,
            "checkpoint_every": 2,
        },
        "validation": {"num_scenes": 1, "num_target_views": 1},
        "evaluation": {"num_scenes": 2, "num_target_views": 2, "num_preview_scenes": 0},
    }
    runs = {
        "small": {"model": {"decoder": {"hidden_width": 16}}},
        "large": {"model": {"decoder": {"hidden_width": 24}}},
    }
    content = {
        "suite_name": "toy",
        "stages": stages or ["train", "evaluate"],
        "seeds": seeds or [0],
        "parallel_runs": parallel_runs,
        "common": common,
        "runs": runs,
    }
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def _speeds(step: dict[str, float], one_view: float, two_views: float) -> dict[str, Speed]:
    return {
        f"{name}/seed_0": Speed(seconds, {ONE_VIEW: one_view, TWO_VIEWS: two_views}) for name, seconds in step.items()
    }


def test_training_time_counts_every_step_of_every_run(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path))
    speeds = _speeds({"small": 1.0, "large": 10.0}, one_view=0.0, two_views=0.0)

    estimates = plan_hours(suite, speeds, NO_SAVING, [STEPS, 100], None, None, runs_at_once=1)

    assert [estimate.steps for estimate in estimates] == [STEPS, 100]
    for estimate, steps in zip(estimates, [STEPS, 100]):
        assert estimate.first_scores * 3600 == pytest.approx(steps * 1.0 + steps * 10.0)
        assert estimate.all_scores == estimate.first_scores  # nothing is scored a second time


def test_scoring_time_counts_every_validation_and_evaluation(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path))
    speeds = _speeds({"small": 0.0, "large": 0.0}, one_view=1.0, two_views=100.0)

    (estimate,) = plan_hours(suite, speeds, NO_SAVING, [STEPS], first_scenes=1, first_target_views=None, runs_at_once=1)

    # Each run: validated at steps 2 and 4, then scored on 1 scene x 2 views with each protocol...
    first = 2 * (2 * 1.0 + 2 * 1.0 + 2 * 100.0)
    # ...then on its 2 scenes, without training again.
    later = 2 * (4 * 1.0 + 4 * 100.0)
    assert estimate.first_scores * 3600 == pytest.approx(first)
    assert estimate.all_scores * 3600 == pytest.approx(first + later)


def test_the_first_scores_can_cover_fewer_views_of_each_scene(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path))
    speeds = _speeds({"small": 0.0, "large": 0.0}, one_view=1.0, two_views=100.0)

    (estimate,) = plan_hours(suite, speeds, NO_SAVING, [STEPS], first_scenes=1, first_target_views=1, runs_at_once=1)

    # Each run: validated at steps 2 and 4, then scored on 1 view of 1 scene with each protocol...
    first = 2 * (2 * 1.0 + 1 * 1.0 + 1 * 100.0)
    # ...then on 2 views of its 2 scenes, as its evaluation settings say.
    later = 2 * (4 * 1.0 + 4 * 100.0)
    assert estimate.first_scores * 3600 == pytest.approx(first)
    assert estimate.all_scores * 3600 == pytest.approx(first + later)


def test_runs_at_the_same_time_last_as_long_as_the_slowest(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path))
    speeds = _speeds({"small": 1.0, "large": 10.0}, one_view=0.0, two_views=0.0)

    (alone,) = plan_hours(suite, speeds, NO_SAVING, [STEPS], None, None, runs_at_once=1)
    (together,) = plan_hours(suite, speeds, NO_SAVING, [STEPS], None, None, runs_at_once=2)

    assert alone.first_scores * 3600 == pytest.approx(STEPS * 1.0 + STEPS * 10.0)
    assert together.first_scores * 3600 == pytest.approx(STEPS * 10.0)

    # Rendering speed is measured one run at a time, whatever else runs together.
    (tmp_path / "profile").mkdir()
    profiling = load_suite(_write_suite(tmp_path / "profile", stages=["profile"]))
    speeds = _speeds({"small": 0.0, "large": 0.0}, one_view=1.0, two_views=1.0)
    (alone,) = plan_hours(profiling, speeds, NO_SAVING, [STEPS], None, None, runs_at_once=1)
    (together,) = plan_hours(profiling, speeds, NO_SAVING, [STEPS], None, None, runs_at_once=2)
    assert together.first_scores == pytest.approx(alone.first_scores) and alone.first_scores > 0


def test_the_largest_budget_that_fits_in_the_hours():
    estimates = [Estimate(100000, 8.0, 30.0), Estimate(200000, 15.0, 40.0), Estimate(300000, 22.0, 50.0)]

    assert largest_budget_within(estimates, 18.0) == 200000
    assert largest_budget_within(estimates, 22.0) == 300000  # the first report counts, not the final one
    assert largest_budget_within(estimates, 5.0) is None


def test_estimates_are_saved_and_read_back(tmp_path: Path):
    estimates = [Estimate(100000, 8.25, 30.5), Estimate(200000, 15.0, 40.0)]

    save_estimates(estimates, tmp_path / "estimates.json")

    assert load_estimates(tmp_path / "estimates.json") == estimates


def test_saving_time_counts_every_checkpoint_save(tmp_path: Path):
    """Each run saves every 2 steps and at the end; each command checks both splits in full once."""
    suite = load_suite(_write_suite(tmp_path))
    speeds = _speeds({"small": 0.0, "large": 0.0}, one_view=0.0, two_views=0.0)

    saving = Saving(1000.0, 1.0, 500.0, 2.0)
    # Per command: the training split read once; the test split read once, then checked again.
    command = 1000 + 500 + (checks_of_test_data_per_command(2) - 1) * 2.0

    (estimate,) = plan_hours(suite, speeds, saving, [STEPS], None, None, runs_at_once=1)
    assert estimate.first_scores * 3600 == pytest.approx(command + 2 * (2 + 1))  # at steps 2 and 4, and at the end

    # Scored on the first scene, then on all: a second command, which checks both splits again.
    (estimate,) = plan_hours(suite, speeds, saving, [STEPS], first_scenes=1, first_target_views=None, runs_at_once=1)
    assert estimate.all_scores * 3600 == pytest.approx(2 * command + 2 * (2 + 1))


def test_a_command_checks_the_test_data_no_more_often_than_the_estimate_counts(tmp_path: Path, monkeypatch):
    checks = []
    real_build_dataset = provenance.build_dataset

    def counting_build_dataset(config, split):
        scenes = real_build_dataset(config, split)
        if split == "test":
            real_fingerprint = scenes.fingerprint
            monkeypatch.setattr(scenes, "fingerprint", lambda: checks.append(1) or real_fingerprint())
        return scenes

    monkeypatch.setattr(provenance, "build_dataset", counting_build_dataset)
    # Two models with two seeds each: four runs, and the report written after each.
    path = _write_suite(tmp_path, stages=["train", "evaluate", "profile"], seeds=[0, 1])
    suite = load_suite(path, overrides=["evaluation.num_scenes=1"])
    for command in (suite, load_suite(path)):  # the first scenes, then all of them, as the launcher does
        checks.clear()
        assert run_suite(command) == {}
        assert 0 < len(checks) <= checks_of_test_data_per_command(len(command.runs)) and len(command.runs) == 4


@pytest.mark.parametrize(
    "source_views, num_scenes, per_scene, views",
    [
        ([0], 2, 2, 4),
        ([0], 100, 2, 8),  # more scenes than the split has: all 4
        ([0], None, None, 44),  # every scene, every view that is not a source view
        ([0, 6], None, None, 40),
        ([0], 1, 50, 11),  # more views than a scene has
    ],
)
def test_the_views_an_evaluation_scores(smoke_config, source_views, num_scenes, per_scene, views):
    assert _views(build_dataset(smoke_config, "test"), source_views, num_scenes, per_scene) == views


def test_the_estimates_are_written_as_a_table():
    speeds = {"run/seed_0": Speed(0.25, {ONE_VIEW: 0.5, TWO_VIEWS: 0.75})}
    estimates = [Estimate(100000, 20.0, 30.0), Estimate(200000, 35.0, 45.0)]

    text = format_estimates(speeds, Saving(30.0, 2.5, 40.0, 3.5), estimates, 100, None, "Test GPU", 1)

    assert "Measured on this machine (Test GPU), one run at a time:" in text
    assert "run/seed_0: 0.250 s per training step; scoring a view 0.50 s with 1 source view, 0.75 s with 2 source views" in text
    assert "checking the training data at each checkpoint save: 2.5 s (30.0 s the first time in each command)" in text
    assert "checking the test data: 40.0 s the first time in each command, 3.5 s each later time" in text
    assert "steps per run | trained, scored on 100 test scenes | final report" in text
    row = next(line for line in text.splitlines() if line.strip().startswith("200,000"))
    assert [cell.strip() for cell in row.split("|")] == ["200,000", "35.0", "45.0"]
    # Scored on every scene at once: no final column.
    assert "final report" not in format_estimates(speeds, NO_SAVING, estimates, None, None, "Test GPU", 1)

    text = format_estimates(speeds, NO_SAVING, estimates, 100, 10, "Test GPU", 3)
    assert "Measured on this machine (Test GPU), 3 runs at the same time, as they will train:" in text
    assert "steps per run | trained, scored on 10 views of 100 test scenes | final report" in text


@pytest.fixture
def few_timed_steps(monkeypatch):
    """Time as few steps and views as possible, so that the tests stay quick on a CPU."""
    for name, value in [("WARMUP_STEPS", 1), ("TIMED_STEPS", 2), ("TIMED_SCENES", 1), ("TIMED_VIEWS_PER_SCENE", 1)]:
        monkeypatch.setattr(time_estimate, name, value)


def test_timing_a_run_leaves_its_folder_alone(smoke_config, few_timed_steps):
    speed = measure_speed(smoke_config)

    assert speed.seconds_per_step > 0
    assert set(speed.seconds_per_view) == {ONE_VIEW, TWO_VIEWS}
    assert all(seconds > 0 for seconds in speed.seconds_per_view.values())
    assert not smoke_config.run_dir.exists()
    saving = measure_saving(smoke_config)
    assert min(saving.first_seconds, saving.seconds_per_save, saving.test_first_seconds, saving.test_later_seconds) >= 0


def test_the_command_prints_an_estimate_for_each_budget(tmp_path: Path, few_timed_steps, capsys):
    config, saved = str(_write_suite(tmp_path)), tmp_path / "estimates.json"
    main(["estimate-time", "--config", config, "--steps", "4", "8", "--first-scenes", "1", "--first-target-views", "1", "--json", str(saved)])

    lines = capsys.readouterr().out.splitlines()
    assert any(line.strip().startswith("4 |") for line in lines)
    assert any(line.strip().startswith("8 |") for line in lines)
    assert [estimate.steps for estimate in load_estimates(saved)] == [4, 8]
    assert not (tmp_path / "runs").exists()  # nothing was trained for real


def test_runs_that_train_together_are_timed_together(tmp_path: Path):
    """Each in a process of its own, with the module's own numbers of timed steps and views."""
    suite = load_suite(_write_suite(tmp_path, parallel_runs=2))

    speeds, runs_at_once = measure_speeds(suite)

    assert runs_at_once == 2
    assert set(speeds) == set(suite.runs)
    for speed in speeds.values():
        assert speed.seconds_per_step > 0 and set(speed.seconds_per_view) == {ONE_VIEW, TWO_VIEWS}
    assert not (tmp_path / "runs").exists()


def test_runs_are_timed_alone_if_timing_them_together_fails(tmp_path: Path, few_timed_steps, monkeypatch):
    def fail(suite):
        raise RuntimeError("the GPU cannot be shared")

    monkeypatch.setattr(time_estimate, "_measure_together", fail)
    suite = load_suite(_write_suite(tmp_path, parallel_runs=2))

    speeds, runs_at_once = measure_speeds(suite)

    assert runs_at_once == 1  # so the estimate counts the runs one after another
    assert set(speeds) == set(suite.runs)
