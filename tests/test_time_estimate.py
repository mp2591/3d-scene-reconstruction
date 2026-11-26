"""The time estimate (scene-recon estimate-time), and a step budget given to the comparison alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from scene_recon import time_estimate
from scene_recon.build import build_dataset
from scene_recon.checkpoints import read_progress
from scene_recon.cli import main
from scene_recon.pilot import choose_and_continue
from scene_recon.run_files import RunFiles
from scene_recon.suite import load_suite, run_suite
from scene_recon.time_estimate import (
    Estimate,
    Saving,
    Speed,
    _views,
    format_estimates,
    measure_saving,
    measure_speed,
    plan_hours,
)

from test_pilot import COMPARISON_STEPS, PILOT_STEPS, _set_scores, _write_comparison, _write_pilot

# The toy suites of test_pilot: the comparison trains 4 steps and the pilot 2, both validated
# every 2 steps on 1 view of 1 scene with source view [0], and scored on 2 views of 2 scenes
# with source views [0] and [0, 6]. Each test split has 4 scenes of 12 views.
ONE_VIEW, TWO_VIEWS = (0,), (0, 6)
NO_SAVING = Saving(0.0, 0.0)


def _suites(tmp_path: Path):
    return load_suite(_write_pilot(tmp_path)), load_suite(_write_comparison(tmp_path))


def _speeds(step: dict[str, float], one_view: float, two_views: float) -> dict[str, Speed]:
    return {
        f"{name}/seed_0": Speed(seconds, {ONE_VIEW: one_view, TWO_VIEWS: two_views}) for name, seconds in step.items()
    }


def test_training_time_counts_each_run_from_where_it_starts(tmp_path: Path):
    """A tuned run continues from the pilot's last step; a run the pilot does not tune starts at 0."""
    pilot, comparison = _suites(tmp_path)
    speeds = _speeds({"tiny": 1.0, "untuned": 10.0}, one_view=0.0, two_views=0.0)

    estimates = plan_hours(pilot, comparison, speeds, NO_SAVING, [COMPARISON_STEPS, 100], first_scenes=None)

    pilot_seconds = 2 * PILOT_STEPS * 1.0  # two rates of the tuned run
    assert [estimate.steps for estimate in estimates] == [COMPARISON_STEPS, 100]
    for estimate, steps in zip(estimates, [COMPARISON_STEPS, 100]):
        comparison_seconds = (steps - PILOT_STEPS) * 1.0 + steps * 10.0
        assert estimate.pilot_done * 3600 == pytest.approx(pilot_seconds)
        assert estimate.first_scores * 3600 == pytest.approx(pilot_seconds + comparison_seconds)
        assert estimate.all_scores == estimate.first_scores  # nothing is scored a second time


def test_scoring_time_counts_every_validation_and_evaluation(tmp_path: Path):
    pilot, comparison = _suites(tmp_path)
    speeds = _speeds({"tiny": 0.0, "untuned": 0.0}, one_view=1.0, two_views=100.0)

    (estimate,) = plan_hours(pilot, comparison, speeds, NO_SAVING, [COMPARISON_STEPS], first_scenes=1)

    # Each pilot run: 1 validation of 1 view, then 2 scenes x 2 views with each protocol.
    pilot_seconds = 2 * (1 * 1.0 + 4 * 1.0 + 4 * 100.0)
    # The comparison scored on 1 scene first: the tuned run validates at step 4 only (step 2 was
    # the pilot's), the other run at steps 2 and 4.
    first = (1 + 2 * 1.0 + 2 * 100.0) + (2 + 2 * 1.0 + 2 * 100.0)
    # Then on its 2 scenes, without training again.
    later = 2 * (4 * 1.0 + 4 * 100.0)
    assert estimate.pilot_done * 3600 == pytest.approx(pilot_seconds)
    assert estimate.first_scores * 3600 == pytest.approx(pilot_seconds + first)
    assert estimate.all_scores * 3600 == pytest.approx(pilot_seconds + first + later)


def test_saving_time_counts_every_checkpoint_save(tmp_path: Path):
    """Both toy suites save every 2 steps and at the end; each suite command checks the data in full once."""
    pilot, comparison = _suites(tmp_path)
    speeds = _speeds({"tiny": 0.0, "untuned": 0.0}, one_view=0.0, two_views=0.0)

    (estimate,) = plan_hours(pilot, comparison, speeds, Saving(1000.0, 1.0), [COMPARISON_STEPS], first_scenes=None)

    pilot_seconds = 1000 + 2 * (1 + 1)  # each pilot run: at step 2, and at the end
    comparison_seconds = 1000 + (1 + 1) + (2 + 1)  # the tuned run from step 2: at step 4; the other at 2 and 4
    assert estimate.pilot_done * 3600 == pytest.approx(pilot_seconds)
    assert estimate.first_scores * 3600 == pytest.approx(pilot_seconds + comparison_seconds)


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
    estimates = [Estimate(100000, 5.0, 20.0, 30.0), Estimate(200000, 5.0, 35.0, 45.0)]

    text = format_estimates(speeds, Saving(30.0, 2.5), estimates, 100, "Test GPU")

    assert "Measured on this machine (Test GPU):" in text
    assert "run/seed_0: 0.250 s per training step; scoring a view 0.50 s with 1 source view, 0.75 s with 2 source views" in text
    assert "checking the training data at each checkpoint save: 2.5 s (30.0 s the first time)" in text
    assert "steps per run | pilot done | trained, scored on 100 test scenes | final report" in text
    row = next(line for line in text.splitlines() if line.strip().startswith("200,000"))
    assert [cell.strip() for cell in row.split("|")] == ["200,000", "5.0", "35.0", "45.0"]
    # Scored on every scene at once: no final column.
    assert "final report" not in format_estimates(speeds, NO_SAVING, estimates, None, "Test GPU")


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
    assert saving.first_seconds >= 0 and saving.seconds_per_save >= 0


def test_the_command_prints_an_estimate_for_each_budget(tmp_path: Path, few_timed_steps, capsys):
    pilot_path, comparison_path = _write_pilot(tmp_path), _write_comparison(tmp_path)

    main(["estimate-time", "--pilot", str(pilot_path), "--comparison", str(comparison_path),
          "--steps", "4", "8", "--first-scenes", "1", "--comparison-set", "training.num_steps=8"])

    lines = capsys.readouterr().out.splitlines()
    assert any(line.strip().startswith("4 |") for line in lines)
    assert any(line.strip().startswith("8 |") for line in lines)
    assert not (tmp_path / "runs").exists()  # nothing was trained for real


# ---------------------------------------------------------------- a step budget for the comparison alone


def test_a_budget_given_to_the_comparison_alone_continues_the_chosen_run_to_it(tmp_path: Path):
    """As scripts/run_comparison.sh does: the schedule for both suites, the length for the comparison."""
    both, comparison_only = ["optimization.decay_steps=600"], ["training.num_steps=6"]
    comparison_path, pilot_path = _write_comparison(tmp_path), _write_pilot(tmp_path)
    assert run_suite(load_suite(pilot_path, both)) == {}
    _set_scores(pilot_path, {1.0e-4: 20.0, 3.0e-4: 21.0})

    choose_and_continue(pilot_path, comparison_path, both, comparison_only)

    comparison = load_suite(comparison_path, [*both, *comparison_only])
    assert run_suite(comparison) == {}
    continued = RunFiles(comparison.runs["tiny/seed_0"].run_dir).checkpoint
    assert read_progress(continued).step == 6
    # The pilot kept its own length.
    pilot = load_suite(pilot_path, both)
    assert all(read_progress(RunFiles(config.run_dir).checkpoint).step == PILOT_STEPS for config in pilot.runs.values())


def test_a_budget_shorter_than_the_pilot_is_refused_before_the_pilot_runs(tmp_path: Path):
    comparison_path, pilot_path = _write_comparison(tmp_path), _write_pilot(tmp_path)

    with pytest.raises(SystemExit, match="more than the 1 of"):
        main(["choose-encoder-rates", "--pilot", str(pilot_path), "--comparison", str(comparison_path),
              "--check-only", "--comparison-set", "training.num_steps=1"])
