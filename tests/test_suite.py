"""Tests of suite files and the suite runner, on runs small enough for a CPU."""

import csv
from pathlib import Path

import pytest
import yaml

from scene_recon import suite as suite_module
from scene_recon.registry import ConfigError
from scene_recon.run_files import RunFiles
from scene_recon.suite import PreflightError, load_suite, run_suite

from conftest import CONFIG_DIR


def _write_suite(tmp_path: Path, runs: dict, stages: list[str] | None = None) -> Path:
    """Write a suite file whose runs build on the smoke test, shrunk to a few training steps."""
    content = {
        "suite_name": "test_suite",
        "stages": stages or ["train", "evaluate", "profile"],
        "common": {
            "base": str(CONFIG_DIR / "smoke_test.yaml"),
            "experiment": {"output_dir": str(tmp_path / "runs"), "device": "cpu"},
            "data": {split: {"image_size": 16} for split in ("train", "val", "test")},
            "training": {
                "num_steps": 4,
                "batch_size": 2,
                "rays_per_scene": 32,
                "log_every": 2,
                "validate_every": 2,
                "checkpoint_every": 2,
            },
            "validation": {"num_scenes": 1, "num_target_views": 1},
            "evaluation": {"num_scenes": 2, "num_target_views": 2, "num_preview_scenes": 2},
            "profiling": {"num_frames": 1, "num_warmup_frames": 0},
        },
        "runs": runs,
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False))  # keep the runs in the given order
    return path


TWO_RUNS = {
    "narrow": {"model": {"decoder": {"hidden_width": 16}}},
    "wide": {"model": {"decoder": {"hidden_width": 48}}},
}


class _CallRecorder:
    """Replaces a stage function in the suite module, recording calls and optionally running the real one."""

    def __init__(self, monkeypatch, name: str, run_real_function: bool = True) -> None:
        self.calls: list[tuple] = []
        real_function = getattr(suite_module, name)

        def recorder(config, *args, **kwargs):
            self.calls.append((config.experiment.run_name, args, kwargs))
            if run_real_function:
                return real_function(config, *args, **kwargs)
            raise AssertionError(f"{name} should not have been called")

        monkeypatch.setattr(suite_module, name, recorder)


# ---------------------------------------------------------------- loading


def test_runs_combine_common_settings_with_their_own(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))

    assert list(suite.runs) == ["narrow", "wide"]
    assert suite.directory == tmp_path / "runs" / "test_suite"
    for run_name, config in suite.runs.items():
        assert config.experiment.run_name == run_name
        assert config.run_dir == suite.directory / run_name
        assert config.training.num_steps == 4  # from common
        assert config.model["encoder"]["num_features"] == 16  # from the smoke test the common section builds on
    assert suite.runs["narrow"].model["decoder"]["hidden_width"] == 16
    assert suite.runs["wide"].model["decoder"]["hidden_width"] == 48


def test_command_line_overrides_apply_to_every_run(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["training.num_steps=9"])

    assert all(config.training.num_steps == 9 for config in suite.runs.values())


def test_a_run_can_name_its_own_base_files(tmp_path: Path):
    model_file = tmp_path / "my_model.yaml"
    model_file.write_text(yaml.safe_dump({"model": {"decoder": {"hidden_width": 24}}}))

    suite = load_suite(_write_suite(tmp_path, {"from_file": {"base": "my_model.yaml"}}))

    assert suite.runs["from_file"].model["decoder"]["hidden_width"] == 24


def test_every_broken_run_is_reported_at_once(tmp_path: Path):
    runs = {
        "typo": {"training": {"num_stepz": 3}},
        "fine": {},
        "missing_file": {"base": "does_not_exist.yaml"},
    }

    with pytest.raises(ConfigError) as error:
        load_suite(_write_suite(tmp_path, runs))

    assert "typo" in str(error.value) and "num_stepz" in str(error.value)
    assert "missing_file" in str(error.value) and "does_not_exist.yaml" in str(error.value)
    assert "fine:" not in str(error.value)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"stages": ["train", "deploy"]}, "Unknown stage"),
        ({"runs": {"bad/name": {}}}, "Invalid run name"),
        ({"suite_name": "has space"}, "Invalid suite_name"),
        ({"extra_key": 1}, "Unknown key"),
    ],
)
def test_malformed_suite_files_are_rejected(tmp_path: Path, change: dict, message: str):
    path = _write_suite(tmp_path, TWO_RUNS)
    content = yaml.safe_load(path.read_text())
    content.update(change)
    path.write_text(yaml.safe_dump(content))

    with pytest.raises(ConfigError, match=message):
        load_suite(path)


def test_stages_always_run_in_pipeline_order(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS, stages=["profile", "train"]))

    assert suite.stages == ["train", "profile"]


@pytest.mark.parametrize(
    "run_change",
    [
        {"evaluation": {"metrics": ["psnr"]}},
        {"validation": {"source_views": [1]}},
        {"profiling": {"num_frames": 2}},
        {"data": {"test": {"num_scenes": 3}}},
    ],
)
def test_runs_must_be_measured_identically(tmp_path: Path, run_change: dict):
    runs = {"reference": {}, "different": run_change}

    with pytest.raises(ConfigError, match="must be the same for every run") as error:
        load_suite(_write_suite(tmp_path, runs))

    assert "different=" in str(error.value)


def test_report_lists_what_differs_between_runs(tmp_path: Path):
    runs = {**TWO_RUNS, "fewer_samples": {"renderer": {"num_coarse_samples": 16}}}
    suite = load_suite(_write_suite(tmp_path, runs))

    run_suite(suite)

    report = (suite.directory / "summary.md").read_text()
    differences = report.split("## What differs between runs")[1].split("##")[0]
    assert "| `model.decoder.hidden_width` | `16` | `48` | `64` |" in differences
    assert "| `renderer.num_coarse_samples` | `24` | `24` | `16` |" in differences
    assert "trained on identical data" in differences
    assert "training.num_steps" not in differences


def test_shipped_smoke_suite_is_valid():
    suite = load_suite(CONFIG_DIR / "suites" / "smoke_suite.yaml")

    assert list(suite.runs) == ["tiny_narrow", "tiny_wide", "tiny_without_fine_pass"]


# ---------------------------------------------------------------- running


def test_suite_runs_every_stage_of_every_run_and_writes_the_report(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))

    failures = run_suite(suite)

    assert failures == {}
    for config in suite.runs.values():
        files = RunFiles(config.run_dir)
        assert files.checkpoint.is_file() and files.evaluation.is_file() and files.profile.is_file()
        assert len(list(files.previews(1).iterdir())) == 2

    with open(suite.directory / "summary.csv") as file:
        rows = list(csv.DictReader(file))
    assert [row["run"] for row in rows] == ["narrow", "wide"]
    assert all(row["status"] == "evaluated" and row["psnr_1v"] and row["ssim_2v"] for row in rows)

    report = (suite.directory / "summary.md").read_text()
    assert "## Quality with 1 input view" in report and "## Quality with 2 input views" in report
    plots = {path.name for path in (suite.directory / "plots").iterdir()}
    assert plots == {
        "metrics_vs_source_views.png",
        "quality_vs_speed.png",
        "training_curves.png",
        "renders_1_source_views.png",
        "renders_2_source_views.png",
    }


def test_running_a_finished_suite_again_repeats_no_work(tmp_path: Path, monkeypatch):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))
    run_suite(suite)

    for stage in ("run_training", "run_evaluation", "run_profiling"):
        _CallRecorder(monkeypatch, stage, run_real_function=False)

    assert run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS))) == {}


def test_more_training_steps_resume_training_and_redo_evaluation(tmp_path: Path, monkeypatch):
    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS)))
    training = _CallRecorder(monkeypatch, "run_training")
    evaluation = _CallRecorder(monkeypatch, "run_evaluation")
    _CallRecorder(monkeypatch, "run_profiling", run_real_function=False)

    suite = load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["training.num_steps=6"])
    run_suite(suite)

    assert training.calls == [("narrow", (), {"resume": True}), ("wide", (), {"resume": True})]
    assert [call[0] for call in evaluation.calls] == ["narrow", "wide"]
    for config in suite.runs.values():
        assert '"training_step": 6' in RunFiles(config.run_dir).evaluation.read_text()


def test_changed_evaluation_settings_redo_only_the_evaluation(tmp_path: Path, monkeypatch):
    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS)))
    _CallRecorder(monkeypatch, "run_training", run_real_function=False)
    evaluation = _CallRecorder(monkeypatch, "run_evaluation")
    _CallRecorder(monkeypatch, "run_profiling", run_real_function=False)

    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["evaluation.metrics=[psnr]"]))

    assert [call[0] for call in evaluation.calls] == ["narrow", "wide"]


def test_a_failing_run_does_not_stop_the_others(tmp_path: Path, monkeypatch):
    real_training = suite_module.run_training

    def training_that_fails_for_narrow(config, *args, **kwargs):
        if config.experiment.run_name == "narrow":
            raise RuntimeError("CUDA out of memory (simulated)")
        return real_training(config, *args, **kwargs)

    monkeypatch.setattr(suite_module, "run_training", training_that_fails_for_narrow)
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))

    failures = run_suite(suite)

    assert list(failures) == ["narrow"] and "out of memory" in failures["narrow"]
    assert RunFiles(suite.runs["wide"].run_dir).evaluation.is_file()
    report = (suite.directory / "summary.md").read_text()
    assert "## Failed runs" in report and "| narrow | failed |" in report


def test_preflight_stops_the_suite_before_any_training(tmp_path: Path):
    runs = {
        "fine": {},
        "unknown_encoder": {"model": {"encoder": {"name": "no_such_encoder"}}},
        "too_many_source_views": {"training": {"num_source_views": [1, 9]}},
    }
    suite = load_suite(_write_suite(tmp_path, runs))

    with pytest.raises(PreflightError) as error:
        run_suite(suite)

    assert "2 of 3 run(s) cannot start" in str(error.value)
    assert "no_such_encoder" in str(error.value) and "up to 9 source views" in str(error.value)
    assert not RunFiles(suite.runs["fine"].run_dir).checkpoint.exists()


def test_a_trained_run_whose_model_changed_is_not_resumed(tmp_path: Path):
    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS)))
    changed = {**TWO_RUNS, "wide": {"model": {"decoder": {"hidden_width": 64}}}}

    with pytest.raises(PreflightError, match="different model config"):
        run_suite(load_suite(_write_suite(tmp_path, changed), overrides=["training.num_steps=6"]))
