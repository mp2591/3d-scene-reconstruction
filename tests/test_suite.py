"""Tests of suite files and the suite runner, on runs small enough for a CPU."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scene_recon import suite as suite_module
from scene_recon.config import flatten_settings
from scene_recon.registry import ConfigError
from scene_recon.report import write_report
from scene_recon.run_files import RunFiles
from scene_recon.suite import PreflightError, SuiteReportError, load_suite, run_suite

from conftest import CONFIG_DIR, srn_overrides, write_srn_splits


def _write_suite(tmp_path: Path, runs: dict, stages: list[str] | None = None, seeds: list[int] | None = None) -> Path:
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
    if seeds is not None:
        content["seeds"] = seeds
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
        ({"suite_name": ".."}, "Invalid suite_name"),
        ({"runs": {".": {}}}, "Invalid run name"),
        ({"runs": {"Run": {}, "run": {}}}, "differ only in upper/lower case"),
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
        {"renderer": {"ray_chunk_size": 512}},  # rendering speed depends on it (review B(c))
        {"experiment": {"float32_matmul_precision": "highest"}},  # so does it on this
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

    with open(suite.directory / "summary.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["run"] for row in rows] == ["narrow", "wide"]
    assert all(row["status"] == "evaluated" and row["psnr_1v"] and row["ssim_2v"] for row in rows)
    # Every row ends in the csv module's "\r\n" exactly once, also on Windows.
    for name in ("summary.csv", "paired_comparison.csv"):
        assert b"\r\r\n" not in (suite.directory / name).read_bytes()

    # The report is UTF-8 on every platform; its tables use characters such as "↑".
    report = (suite.directory / "summary.md").read_bytes().decode("utf-8")
    assert "↑" in report
    assert "## Quality with 1 input view" in report and "## Quality with 2 input views" in report
    # Figures are linked with forward slashes, which every Markdown viewer follows (Windows paths use "\\").
    assert "](plots/metrics_vs_source_views.png)" in report and "](plots\\" not in report
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
        evaluation = json.loads(RunFiles(config.run_dir).evaluation.read_text())
        assert evaluation["checkpoint"]["step"] == 6


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

    with pytest.raises(PreflightError, match="model.decoder.hidden_width: trained with 48, requested 64"):
        run_suite(load_suite(_write_suite(tmp_path, changed), overrides=["training.num_steps=6"]))



# ---------------------------------------------------------------- provenance and recovery


def test_a_finished_run_whose_training_settings_changed_is_not_skipped_silently(tmp_path: Path):
    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS)))
    changed = {**TWO_RUNS, "wide": {**TWO_RUNS["wide"], "optimization": {"learning_rate": 0.05}}}

    with pytest.raises(PreflightError, match="optimization.learning_rate: trained with 0.001, requested 0.05"):
        run_suite(load_suite(_write_suite(tmp_path, changed)))


def _evaluated_elsewhere(suite, run: str, change) -> None:
    """Change the software record of a run's evaluation.json, as if it was evaluated on another machine."""
    evaluation_file = RunFiles(suite.runs[run].run_dir).evaluation
    evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
    change(evaluation)
    evaluation_file.write_text(json.dumps(evaluation), encoding="utf-8")


def _paired_rows(suite) -> dict[str, dict]:
    with open(suite.directory / "paired_comparison.csv", encoding="utf-8") as file:
        return {row["metric"]: row for row in csv.DictReader(file)}


def test_report_keeps_evaluations_from_other_software_and_says_which(tmp_path: Path):
    """Review F46, F54, F66 and F67: a saved evaluation stays current, so the report can be made on another machine."""
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))
    run_suite(suite)
    # SciPy is not named anywhere in the project: SSIM uses it through scikit-image.
    _evaluated_elsewhere(suite, "wide", lambda evaluation: evaluation["software"]["packages"].update(scipy="0.0.1"))

    write_report(suite.name, suite.directory, suite.grouped_runs(), {}, suite.stages)

    report = (suite.directory / "summary.md").read_text(encoding="utf-8")
    assert "## Quality with" in report and "evaluation:" not in report  # nothing left out
    assert "the runs were evaluated with different software" in report
    assert "| evaluation software that differs | scipy " in report and "scipy 0.0.1 |" in report
    with open(suite.directory / "summary.csv", encoding="utf-8") as file:
        hashes = {row["run"]: row["evaluation_software"] for row in csv.DictReader(file)}
    assert len(hashes["wide"]) == 12 and hashes["wide"] != hashes["narrow"]
    for row in _paired_rows(suite).values():  # every metric of the pair: the report cannot tell which depends on SciPy
        assert row["software"] == "different" and row["verdict"].endswith("scored with different software")


@pytest.mark.parametrize(
    ("damage", "problem"),
    [
        (lambda evaluation: evaluation.pop("software"), "not recorded"),
        (lambda evaluation: evaluation.update(software=None), "not recorded"),
        (lambda evaluation: evaluation["software"]["packages"].update(torch=["2.8"]), "damaged"),
        (lambda evaluation: evaluation["software"].update(device=None), "device model unknown"),
        (lambda evaluation: evaluation["software"].update(packages=None), "package list unreadable"),
    ],
    ids=["missing", "null", "malformed_version", "unnamed_device", "unreadable_packages"],
)
def test_an_evaluation_without_a_readable_software_record_makes_the_comparison_unknown(tmp_path: Path, damage, problem):
    """Review F59, F65, F68, F76 and F77: never taken for the same software; the report says why, and is written."""
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))
    run_suite(suite)
    _evaluated_elsewhere(suite, "wide", damage)

    write_report(suite.name, suite.directory, suite.grouped_runs(), {}, suite.stages)

    report = (suite.directory / "summary.md").read_text(encoding="utf-8")
    assert "## Quality with" in report and "the software of some evaluations cannot be compared" in report
    assert f"*{problem}*: " in report and f"| {problem} |" in report
    for row in _paired_rows(suite).values():
        assert row["software"] == "unknown" and row["verdict"].endswith("the software of a score is unknown")


def test_report_leaves_out_results_that_no_longer_match(tmp_path: Path):
    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS)))
    changed = load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["evaluation.source_view_sets=[[0, 6]]"])

    write_report(changed.name, changed.directory, changed.grouped_runs(), {}, changed.stages)

    report = (changed.directory / "summary.md").read_text()
    assert "## Quality with" not in report
    assert "- **narrow**, evaluation: made with different evaluation" in report
    with open(changed.directory / "summary.csv") as file:
        rows = list(csv.DictReader(file))
    assert all(row["status"] == "trained" for row in rows)
    assert not any(column.startswith("psnr") for column in rows[0])


def test_a_failed_reevaluation_does_not_bring_back_the_old_score(tmp_path: Path, monkeypatch):
    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS)))

    def failing_evaluation(config, *args, **kwargs):
        raise RuntimeError("evaluation crashed (simulated)")

    monkeypatch.setattr(suite_module, "run_evaluation", failing_evaluation)
    changed = load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["evaluation.source_view_sets=[[0, 6]]"])

    failures = run_suite(changed)

    assert set(failures) == {"narrow", "wide"}
    report = (changed.directory / "summary.md").read_text()
    assert "## Quality with" not in report and "| narrow | failed |" in report


def test_unknown_metric_stops_the_suite_before_training(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["evaluation.metrics=[definitely_unknown_metric]"])

    with pytest.raises(PreflightError, match="Unknown metric 'definitely_unknown_metric'"):
        run_suite(suite)

    assert not any(RunFiles(config.run_dir).checkpoint.exists() for config in suite.runs.values())


def test_a_profile_only_suite_needs_no_training_data(tmp_path: Path):
    runs = {"only_profiled": {"data": {"train": {"name": "srn", "split_dir": "/does/not/exist", "near": 0.8, "far": 1.8}}}}
    suite = load_suite(_write_suite(tmp_path, runs, stages=["profile"]))

    assert run_suite(suite) == {}
    assert RunFiles(suite.runs["only_profiled"].run_dir).profile.is_file()


def test_a_damaged_results_file_is_redone_and_does_not_stop_the_suite(tmp_path: Path, monkeypatch):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))
    run_suite(suite)
    narrow = RunFiles(suite.runs["narrow"].run_dir)
    narrow.evaluation.write_text('{"checkpoint":')  # as if a crash cut the write short
    narrow.profile.write_text("not json at all")
    _CallRecorder(monkeypatch, "run_training", run_real_function=False)
    evaluation = _CallRecorder(monkeypatch, "run_evaluation")
    profiling = _CallRecorder(monkeypatch, "run_profiling")

    assert run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS))) == {}

    assert [call[0] for call in evaluation.calls] == ["narrow"]
    assert [call[0] for call in profiling.calls] == ["narrow"]
    assert json.loads(narrow.evaluation.read_text())["protocols"]


@pytest.mark.parametrize(
    "damaged",
    ['{"step": 4,', '{"step": 4, "training_seconds": 1.0, "step": 300}', '{"step": "4", "training_seconds": 1.0}'],
    ids=["cut_short", "key_given_twice", "not_a_number"],
)
def test_a_damaged_progress_file_is_replaced_by_the_checkpoint(tmp_path: Path, monkeypatch, damaged: str):
    """Review F94: the step is read from the checkpoint itself, so more training steps resume training."""
    from scene_recon.checkpoints import progress_file

    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))
    run_suite(suite)
    narrow = RunFiles(suite.runs["narrow"].run_dir)
    progress_file(narrow.checkpoint).write_text(damaged)
    training = _CallRecorder(monkeypatch, "run_training")
    _CallRecorder(monkeypatch, "run_profiling", run_real_function=False)

    assert run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["training.num_steps=6"])) == {}

    assert training.calls == [("narrow", (), {"resume": True}), ("wide", (), {"resume": True})]
    assert json.loads(progress_file(narrow.checkpoint).read_text())["step"] == 6  # written again by the training


def test_report_survives_damaged_files(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))
    run_suite(suite)
    RunFiles(suite.runs["wide"].run_dir).profile.write_text("{")

    write_report(suite.name, suite.directory, suite.grouped_runs(), {}, suite.stages)

    assert "- **wide**, profile: profile.json is damaged" in (suite.directory / "summary.md").read_text()


def test_output_directory_can_be_overridden_for_a_whole_suite(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=[f"experiment.output_dir={tmp_path / 'elsewhere'}"])

    assert suite.directory == tmp_path / "elsewhere" / "test_suite"
    assert all(config.run_dir.parent == suite.directory for config in suite.runs.values())


def test_runs_cannot_choose_their_own_output_directory(tmp_path: Path):
    runs = {**TWO_RUNS, "elsewhere": {"experiment": {"output_dir": str(tmp_path / "other")}}}

    with pytest.raises(ConfigError, match="different experiment.output_dir"):
        load_suite(_write_suite(tmp_path, runs))



# ---------------------------------------------------------------- several seeds


def test_seeds_repeat_every_run_once_per_seed(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS, seeds=[3, 7]))

    assert list(suite.runs) == ["narrow/seed_3", "narrow/seed_7", "wide/seed_3", "wide/seed_7"]
    assert suite.groups == {"narrow": ["narrow/seed_3", "narrow/seed_7"], "wide": ["wide/seed_3", "wide/seed_7"]}
    config = suite.runs["wide/seed_7"]
    assert config.experiment.seed == 7
    assert config.run_dir == suite.directory / "wide" / "seed_7"


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1], [0.5], "0"])
def test_invalid_seed_lists_are_rejected(tmp_path: Path, seeds):
    with pytest.raises(ConfigError, match="'seeds' must be"):
        load_suite(_write_suite(tmp_path, TWO_RUNS, seeds=seeds))


def test_report_of_a_multi_seed_suite_shows_spread_and_paired_comparisons(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS, seeds=[0, 1]))

    assert run_suite(suite) == {}

    report = (suite.directory / "summary.md").read_text()
    assert "Every run was trained 2 times, with seeds 0, 1." in report
    assert "| narrow | 2 of 2 evaluated |" in report
    quality = report.split("## Quality with 1 input view")[1].split("##")[0]
    assert "±" in quality
    comparison = report.split("## Paired comparison with narrow")[1]
    assert "| wide |" in comparison and "[" in comparison
    assert "For each seed, all runs were trained on identical data" in report

    with open(suite.directory / "summary.csv") as file:
        rows = list(csv.DictReader(file))
    assert [(row["run"], row["seed"]) for row in rows] == [("narrow", "0"), ("narrow", "1"), ("wide", "0"), ("wide", "1")]
    with open(suite.directory / "paired_comparison.csv") as file:
        comparisons = list(csv.DictReader(file))
    assert {row["run"] for row in comparisons} == {"wide"}
    assert all(row["reference"] == "narrow" and int(row["num_scenes"]) == 2 for row in comparisons)


def test_single_seed_report_also_compares_runs_scene_by_scene(tmp_path: Path):
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))
    run_suite(suite)

    report = (suite.directory / "summary.md").read_text()

    assert "## Paired comparison with narrow" in report and "±" not in report


# ---------------------------------------------------------------- follow-up review (#2)


def _write_srn_suite(path: Path, runs: dict) -> Path:
    content = {
        "suite_name": "srn_suite",
        "stages": ["train", "evaluate"],
        "common": {"base": str(CONFIG_DIR / "smoke_test.yaml")},
        "runs": runs,
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


def _run_cli_in_new_process(suite_file: Path, overrides: list[str]) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "scene_recon", "run-suite", "--config", str(suite_file), "--set", *overrides]
    return subprocess.run(command, capture_output=True, text=True, timeout=600)


def test_a_finished_run_is_not_reused_after_its_training_data_changed(tmp_path: Path):
    """R1: the review's reproduction, with two separate command-line processes."""
    splits = write_srn_splits(tmp_path / "srn")
    overrides = srn_overrides(splits, tmp_path / "runs")
    first = _run_cli_in_new_process(_write_srn_suite(tmp_path / "suite.yaml", {"run_a": {}}), overrides)
    assert first.returncode == 0, first.stderr

    # Change the training data at the same path, as replacing a dataset copy would.
    intrinsics = splits["train"] / "car_a" / "intrinsics.txt"
    intrinsics.write_text(intrinsics.read_text().replace("131.25", "262.5"))
    runs = {"run_a": {}, "run_b": {"model": {"decoder": {"hidden_width": 32}}}}
    second = _run_cli_in_new_process(_write_srn_suite(tmp_path / "suite.yaml", runs), overrides)

    assert second.returncode != 0
    assert "run_a" in second.stderr and "trained on different data" in second.stderr
    assert not (tmp_path / "runs" / "srn_suite" / "run_b" / "checkpoints").exists()


def test_report_warns_when_checkpoints_record_different_training_data(smoke_config, monkeypatch):
    from scene_recon.config import parse_config
    from scene_recon.data.synthetic import SyntheticSpheresDataset
    from scene_recon.training import Trainer

    def config_for(run_name: str):
        return parse_config({**smoke_config.raw, "experiment": {**smoke_config.raw["experiment"], "run_name": run_name}})

    Trainer(config_for("before")).train()
    real_fingerprint = SyntheticSpheresDataset.fingerprint
    monkeypatch.setattr(SyntheticSpheresDataset, "fingerprint", lambda self: {**real_fingerprint(self), "version": 2})
    Trainer(config_for("after")).train()

    groups = {name: {name: config_for(name)} for name in ("before", "after")}
    write_report("mixed", smoke_config.run_dir.parent, groups, {}, ["train", "evaluate", "profile"])

    report = (smoke_config.run_dir.parent / "summary.md").read_text()
    assert "Warning: the runs' checkpoints record different training data" in report
    assert "| training data recorded in the checkpoint |" in report
    assert "trained on identical data" not in report


def test_suites_reject_invalid_group_overrides_before_anything_runs(tmp_path: Path):
    runs = {**TWO_RUNS, "nan_lr": {"optimization": {"group_overrides": {"encoder": {"learning_rate": float("nan")}}}}}

    with pytest.raises(ConfigError, match="(?s)nan_lr.*must be a finite number"):
        load_suite(_write_suite(tmp_path, runs))


# ---------------------------------------------------------------- suite outputs and run names (R9)


@pytest.mark.parametrize(
    "name", ["summary.md", "Summary.MD", "summary.csv", "paired_comparison.csv", "plots", "summary.md.tmp"]
)
def test_run_names_that_collide_with_suite_outputs_are_rejected(tmp_path: Path, name: str):
    with pytest.raises(ConfigError, match="is reserved"):
        load_suite(_write_suite(tmp_path, {name: {}}))


def test_a_run_named_like_the_suite_log_is_rejected_by_the_command_line(tmp_path: Path):
    from scene_recon import cli

    suite_file = _write_suite(tmp_path, {"suite.log": {}})

    with pytest.raises(SystemExit, match="is reserved"):
        cli.main(["run-suite", "--config", str(suite_file)])
    assert not (tmp_path / "runs" / "test_suite" / "suite.log").exists()


def test_a_report_that_keeps_failing_fails_the_suite_after_all_runs(tmp_path: Path, monkeypatch):
    def broken_report(*args, **kwargs):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(suite_module, "write_report", broken_report)
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))

    with pytest.raises(SuiteReportError, match="disk full") as error:
        run_suite(suite)

    assert error.value.failures == {}
    assert all(RunFiles(config.run_dir).evaluation.is_file() for config in suite.runs.values())


def test_a_report_that_fails_once_does_not_fail_the_suite(tmp_path: Path, monkeypatch):
    real_write_report, calls = suite_module.write_report, []

    def flaky_report(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("temporary failure (simulated)")
        return real_write_report(*args, **kwargs)

    monkeypatch.setattr(suite_module, "write_report", flaky_report)
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))

    assert run_suite(suite) == {}
    assert (suite.directory / "summary.md").is_file()


# ---------------------------------------------------------------- results of unfinished training (#3 N3)


def _fail_training_of(monkeypatch, run_name: str) -> None:
    """Make training of one run fail before it writes a new checkpoint (a simulated crash)."""
    real_training = suite_module.run_training

    def training(config, *args, **kwargs):
        if config.experiment.run_name == run_name:
            raise RuntimeError("training crashed (simulated)")
        return real_training(config, *args, **kwargs)

    monkeypatch.setattr(suite_module, "run_training", training)


def _comparison_rows(suite) -> list[dict]:
    with open(suite.directory / "paired_comparison.csv") as file:
        return list(csv.DictReader(file))


def test_a_failed_training_extension_is_not_compared_as_a_finished_experiment(tmp_path: Path, monkeypatch):
    """The review's scenario: both runs extended from 4 to 6 steps; the reference's extension fails."""
    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS)))
    _fail_training_of(monkeypatch, "narrow")
    extended = load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["training.num_steps=6"])

    assert list(run_suite(extended)) == ["narrow"]

    report = (extended.directory / "summary.md").read_text()
    assert "| narrow | failed | 4 of 6 |" in report and "| wide | evaluated | 6 |" in report
    quality = report.split("## Quality with 1 input view")[1].split("##")[0]
    assert " *" in quality.split("| narrow |")[1].split("\n")[0]  # the step-4 score, marked interim
    comparison = report.split("## Paired comparison with narrow")[1].split("## ")[0]
    assert "Not available yet" in comparison
    assert "evaluated at step 4, but 6 training steps were requested and training failed" in comparison
    assert _comparison_rows(extended) == []

    # Once the extension succeeds, the ordinary comparison is back.
    monkeypatch.undo()
    assert run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["training.num_steps=6"])) == {}
    assert {row["run"] for row in _comparison_rows(extended)} == {"wide"}


def test_a_run_trained_past_the_requested_steps_is_not_compared(tmp_path: Path):
    """The review's scenario: one run was trained longer by hand; the suite asks for the shorter length again."""
    longer = {**TWO_RUNS, "wide": {**TWO_RUNS["wide"], "training": {"num_steps": 6}}}
    run_suite(load_suite(_write_suite(tmp_path, longer)))

    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))
    assert run_suite(suite) == {}

    assert _comparison_rows(suite) == []
    report = (suite.directory / "summary.md").read_text()
    assert "- **wide**: evaluated at step 6, past the 4 training steps requested" in report


def test_figures_of_an_earlier_report_are_not_left_behind(tmp_path: Path):
    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS), overrides=["training.num_steps=6"]))
    assert len(list((tmp_path / "runs" / "test_suite" / "plots").iterdir())) > 1

    (tmp_path / "runs" / "test_suite" / "plots" / "renders_annotated_source_views.png").write_bytes(b"a teammate's")

    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))  # 4 steps: both runs are past it now, so neither is scored
    run_suite(suite)

    # The report's own figures it has no data for are gone; a file it did not make stays.
    plots = {path.name for path in (suite.directory / "plots").iterdir()}
    assert plots == {"training_curves.png", "renders_annotated_source_views.png"}


def test_an_unfinished_candidate_is_listed_as_not_compared(tmp_path: Path, monkeypatch):
    runs = {**TWO_RUNS, "third": {"model": {"decoder": {"hidden_width": 24}}}}
    run_suite(load_suite(_write_suite(tmp_path, runs)))
    _fail_training_of(monkeypatch, "wide")
    extended = load_suite(_write_suite(tmp_path, runs), overrides=["training.num_steps=6"])

    run_suite(extended)

    assert {row["run"] for row in _comparison_rows(extended)} == {"third"}
    report = (extended.directory / "summary.md").read_text()
    assert "Not compared, because they are not at their requested number of training steps" in report
    assert "- **wide**: evaluated at step 4, but 6 training steps were requested and training failed" in report


def test_evaluating_existing_checkpoints_only_still_compares_them(tmp_path: Path):
    run_suite(load_suite(_write_suite(tmp_path, TWO_RUNS)))
    # No training requested: the step-4 checkpoints are what is being compared, whatever num_steps says.
    evaluation_only = load_suite(
        _write_suite(tmp_path, TWO_RUNS, stages=["evaluate"]), overrides=["training.num_steps=6"]
    )

    assert run_suite(evaluation_only) == {}
    assert {row["run"] for row in _comparison_rows(evaluation_only)} == {"wide"}


def test_a_profiling_failure_after_a_finished_evaluation_keeps_the_comparison(tmp_path: Path, monkeypatch):
    real_profiling = suite_module.run_profiling

    def profiling(config, *args, **kwargs):
        if config.experiment.run_name == "wide":
            raise RuntimeError("profiling crashed (simulated)")
        return real_profiling(config, *args, **kwargs)

    monkeypatch.setattr(suite_module, "run_profiling", profiling)
    suite = load_suite(_write_suite(tmp_path, TWO_RUNS))

    assert list(run_suite(suite)) == ["wide"]
    assert {row["run"] for row in _comparison_rows(suite)} == {"wide"}


def test_the_proposal_comparison_loads_as_shipped():
    """Review F1: the complete suite must load, with the baselines waiting in comments until they exist."""
    path = CONFIG_DIR / "suites" / "proposal_comparison.yaml"

    suite = load_suite(path)

    assert list(suite.grouped_runs()) == ["convnextv2_mlp", "hiera_mlp", "hiera_nerf"]
    text = path.read_text(encoding="utf-8")
    seeds = yaml.safe_load(text)["seeds"]
    assert set(suite.runs) == {f"{run}/seed_{seed}" for run in suite.grouped_runs() for seed in seeds}
    # Review F52: uncommenting transnerf_unet alone keeps convnextv2_mlp the reference.
    assert text.index("  # pixelnerf_resnet34:") < text.index("  convnextv2_mlp:") < text.index("  # transnerf_unet:")


def test_every_comparison_run_trains_with_the_same_untuned_recipe():
    """The runs of each seed differ in their model alone: the same training, schedule and measurement for all."""
    comparison = load_suite(CONFIG_DIR / "suites" / "proposal_comparison.yaml")

    def everything_but_the_model(config) -> dict:
        return {
            name: value
            for name, value in flatten_settings(config.raw).items()
            if not name.startswith("model.") and name != "experiment.run_name"
        }

    by_seed: dict[int, list[dict]] = {}
    for config in comparison.runs.values():
        by_seed.setdefault(config.experiment.seed, []).append(everything_but_the_model(config))
    for settings in by_seed.values():
        assert len(settings) == len(comparison.grouped_runs()), "every model has every seed"
        assert all(each == settings[0] for each in settings), "the comparison runs differ in more than the model"
    # No part of any model learns at a rate of its own.
    assert all(config.optimization.group_overrides == {} for config in comparison.runs.values())
