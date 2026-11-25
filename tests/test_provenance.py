"""A result must describe the experiment that produced it.

Each test reproduces a scenario from the repository review in which a result
could be attributed to the wrong training settings, checkpoint or device.
"""

from pathlib import Path

import pytest
import torch

from scene_recon import evaluation as evaluation_module
from scene_recon import provenance as provenance_module
from scene_recon.checkpoints import COMPATIBILITY_VERSION
from scene_recon.config import Config, parse_config
from scene_recon.data.synthetic import SyntheticSpheresDataset
from scene_recon.provenance import (
    evaluation_status,
    profile_status,
    profiling_settings,
    training_identity,
)
from scene_recon.registry import ConfigError
from scene_recon.run_files import RunFiles
from scene_recon.stages import run_evaluation, run_profiling
from scene_recon.training import Trainer

from conftest import write_srn_image


def _changed(config: Config, section: str, **changes) -> Config:
    """A copy of ``config`` with some settings of one section replaced."""
    return parse_config({**config.raw, section: {**config.raw[section], **changes}})


def _with_normalization(config: Config, name: str) -> Config:
    model = {**config.raw["model"], "encoder": {**config.raw["model"]["encoder"], "input_normalization": name}}
    return parse_config({**config.raw, "model": model})


# ---------------------------------------------------------------- training identity (F1)


@pytest.mark.parametrize(
    "change",
    [
        lambda config: _changed(config, "optimization", learning_rate=0.01),
        lambda config: _changed(config, "experiment", seed=99),
        lambda config: _changed(config, "training", rays_per_scene=64),
        lambda config: _changed(config, "renderer", num_coarse_samples=16),
        lambda config: _with_normalization(config, "imagenet"),
    ],
    ids=["learning_rate", "seed", "rays_per_scene", "renderer", "input_normalization"],
)
def test_resuming_with_other_training_settings_is_refused(smoke_config: Config, change):
    Trainer(smoke_config).train()
    files = RunFiles(smoke_config.run_dir)
    checkpoint_before, config_before = files.checkpoint.read_bytes(), files.config.read_text()
    changed = _changed(change(smoke_config), "training", num_steps=smoke_config.training.num_steps + 2)

    with pytest.raises(ConfigError, match="was trained with different settings"):
        Trainer(changed, resume=True)

    # Nothing about the run was rewritten to claim the new settings.
    assert files.checkpoint.read_bytes() == checkpoint_before
    assert files.config.read_text() == config_before


def test_continuation_settings_may_change(smoke_config: Config):
    Trainer(smoke_config).train()
    longer = _changed(smoke_config, "training", num_steps=smoke_config.training.num_steps + 2, log_every=1)
    longer = _changed(longer, "evaluation", metrics=["psnr"])

    trainer = Trainer(longer, resume=True)
    trainer.train()

    assert trainer.progress.step == longer.training.num_steps


def test_evaluating_a_checkpoint_with_other_model_settings_is_refused(smoke_config: Config):
    Trainer(smoke_config).train()

    with pytest.raises(ConfigError, match="model.encoder.input_normalization: trained with none, requested imagenet"):
        run_evaluation(_with_normalization(smoke_config, "imagenet"))


def test_training_identity_ignores_only_continuation_settings(smoke_config: Config):
    identity = training_identity(smoke_config.raw)

    assert "optimization.learning_rate" in identity and "experiment.seed" in identity
    assert "model.decoder.hidden_width" in identity and "data.train.seed" in identity
    assert "training.num_steps" not in identity and "evaluation.metrics" not in identity
    assert "experiment.device" not in identity and "data.test.seed" not in identity


# ---------------------------------------------------------------- evaluation freshness (F2)


def test_an_evaluation_of_another_checkpoint_at_the_same_step_is_not_reused(smoke_config: Config, tmp_path: Path):
    Trainer(smoke_config).train()
    files = RunFiles(smoke_config.run_dir)

    # A second checkpoint of this run at the same step, with different weights.
    state = torch.load(files.checkpoint, weights_only=True)
    state["model"] = {name: value + 0.05 if value.is_floating_point() else value for name, value in state["model"].items()}
    alternate = tmp_path / "alternate.pt"
    torch.save(state, alternate)

    run_evaluation(smoke_config)
    assert evaluation_status(smoke_config).is_current

    run_evaluation(smoke_config, alternate)  # writes its results into the run directory
    status = evaluation_status(smoke_config)

    assert status.state == "stale" and "different checkpoint" in status.reason


def test_an_evaluation_with_other_settings_is_stale(smoke_config: Config):
    Trainer(smoke_config).train()
    run_evaluation(smoke_config)

    status = evaluation_status(_changed(smoke_config, "evaluation", source_view_sets=[[0, 6]]))

    assert status.state == "stale" and "different evaluation" in status.reason


def test_a_damaged_evaluation_file_is_reported_as_unreadable(smoke_config: Config):
    Trainer(smoke_config).train()
    run_evaluation(smoke_config)
    RunFiles(smoke_config.run_dir).evaluation.write_text('{"checkpoint":')

    assert evaluation_status(smoke_config).state == "unreadable"


def test_nan_metrics_are_never_written_as_results(smoke_config: Config, monkeypatch):
    Trainer(smoke_config).train()
    monkeypatch.setattr(evaluation_module.Evaluator, "evaluate", _evaluate_with_nan(evaluation_module.Evaluator.evaluate))

    with pytest.raises(RuntimeError, match="non-finite"):
        run_evaluation(smoke_config)
    assert not RunFiles(smoke_config.run_dir).evaluation.exists()


def _evaluate_with_nan(real_evaluate):
    def evaluate(self, *args, **kwargs):
        result = real_evaluate(self, *args, **kwargs)
        result.mean["psnr"] = float("nan")
        return result

    return evaluate


# ---------------------------------------------------------------- profiling freshness (F9)


def test_a_profile_from_another_device_is_stale(smoke_config: Config):
    run_profiling(smoke_config)
    assert profile_status(smoke_config).is_current

    on_gpu = _changed(smoke_config, "experiment", device="cuda:0")

    assert profiling_settings(on_gpu) != profiling_settings(smoke_config)
    assert profile_status(on_gpu).state == "stale"


# ---------------------------------------------------------------- dataset identity (I4)


def _pretend_the_data_changed(monkeypatch) -> None:
    """Same config, different content: as if the dataset directory had been replaced."""
    real_fingerprint = SyntheticSpheresDataset.fingerprint

    def changed_fingerprint(self):
        fingerprint = real_fingerprint(self)
        return {**fingerprint, "seed": fingerprint["seed"] + 1000}  # same format, other content

    monkeypatch.setattr(SyntheticSpheresDataset, "fingerprint", changed_fingerprint)


def test_an_evaluation_on_other_test_data_is_stale(smoke_config: Config, monkeypatch):
    Trainer(smoke_config).train()
    run_evaluation(smoke_config)

    _pretend_the_data_changed(monkeypatch)

    status = evaluation_status(smoke_config)
    assert status.state == "stale" and "other test data" in status.reason


def test_resuming_on_other_training_data_is_refused(smoke_config: Config, monkeypatch):
    Trainer(smoke_config).train()
    _pretend_the_data_changed(monkeypatch)

    with pytest.raises(ConfigError, match="trained on different data"):
        Trainer(_changed(smoke_config, "training", num_steps=smoke_config.training.num_steps + 2), resume=True)


def test_results_record_the_data_they_came_from(smoke_config: Config):
    Trainer(smoke_config).train()
    report = run_evaluation(smoke_config)
    state = torch.load(RunFiles(smoke_config.run_dir).checkpoint, weights_only=True)

    expected = {"dataset": "synthetic_spheres", "num_scenes": 32, "num_views": 12, "image_size": 16, "seed": 0}
    assert state["training_data"] == expected
    assert report["settings"]["test_data_fingerprint"]["seed"] == 2


# ---------------------------------------------------------------- profiling workload (R8)


def test_a_profile_is_stale_after_the_images_are_replaced_by_larger_ones(srn_config: Config):
    """The review's reproduction: 8x8 images replaced by 16x16 ones at the same paths, config unchanged."""
    run_profiling(srn_config)
    assert profile_status(srn_config).is_current

    test_dir = Path(srn_config.data["test"]["split_dir"])
    for image in test_dir.glob("*/rgb/*.png"):
        write_srn_image(image, image_size=16, grey_level=50)

    status = profile_status(srn_config)
    assert status.state == "stale" and "test data" in status.reason
    assert run_profiling(srn_config)["image_size"] == [16, 16]
    assert profile_status(srn_config).is_current


# ---------------------------------------------------------------- older fingerprint formats (#3 compatibility note)


def test_a_checkpoint_with_an_older_fingerprint_format_is_explained_not_called_different_data(srn_config: Config):
    from scene_recon.stages import check_run_can_start

    Trainer(srn_config).train()
    checkpoint = RunFiles(srn_config.run_dir).checkpoint
    state = torch.load(checkpoint, weights_only=True)
    # The format of the previous version: names and sizes only, which cannot confirm the content.
    state["training_data"] = {"dataset": "srn", "num_scenes": 2, "files_sha256": "0" * 64}
    torch.save(state, checkpoint)

    with pytest.raises(ConfigError, match="older fingerprint format") as error:
        check_run_can_start(srn_config, ["train"])
    assert "different data" not in str(error.value)
    with pytest.raises(ConfigError, match="older fingerprint format"):
        Trainer(_changed(srn_config, "training", num_steps=srn_config.training.num_steps + 1), resume=True)

    run_evaluation(srn_config)  # evaluation does not need the training data
    assert evaluation_status(srn_config).is_current


# ---------------------------------------------------------------- code compatibility (review F19)


def _rewrite_checkpoint_version(checkpoint: Path, version: int | None) -> None:
    """Make a checkpoint look as if it was written by another version of the code (None: before versions existed)."""
    state = torch.load(checkpoint, weights_only=True)
    if version is None:
        del state["compatibility_version"]
    else:
        state["compatibility_version"] = version
    torch.save(state, checkpoint)


@pytest.mark.parametrize("version", [None, COMPATIBILITY_VERSION - 1], ids=["no_version", "older_version"])
def test_a_checkpoint_from_older_code_is_refused_everywhere(srn_config: Config, version):
    from scene_recon.stages import check_run_can_start

    Trainer(srn_config).train()
    checkpoint = RunFiles(srn_config.run_dir).checkpoint
    _rewrite_checkpoint_version(checkpoint, version)

    with pytest.raises(ConfigError, match="older version of the code") as error:
        check_run_can_start(srn_config, ["train", "evaluate", "profile"])
    assert "Train it again under a new name" in str(error.value) and "delete" not in str(error.value)
    with pytest.raises(ConfigError, match="older version of the code"):
        Trainer(_changed(srn_config, "training", num_steps=srn_config.training.num_steps + 1), resume=True)
    with pytest.raises(ConfigError, match="older version of the code"):
        run_evaluation(srn_config)
    with pytest.raises(ConfigError, match="older version of the code"):
        run_profiling(srn_config)


def test_a_checkpoint_of_the_retired_version_2_is_refused(smoke_config: Config):
    """Review F64: version 2 was written by intermediate commits whose code trained differently."""
    from scene_recon.stages import check_run_can_start

    assert COMPATIBILITY_VERSION > 2
    Trainer(smoke_config).train()
    _rewrite_checkpoint_version(RunFiles(smoke_config.run_dir).checkpoint, 2)

    with pytest.raises(ConfigError, match="compatibility version 2"):
        check_run_can_start(smoke_config, ["train"])


def test_a_checkpoint_from_newer_code_is_refused(smoke_config: Config):
    Trainer(smoke_config).train()
    _rewrite_checkpoint_version(RunFiles(smoke_config.run_dir).checkpoint, COMPATIBILITY_VERSION + 1)

    with pytest.raises(ConfigError, match="newer version of the code"):
        run_evaluation(smoke_config)


def test_results_from_older_code_are_out_of_date(smoke_config: Config, monkeypatch):
    Trainer(smoke_config).train()
    run_evaluation(smoke_config)
    run_profiling(smoke_config)
    assert evaluation_status(smoke_config).is_current and profile_status(smoke_config).is_current

    # The code moves on to the next version; the results on disk were made with the previous one.
    monkeypatch.setattr(provenance_module, "COMPATIBILITY_VERSION", COMPATIBILITY_VERSION + 1)

    for status in (evaluation_status(smoke_config), profile_status(smoke_config)):
        assert status.state == "stale" and "older version of the code" in status.reason


# ---------------------------------------------------------------- starting weights (review B(b))


def test_checkpoints_record_the_pretrained_weights_the_run_started_from(smoke_config: Config, monkeypatch):
    from scene_recon.checkpoints import read_checkpoint_starting_weights
    from scene_recon.models.testing.tiny_encoder import TinyConvEncoder

    checkpoint = RunFiles(smoke_config.run_dir).checkpoint
    first = {"source": "weights/encoder.pth", "sha256": "1" * 64}
    monkeypatch.setattr(TinyConvEncoder, "pretrained_weights", first)
    Trainer(smoke_config).train()
    assert read_checkpoint_starting_weights(checkpoint) == {"encoder": first}

    # The weights file is replaced before the run is resumed; the run still started from the first one.
    monkeypatch.setattr(TinyConvEncoder, "pretrained_weights", {"source": "weights/encoder.pth", "sha256": "2" * 64})
    Trainer(_changed(smoke_config, "training", num_steps=smoke_config.training.num_steps + 2), resume=True).train()
    assert read_checkpoint_starting_weights(checkpoint) == {"encoder": first}


@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_a_weights_record_a_checkpoint_could_not_hold_is_refused_before_training(
    smoke_config: Config, monkeypatch, part: str
):
    """Review F44 and F58: suite preflight and check-model find it in any part of the model."""
    from scene_recon.models.testing.tiny_decoder import TinyPooledMLPDecoder
    from scene_recon.models.testing.tiny_encoder import TinyConvEncoder
    from scene_recon.stages import check_model, check_run_can_start

    owner = {"encoder": TinyConvEncoder, "decoder": TinyPooledMLPDecoder}[part]
    monkeypatch.setattr(owner, "pretrained_weights", {"source": Path("weights/part.pth"), "sha256": "1" * 64}, raising=False)

    with pytest.raises(ConfigError, match="has pretrained_weights"):
        check_run_can_start(smoke_config, ["train"])
    with pytest.raises(ConfigError, match="has pretrained_weights"):
        check_model(smoke_config, stages=["train"])
    assert not RunFiles(smoke_config.run_dir).checkpoint.exists()


def test_a_model_that_starts_from_random_values_records_no_pretrained_weights(smoke_config: Config):
    from scene_recon.checkpoints import read_checkpoint_starting_weights

    Trainer(smoke_config).train()

    assert read_checkpoint_starting_weights(RunFiles(smoke_config.run_dir).checkpoint) == {}


# ---------------------------------------------------------------- software versions (review F13)


def test_every_evaluation_records_the_packages_that_compute_its_metrics():
    """Review F53: otherwise the report could not tell that two runs were scored with different versions."""
    from scene_recon.environment import environment
    from scene_recon.registry import METRICS
    from scene_recon.report import metric_software

    recorded = environment()

    for metric in METRICS.names():
        assert set(metric_software(metric)) <= recorded.keys(), metric
    assert "scikit-image" in metric_software("ssim") and "lpips" in metric_software("lpips")


def test_the_timm_version_is_recorded_and_decides_whether_a_profile_is_current(smoke_config: Config, monkeypatch):
    import timm

    from scene_recon import environment as environment_module

    assert environment_module.environment()["timm"] == timm.__version__
    run_profiling(smoke_config)
    assert profile_status(smoke_config).is_current

    real_version = environment_module.package_version
    monkeypatch.setattr(
        environment_module, "package_version", lambda name: "0.0.1" if name == "timm" else real_version(name)
    )

    assert profile_status(smoke_config).state == "stale"
