"""End-to-end tests: the commands a user runs, on a problem small enough for a CPU."""

import dataclasses
import json

import pytest

from scene_recon import cli
from scene_recon.build import build_model
from scene_recon.checkpoints import load_checkpoint
from scene_recon.config import Config, parse_config
from scene_recon.profiling import count_parameters
from scene_recon.registry import ConfigError
from scene_recon.training import Trainer


def _with_overrides(config: Config, section: str, **changes) -> Config:
    """A copy of ``config`` with some settings of one section replaced."""
    raw = {**config.raw, section: {**config.raw[section], **changes}}
    return parse_config(raw)


def test_training_writes_a_resumable_checkpoint(smoke_config: Config):
    Trainer(smoke_config).train()

    checkpoint_path = smoke_config.run_dir / "checkpoints" / "last.pt"
    progress = load_checkpoint(checkpoint_path, build_model(smoke_config))
    assert progress.step == smoke_config.training.num_steps
    assert progress.training_seconds > 0
    assert (smoke_config.run_dir / "config.yaml").is_file()
    assert any((smoke_config.run_dir / "tensorboard").iterdir())


def test_resuming_continues_from_the_saved_step(smoke_config: Config):
    Trainer(smoke_config).train()
    longer_config = _with_overrides(smoke_config, "training", num_steps=smoke_config.training.num_steps + 3)

    trainer = Trainer(longer_config, resume=True)
    assert trainer.progress.step == smoke_config.training.num_steps
    trainer.train()

    assert trainer.progress.step == longer_config.training.num_steps


def test_starting_over_an_existing_run_is_refused(smoke_config: Config):
    Trainer(smoke_config).train()

    with pytest.raises(ConfigError, match="already exists"):
        Trainer(smoke_config)


def test_unusable_validation_views_are_reported_before_training_starts(smoke_config: Config):
    config = _with_overrides(smoke_config, "validation", source_views=[64])

    with pytest.raises(ConfigError, match="validation.source_views"):
        Trainer(config)


def test_training_improves_the_model(smoke_config: Config):
    """Fitting a single scene for a few hundred steps must clearly improve its novel views."""
    config = _with_overrides(smoke_config, "training", num_steps=300, rays_per_scene=128, validate_every=1000)
    config = _with_overrides(config, "data", train={**config.data["train"], "num_scenes": 1})
    trainer = Trainer(config)

    def psnr_on_the_training_scene() -> float:
        result = trainer.evaluator.evaluate(trainer.train_scenes, source_view_indices=[0], num_target_views=3)
        return result.mean["psnr"]

    psnr_before = psnr_on_the_training_scene()
    trainer.train()
    psnr_after = psnr_on_the_training_scene()

    assert psnr_after > psnr_before + 3.0


def test_command_line_runs_train_evaluate_and_profile(smoke_config: Config, tmp_path):
    # The command line reads configs from files, so the fixture is written to disk first.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(smoke_config.raw))  # JSON is valid YAML

    cli.main(["check-model", "--config", str(config_path)])
    cli.main(["train", "--config", str(config_path)])
    cli.main(["evaluate", "--config", str(config_path), "--set", "evaluation.save_images=true"])
    cli.main(["profile", "--config", str(config_path), "--num-frames", "1"])

    evaluation = json.loads((smoke_config.run_dir / "evaluation.json").read_text())
    assert [protocol["source_views"] for protocol in evaluation["protocols"]] == [[0], [0, 6]]
    for protocol in evaluation["protocols"]:
        assert set(protocol["mean"]) == {"psnr", "ssim"}
        assert len(protocol["per_scene"]) == smoke_config.evaluation.num_scenes

    rendered_images = list((smoke_config.run_dir / "renders" / "1_source_views").glob("*/*.png"))
    assert len(rendered_images) == smoke_config.evaluation.num_scenes * smoke_config.evaluation.num_target_views

    profile = json.loads((smoke_config.run_dir / "profile.json").read_text())
    assert profile["parameters"] == count_parameters(build_model(smoke_config))
    assert profile["rendering"]["frames_per_second"] > 0


def test_command_line_reports_config_mistakes_without_a_stack_trace(smoke_config: Config, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(smoke_config.raw))

    with pytest.raises(SystemExit, match="Config error: Unknown model 'no_such_model'"):
        cli.main(["check-model", "--config", str(config_path), "--set", "model.name=no_such_model"])


def test_config_dataclasses_are_immutable(smoke_config: Config):
    with pytest.raises(dataclasses.FrozenInstanceError):
        smoke_config.training.batch_size = 99
