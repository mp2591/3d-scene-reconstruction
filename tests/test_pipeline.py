"""End-to-end tests: the commands a user runs, on a problem small enough for a CPU."""

import dataclasses
import json
import shutil

import pytest
import torch

from scene_recon import cli
from scene_recon.build import build_model
from scene_recon.checkpoints import load_checkpoint
from scene_recon.config import Config, parse_config
from scene_recon.profiling import count_parameters
from scene_recon.registry import ConfigError
from scene_recon.stages import check_run_can_start
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
    cli.main(["profile", "--config", str(config_path), "--set", "profiling.num_frames=1", "profiling.num_warmup_frames=0"])

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


def test_training_stops_before_a_non_finite_loss_changes_the_weights(smoke_config: Config):
    from scene_recon.run_files import RunFiles

    trainer = Trainer(smoke_config)
    real_loss = trainer.loss_function

    def nan_loss(renders, target_rgb):
        terms = real_loss(renders, target_rgb)
        return {**terms, "total": terms["total"] * float("nan")}

    trainer.loss_function = nan_loss
    weights_before = {name: value.clone() for name, value in trainer.model.state_dict().items()}

    with pytest.raises(RuntimeError, match="training loss is nan"):
        trainer.train()

    assert all(torch.equal(value, weights_before[name]) for name, value in trainer.model.state_dict().items())
    assert not RunFiles(smoke_config.run_dir).checkpoint.exists()


def test_weights_with_nan_are_never_written_to_a_checkpoint(smoke_config: Config):
    from scene_recon.run_files import RunFiles

    trainer = Trainer(smoke_config)
    with torch.no_grad():
        next(trainer.model.parameters()).fill_(float("nan"))

    with pytest.raises(RuntimeError, match="no checkpoint was written"):
        trainer._save_checkpoint()
    assert not RunFiles(smoke_config.run_dir).checkpoint.exists()


# ---------------------------------------------------------------- protocols that cannot run (R7)


def _with_changes(config: Config, **sections) -> Config:
    raw = dict(config.raw)
    for section, changes in sections.items():
        if section.startswith("data_"):
            split = section.removeprefix("data_")
            raw["data"] = {**raw["data"], split: {**raw["data"][split], **changes}}
        else:
            raw[section] = {**raw[section], **changes}
    return parse_config(raw)


def test_an_evaluation_protocol_using_every_view_is_rejected_before_training(smoke_config: Config):
    """The review's reproduction: two test views, both given to the model."""
    config = _with_changes(smoke_config, data_test={"num_views": 2}, evaluation={"source_view_sets": [[0, 1]]})

    with pytest.raises(ConfigError, match="uses every view of scene 0"):
        check_run_can_start(config, ["train", "evaluate"])


def test_a_validation_protocol_using_every_view_is_rejected_before_training(smoke_config: Config):
    config = _with_changes(smoke_config, data_val={"num_views": 1}, validation={"source_views": [0]})

    with pytest.raises(ConfigError, match="validation.source_views.*uses every view"):
        check_run_can_start(config, ["train"])


def test_a_profile_only_run_needs_a_view_to_render(smoke_config: Config):
    config = _with_changes(smoke_config, data_test={"num_views": 1}, evaluation={"source_view_sets": [[0]]})

    with pytest.raises(ConfigError, match="uses every view"):
        check_run_can_start(config, ["profile"])


def test_a_later_scene_with_fewer_views_is_caught(tmp_path):
    from conftest import CONFIG_DIR, srn_overrides, write_srn_scene, write_srn_splits
    from scene_recon.config import load_config

    splits = write_srn_splits(tmp_path / "srn", num_views=6)
    shutil.rmtree(splits["test"] / "car_b")
    write_srn_scene(splits["test"] / "car_b", num_views=2, image_size=8)  # the second scene has 2 views only
    overrides = srn_overrides(splits, tmp_path / "runs") + ["evaluation.source_view_sets=[[0, 1]]"]
    config = load_config(CONFIG_DIR / "smoke_test.yaml", overrides=overrides)

    with pytest.raises(ConfigError, match="uses every view of scene 1"):
        check_run_can_start(config, ["evaluate"])
