"""End-to-end tests: the commands a user runs, on a problem small enough for a CPU."""

import dataclasses
import json
import shutil

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scene_recon import cli
from scene_recon.build import build_model
from scene_recon.checkpoints import load_checkpoint
from scene_recon.config import Config, parse_config
from scene_recon.models.base import ReconstructionModel
from scene_recon.profiling import count_parameters
from scene_recon.registry import MODELS, ConfigError
from scene_recon.stages import check_model, check_run_can_start
from scene_recon.structures import FieldOutput
from scene_recon.training import Trainer


def _with_overrides(config: Config, section: str, **changes) -> Config:
    """A copy of ``config`` with some settings of one section replaced."""
    raw = {**config.raw, section: {**config.raw[section], **changes}}
    return parse_config(raw)


def test_an_interrupted_save_never_leaves_the_progress_of_the_older_checkpoint(tmp_path, monkeypatch):
    """Review F97: interrupted after the checkpoint was replaced, the progress is read from the new checkpoint."""
    from scene_recon import checkpoints

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    path = tmp_path / "checkpoints" / "last.pt"

    def save(step: int) -> None:
        progress = checkpoints.TrainingProgress(step=step, training_seconds=float(step))
        checkpoints.save_checkpoint(path, model, optimizer, scheduler, progress, {}, {}, {})

    save(4)
    real_write = checkpoints.write_atomically

    def interrupted_before_the_progress_file(target, write):
        if target == checkpoints.progress_file(path):
            raise KeyboardInterrupt
        real_write(target, write)

    monkeypatch.setattr(checkpoints, "write_atomically", interrupted_before_the_progress_file)
    with pytest.raises(KeyboardInterrupt):
        save(6)

    assert checkpoints.read_progress(path).step == 6


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



@pytest.mark.parametrize(("split", "stages"), [("val", ["train", "evaluate", "profile"]), ("train", ["train"])])
def test_the_model_is_checked_at_the_image_size_of_every_split(smoke_config: Config, split: str, stages: list[str]):
    """The review's reproduction: one split's images have a size the encoder's stride does not divide.

    Before, only one split's size was checked (the test split's overrode the
    validation split's, and the training split's was never checked), so the
    run passed preflight and failed at its first validation or batch.
    """
    config = _with_changes(smoke_config, **{f"data_{split}": {"image_size": 17}})  # tiny_conv needs even sides

    with pytest.raises(ConfigError, match=rf"fails on the {split} images \(17x17\).*multiples of 2"):
        check_run_can_start(config, stages)
    with pytest.raises(ConfigError, match=rf"fails on the {split} images \(17x17\)"):
        check_model(config, stages=stages)



class _TrainsOnLargerImagesOnly(ReconstructionModel):
    """Trains only on images of at least 16x16 pixels, but runs in evaluation mode on any size.

    Like a network with batch or instance statistics, which needs more than
    one feature cell per image to train but not to evaluate.
    """

    def __init__(self) -> None:
        super().__init__()
        self.color = nn.Parameter(torch.zeros(3))
        self.density = nn.Parameter(torch.zeros(()))

    def encode_scene(self, source_views):
        if self.training and min(source_views.image_size) < 16:
            raise ValueError("cannot train on images smaller than 16x16")
        return None

    def query_field(self, scene_encoding, points, view_directions, level):
        shape = points.shape[:-1]
        return FieldOutput(rgb=torch.sigmoid(self.color).expand(*shape, 3), density=F.softplus(self.density).expand(shape))


def test_only_the_training_split_is_checked_in_training_mode(smoke_config: Config, monkeypatch: pytest.MonkeyPatch):
    """Validation and test images are used in evaluation mode only, so that is how they are checked."""
    # Registered for this test only, so that no other test can build it by name.
    monkeypatch.setitem(MODELS._classes, "trains_on_larger_images_only", _TrainsOnLargerImagesOnly)
    model_config = parse_config({**smoke_config.raw, "model": {"name": "trains_on_larger_images_only"}})
    config = _with_changes(model_config, data_val={"image_size": 8})  # training and test images stay 16x16

    check_run_can_start(config, ["train", "evaluate", "profile"])
    check_model(config)

    small_training_images = _with_changes(config, data_train={"image_size": 8})
    with pytest.raises(ConfigError, match=r"fails on the train and val images \(8x8\).*cannot train"):
        check_run_can_start(small_training_images, ["train", "evaluate"])


# ---------------------------------------------------------------- profiling validates only what it renders (#3 N1)


def _srn_config_with_test_views(tmp_path, views_per_test_scene: list[int], extra_overrides: list[str]) -> Config:
    """An on-disk SRN config whose test scenes have the given numbers of views (scene A first)."""
    from conftest import CONFIG_DIR, srn_overrides, write_srn_scene, write_srn_splits
    from scene_recon.config import load_config

    splits = write_srn_splits(tmp_path / "srn", num_views=6)
    shutil.rmtree(splits["test"])
    for scene, num_views in zip(("car_a", "car_b"), views_per_test_scene):
        write_srn_scene(splits["test"] / scene, num_views=num_views, image_size=8)
    overrides = srn_overrides(splits, tmp_path / "runs") + extra_overrides
    return load_config(CONFIG_DIR / "smoke_test.yaml", overrides=overrides)


def test_profiling_ignores_later_scenes_it_never_renders(tmp_path):
    """The review's example: scene A has 3 views, scene B 2; source views [0, 1] for both scenes."""
    from scene_recon.stages import run_profiling

    config = _srn_config_with_test_views(
        tmp_path, [3, 2], ["evaluation.source_view_sets=[[0, 1]]", "evaluation.num_scenes=2"]
    )

    check_run_can_start(config, ["profile"])
    assert run_profiling(config)["image_size"] == [8, 8]
    with pytest.raises(ConfigError, match="uses every view of scene 1"):
        check_run_can_start(config, ["evaluate"])
    with pytest.raises(ConfigError, match="uses every view of scene 1"):
        check_run_can_start(config, ["evaluate", "profile"])


def test_profiling_still_needs_a_target_view_in_the_first_scene(tmp_path):
    config = _srn_config_with_test_views(tmp_path, [2, 6], ["evaluation.source_view_sets=[[0, 1]]"])

    with pytest.raises(ConfigError, match=r"profiling.*uses every view of scene 0"):
        check_run_can_start(config, ["profile"])


def test_unused_later_protocols_do_not_affect_profiling(tmp_path):
    # The second protocol uses every view of scene A, so it cannot be evaluated, but profiling never uses it.
    config = _srn_config_with_test_views(tmp_path, [3, 3], ["evaluation.source_view_sets=[[0], [0, 1, 2]]"])

    check_run_can_start(config, ["profile"])
    with pytest.raises(ConfigError, match="uses every view"):
        check_run_can_start(config, ["evaluate"])
