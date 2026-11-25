"""The training loop.

One optimisation step:

1. Load a batch of scenes, a few random views each.
2. Pick some of those views as the model's input (the source views).
3. Sample random pixels from all loaded views and cast rays through them.
4. Encode the source views, render the rays, compare with the true pixel colours.
5. Back-propagate and update the weights.

Every random choice in steps 1 to 4 comes from the ``TrainingSchedule``, which
derives it from the seed and the step number alone. Every model trained with
the same seed therefore sees exactly the same data, step by step, and a
resumed run continues exactly where it stopped. PyTorch's global random
generator is left to the model (weight initialisation, dropout, ...).

Progress is logged to the console and to TensorBoard, the model is validated
on a few held-out scenes at regular intervals, and checkpoints allow an
interrupted run to be resumed exactly where it stopped.
"""

from __future__ import annotations

import logging
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from scene_recon.build import (
    apply_float32_matmul_precision,
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    resolve_device,
    seed_everything,
)
from scene_recon.checkpoints import (
    TrainingProgress,
    load_checkpoint,
    read_checkpoint_starting_weights,
    save_checkpoint,
)
from scene_recon.config import Config, save_config
from scene_recon.data import SceneDataset, TrainingSchedule, training_batches
from scene_recon.evaluation import Evaluator, check_protocol_feasible
from scene_recon.metrics import mse_to_psnr
from scene_recon.models.base import pretrained_weights_of
from scene_recon.optim import build_optimizer_and_scheduler
from scene_recon.provenance import check_checkpoint_matches_config, check_training_data_unchanged
from scene_recon.registry import ConfigError
from scene_recon.rendering import VolumeRenderer, final_output, sample_training_rays
from scene_recon.run_files import RunFiles
from scene_recon.structures import SceneViews

logger = logging.getLogger(__name__)


class Trainer:
    """Trains the model described by a config.

    Writes the run's config, checkpoints and TensorBoard logs into the run
    directory; see ``scene_recon.run_files`` for the layout.
    """

    def __init__(self, config: Config, resume: bool = False) -> None:
        """
        Args:
            config: The experiment to run.
            resume: Continue from the run directory's latest checkpoint
                instead of starting from scratch.
        """
        self.config = config
        self.device = resolve_device(config.experiment.device)
        apply_float32_matmul_precision(config)
        seed_everything(config.experiment.seed)

        self.model = build_model(config).to(self.device)
        # Recorded in every checkpoint: which pretrained weights this run started from.
        self.starting_weights = pretrained_weights_of(self.model)
        self.renderer = VolumeRenderer(config.renderer)
        self.loss_function = build_loss(config)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.model, config.optimization)

        self.train_scenes = build_dataset(config, "train")
        self.validation_scenes = build_dataset(config, "val")
        self.schedule = TrainingSchedule(
            self.train_scenes,
            seed=config.experiment.seed,
            batch_size=config.training.batch_size,
            views_per_scene=config.training.views_per_scene,
            num_source_views=config.training.num_source_views,
        )
        self.evaluator = Evaluator(
            self.model, self.renderer, build_metrics(config.validation.metrics), self.device
        )
        check_training_settings(config, self.validation_scenes)

        self.files = RunFiles(config.run_dir)
        self.checkpoint_path = self.files.checkpoint
        self.progress = TrainingProgress()
        if resume:
            # Continuing with other training settings would leave weights that
            # no config describes; only e.g. the step budget may change.
            check_checkpoint_matches_config(self.checkpoint_path, config)
            check_training_data_unchanged(self.checkpoint_path, self.train_scenes)
            self.progress = load_checkpoint(self.checkpoint_path, self.model, self.optimizer, self.scheduler)
            # The run started from the weights its first checkpoint recorded, not
            # from whatever the pretrained file holds today.
            self.starting_weights = read_checkpoint_starting_weights(self.checkpoint_path)
            logger.info("Resumed from %s at step %d", self.checkpoint_path, self.progress.step)
        elif self.checkpoint_path.exists():
            raise ConfigError(
                f"{self.checkpoint_path} already exists. Pass --resume to continue that run, or change "
                "experiment.run_name to start a new one. Refusing to overwrite a trained model."
            )

        config.run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.files.config)
        self.writer = SummaryWriter(str(self.files.tensorboard))

    def train(self) -> None:
        """Run optimisation steps until ``training.num_steps`` is reached."""
        settings = self.config.training
        batches = training_batches(self.schedule, start_step=self.progress.step, num_workers=settings.num_workers)
        logger.info("Training on %s for %d steps", self.device, settings.num_steps)

        try:
            while self.progress.step < settings.num_steps:
                step_start = time.perf_counter()
                batch = next(batches).to(self.device)
                scalars = self._training_step(batch, self.progress.step)
                self.progress.training_seconds += time.perf_counter() - step_start
                self.progress.step += 1

                step = self.progress.step
                if step % settings.log_every == 0:
                    self._log_training(scalars)
                if step % settings.validate_every == 0:
                    self._validate()
                if step % settings.checkpoint_every == 0:
                    self._save_checkpoint()
            self._save_checkpoint()
        finally:
            # Stops the data-loading worker processes and flushes the logs,
            # also when training stops with an error.
            batches.close()
            self.writer.close()

        logger.info("Finished after %.2f training hours", self.progress.training_seconds / 3600)

    def _training_step(self, batch: SceneViews, step: int) -> dict[str, float]:
        """Perform optimisation step number ``step`` (counted from 0) and return the scalars worth logging."""
        self.model.train()

        source_views = batch.select_views(self.schedule.source_view_indices(step))
        rays, target_rgb = sample_training_rays(
            batch, self.config.training.rays_per_scene, generator=self.schedule.generator("rays", step)
        )

        scene_encoding = self.model.encode_scene(source_views)
        renders = self.renderer.render_rays(
            self.model, scene_encoding, rays, generator=self.schedule.generator("sample_depths", step)
        )
        loss_terms = self.loss_function(renders, target_rgb)

        # Stop before a non-finite value can reach the weights (and from there a checkpoint).
        if not torch.isfinite(loss_terms["total"]):
            raise RuntimeError(
                f"The training loss is {loss_terms['total'].item()} at step {step}. Stopping before the weights "
                "change; check the learning rates and the model's outputs."
            )
        self.optimizer.zero_grad(set_to_none=True)
        loss_terms["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float("inf"))
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(
                f"The gradients are not finite at step {step}. Stopping before the weights change; "
                "check the learning rates and the model's outputs."
            )
        self.optimizer.step()
        self.scheduler.step()

        scalars = {f"loss/{name}": value.item() for name, value in loss_terms.items()}
        final_mse = F.mse_loss(final_output(renders).rgb.detach(), target_rgb)
        scalars["train/psnr"] = mse_to_psnr(final_mse).item()
        return scalars

    def _log_training(self, scalars: dict[str, float]) -> None:
        step = self.progress.step
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
        for group in self.optimizer.param_groups:
            self.writer.add_scalar(f"learning_rate/{group['name']}", group["lr"], step)
        self.writer.add_scalar("time/training_hours", self.progress.training_seconds / 3600, step)

        logger.info(
            "step %d | loss %.5f | psnr %.2f dB", step, scalars["loss/total"], scalars["train/psnr"]
        )

    def _validate(self) -> None:
        """Render a few held-out views of a few validation scenes and log their quality."""
        settings = self.config.validation
        result = self.evaluator.evaluate(
            self.validation_scenes,
            settings.source_views,
            num_scenes=settings.num_scenes,
            num_target_views=settings.num_target_views,
        )

        step = self.progress.step
        for name, value in result.mean.items():
            self.writer.add_scalar(f"validation/{name}", value, step)
        self.writer.add_image("validation/source_prediction_truth", result.previews[0].side_by_side(), step)

        summary = " | ".join(f"{name} {value:.4f}" for name, value in result.mean.items())
        logger.info("step %d | validation | %s", step, summary)

    def _save_checkpoint(self) -> None:
        non_finite = [name for name, parameter in self.model.named_parameters() if not torch.isfinite(parameter).all()]
        if non_finite:
            raise RuntimeError(
                f"{len(non_finite)} parameter tensor(s) contain NaN or infinite values at step {self.progress.step} "
                f"(e.g. {', '.join(non_finite[:3])}); no checkpoint was written."
            )
        save_checkpoint(
            self.checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.progress,
            self.config.raw,
            training_data=self.train_scenes.fingerprint(),
            starting_weights=self.starting_weights,
        )


def check_training_settings(config: Config, validation_scenes: SceneDataset) -> None:
    """Catch training settings that cannot work, before any time is spent on training.

    Raises:
        ConfigError: Describing the first problem found.
    """
    settings = config.training
    if max(settings.num_source_views) > settings.views_per_scene:
        raise ConfigError(
            f"training.num_source_views asks for up to {max(settings.num_source_views)} source views, "
            f"but only training.views_per_scene = {settings.views_per_scene} views are loaded per scene."
        )
    if min(settings.num_source_views) < 1:
        raise ConfigError("training.num_source_views must only contain positive numbers.")

    # Validation runs thousands of steps into training; a mistake here
    # should not cost those steps.
    check_protocol_feasible(
        validation_scenes,
        config.validation.source_views,
        config.validation.num_scenes,
        where="validation.source_views",
    )
