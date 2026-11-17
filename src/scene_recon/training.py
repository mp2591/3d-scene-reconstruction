"""The training loop.

One optimisation step:

1. Load a batch of scenes, a few random views each.
2. Pick some of those views as the model's input (the source views).
3. Sample random pixels from all loaded views and cast rays through them.
4. Encode the source views, render the rays, compare with the true pixel colours.
5. Back-propagate and update the weights.

Progress is logged to the console and to TensorBoard, the model is validated
on a few held-out scenes at regular intervals, and checkpoints allow an
interrupted run to be resumed exactly where it stopped.
"""

from __future__ import annotations

import logging
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from scene_recon.build import (
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    resolve_device,
    seed_everything,
)
from scene_recon.checkpoints import TrainingProgress, load_checkpoint, save_checkpoint
from scene_recon.config import Config, save_config
from scene_recon.data import SceneDataset, endless_training_batches
from scene_recon.evaluation import Evaluator
from scene_recon.metrics import mse_to_psnr
from scene_recon.optim import build_optimizer_and_scheduler
from scene_recon.registry import ConfigError
from scene_recon.rendering import VolumeRenderer, final_output, sample_training_rays
from scene_recon.structures import SceneViews

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "last.pt"


class Trainer:
    """Trains the model described by a config and writes all outputs to the run directory.

    Layout of the run directory (``<output_dir>/<run_name>/``)::

        config.yaml          the full config of the run
        checkpoints/last.pt  the latest model, optimiser and scheduler state
        tensorboard/         training curves and validation renders
    """

    def __init__(self, config: Config, resume: bool = False) -> None:
        """
        Args:
            config: The experiment to run.
            resume: Continue from the run directory's latest checkpoint
                instead of starting from scratch.
        """
        _check_training_settings(config)
        self.config = config
        self.device = resolve_device(config.experiment.device)
        seed_everything(config.experiment.seed)

        self.model = build_model(config).to(self.device)
        self.renderer = VolumeRenderer(config.renderer)
        self.loss_function = build_loss(config)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.model, config.optimization)

        self.train_scenes = build_dataset(config, "train")
        self.validation_scenes = build_dataset(config, "val")
        self.evaluator = Evaluator(
            self.model, self.renderer, build_metrics(config.validation.metrics), self.device
        )
        _check_validation_settings(config, self.validation_scenes)

        self.checkpoint_path = config.run_dir / "checkpoints" / CHECKPOINT_NAME
        self.progress = TrainingProgress()
        if resume:
            self.progress = load_checkpoint(self.checkpoint_path, self.model, self.optimizer, self.scheduler)
            logger.info("Resumed from %s at step %d", self.checkpoint_path, self.progress.step)
        elif self.checkpoint_path.exists():
            raise ConfigError(
                f"{self.checkpoint_path} already exists. Pass --resume to continue that run, or change "
                "experiment.run_name to start a new one. Refusing to overwrite a trained model."
            )

        config.run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, config.run_dir / "config.yaml")
        self.writer = SummaryWriter(str(config.run_dir / "tensorboard"))

    def train(self) -> None:
        """Run optimisation steps until ``training.num_steps`` is reached."""
        settings = self.config.training
        batches = endless_training_batches(
            self.train_scenes, settings.batch_size, settings.views_per_scene, settings.num_workers
        )
        logger.info("Training on %s for %d steps", self.device, settings.num_steps)

        while self.progress.step < settings.num_steps:
            step_start = time.perf_counter()
            batch = next(batches).to(self.device)
            scalars = self._training_step(batch)
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
        self.writer.close()
        logger.info("Finished after %.2f training hours", self.progress.training_seconds / 3600)

    def _training_step(self, batch: SceneViews) -> dict[str, float]:
        """Perform one optimisation step and return the scalars worth logging."""
        settings = self.config.training
        self.model.train()

        # The number of source views changes from step to step, so one model
        # learns to work with any of the configured view counts.
        num_source_views = random.choice(settings.num_source_views)
        source_view_indices = torch.randperm(batch.num_views)[:num_source_views]
        source_views = batch.select_views(source_view_indices)

        rays, target_rgb = sample_training_rays(batch, settings.rays_per_scene)

        scene_encoding = self.model.encode_scene(source_views)
        renders = self.renderer.render_rays(self.model, scene_encoding, rays, randomize=True)
        loss_terms = self.loss_function(renders, target_rgb)

        self.optimizer.zero_grad(set_to_none=True)
        loss_terms["total"].backward()
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
        self.writer.add_image("validation/source_prediction_truth", result.preview, step)

        summary = " | ".join(f"{name} {value:.4f}" for name, value in result.mean.items())
        logger.info("step %d | validation | %s", step, summary)

    def _save_checkpoint(self) -> None:
        save_checkpoint(
            self.checkpoint_path, self.model, self.optimizer, self.scheduler, self.progress, self.config.raw
        )


def _check_training_settings(config: Config) -> None:
    """Catch settings that contradict each other before any work is done."""
    settings = config.training
    if max(settings.num_source_views) > settings.views_per_scene:
        raise ConfigError(
            f"training.num_source_views asks for up to {max(settings.num_source_views)} source views, "
            f"but only training.views_per_scene = {settings.views_per_scene} views are loaded per scene."
        )
    if min(settings.num_source_views) < 1:
        raise ConfigError("training.num_source_views must only contain positive numbers.")


def _check_validation_settings(config: Config, validation_scenes: SceneDataset) -> None:
    """Make sure validation can run, now rather than thousands of steps into training."""
    num_views = validation_scenes.num_views(0)
    out_of_range = [index for index in config.validation.source_views if not 0 <= index < num_views]
    if out_of_range:
        raise ConfigError(
            f"validation.source_views contains {out_of_range}, but validation scenes have only "
            f"{num_views} views (valid indices are 0 to {num_views - 1})."
        )
