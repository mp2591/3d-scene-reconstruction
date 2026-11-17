"""The stages of an experiment, each as one function.

    run_training    -> checkpoint and TensorBoard logs
    run_evaluation  -> evaluation.json and preview renders
    run_profiling   -> profile.json

The single-run commands (``scene-recon train``, ``evaluate``, ``profile``)
and the suite runner both call these functions, so a run produces the same
outputs whichever way it is started.

``evaluation.json`` and ``profile.json`` record the settings they were
produced with, which lets the suite runner tell whether they are still
up to date.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from scene_recon.build import build_dataset, build_metrics, build_model, resolve_device
from scene_recon.checkpoints import load_checkpoint
from scene_recon.config import Config
from scene_recon.data.base import SceneDataset
from scene_recon.evaluation import Evaluator, check_source_views, save_preview
from scene_recon.models.contract import check_model_contract
from scene_recon.profiling import count_parameters, measure_rendering_speed
from scene_recon.rendering import VolumeRenderer
from scene_recon.run_files import RunFiles
from scene_recon.training import Trainer

logger = logging.getLogger(__name__)


def run_training(config: Config, resume: bool = False) -> None:
    """Train the model of ``config``, optionally resuming from the run's latest checkpoint."""
    Trainer(config, resume=resume).train()


def run_evaluation(config: Config, checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Evaluate a trained model on the test split with every configured source-view set.

    Writes ``evaluation.json`` and the preview renders into the run directory.

    Args:
        config: The run to evaluate.
        checkpoint_path: Weights to load. Defaults to the run's latest checkpoint.

    Returns:
        The content written to ``evaluation.json``.
    """
    files = RunFiles(config.run_dir)
    checkpoint_path = checkpoint_path or files.checkpoint
    device = resolve_device(config.experiment.device)
    model = build_model(config).to(device)
    progress = load_checkpoint(checkpoint_path, model)

    settings = config.evaluation
    evaluator = Evaluator(model, VolumeRenderer(config.renderer), build_metrics(settings.metrics), device)
    test_scenes = build_dataset(config, "test")
    check_evaluation_settings(config, test_scenes)

    protocols = []
    for source_views in settings.source_view_sets:
        num_source_views = len(source_views)
        logger.info("Evaluating with source views %s", source_views)
        result = evaluator.evaluate(
            test_scenes,
            source_views,
            num_scenes=settings.num_scenes,
            num_target_views=settings.num_target_views,
            num_preview_scenes=settings.num_preview_scenes,
            image_dir=files.renders(num_source_views) if settings.save_images else None,
        )
        logger.info("Source views %s: %s", source_views, result.mean)

        # Replace the previews of an earlier evaluation, so none are left over.
        preview_dir = files.previews(num_source_views)
        shutil.rmtree(preview_dir, ignore_errors=True)
        for preview in result.previews:
            save_preview(preview, preview_dir / preview.scene_id)

        protocols.append(
            {
                "source_views": source_views,
                "num_source_views": num_source_views,
                "mean": result.mean,
                "per_scene": result.per_scene,
                "preview_scene_ids": [preview.scene_id for preview in result.previews],
            }
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "training_step": progress.step,
        "training_hours": progress.training_seconds / 3600,
        "settings": evaluation_settings_snapshot(config),
        "protocols": protocols,
    }
    _write_json(report, files.evaluation)
    return report


def run_profiling(config: Config, checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Count parameters and time full-frame rendering on the first test scene.

    Writes ``profile.json`` into the run directory. Speed does not depend on
    the weights, so an untrained model can be profiled too: a checkpoint is
    loaded only if one is given or the run has one.

    Returns:
        The content written to ``profile.json``.
    """
    files = RunFiles(config.run_dir)
    device = resolve_device(config.experiment.device)
    model = build_model(config).to(device)
    if checkpoint_path is not None or files.checkpoint.exists():
        load_checkpoint(checkpoint_path or files.checkpoint, model)

    test_scenes = build_dataset(config, "test")
    check_evaluation_settings(config, test_scenes)
    source_view_indices = config.evaluation.source_view_sets[0]
    target_view_index = next(
        index for index in range(test_scenes.num_views(0)) if index not in source_view_indices
    )
    views = test_scenes.load_scene(0, [*source_view_indices, target_view_index]).to(device)
    num_source_views = len(source_view_indices)

    speed = measure_rendering_speed(
        model,
        VolumeRenderer(config.renderer),
        source_views=views.select_views(range(num_source_views)),
        target_view=views.select_views([num_source_views]),
        num_frames=config.profiling.num_frames,
        num_warmup_frames=config.profiling.num_warmup_frames,
    )
    height, width = views.image_size
    report = {
        "device": str(device),
        "image_size": [height, width],
        "num_source_views": num_source_views,
        "settings": profiling_settings_snapshot(config),
        "parameters": count_parameters(model),
        "rendering": speed,
    }
    logger.info(
        "%s parameters | %.2f frames per second at %dx%d",
        f"{report['parameters']['total']:,}",
        speed["frames_per_second"],
        height,
        width,
    )
    _write_json(report, files.profile)
    return report


def check_model(config: Config, image_size: tuple[int, int] | None = None) -> None:
    """Build the model of ``config`` and verify it against the pipeline's contract.

    Args:
        config: The run whose model to check.
        image_size: Size of the random test images. Defaults to the size of
            the test split's images.
    """
    if image_size is None:
        image_size = dataset_image_size(build_dataset(config, "test"))
    model = build_model(config).to(resolve_device(config.experiment.device))
    check_model_contract(model, image_size=image_size)
    logger.info("Model '%s' satisfies the pipeline contract.", config.model["name"])


def check_evaluation_settings(config: Config, test_scenes: SceneDataset) -> None:
    """Make sure every evaluation source view exists in the test scenes."""
    num_views = test_scenes.num_views(0)
    for source_views in config.evaluation.source_view_sets:
        check_source_views(source_views, num_views, where="evaluation.source_view_sets")


def dataset_image_size(scenes: SceneDataset) -> tuple[int, int]:
    """(height, width) of a dataset's images, read from its first view."""
    return scenes.load_scene(0, [0]).image_size


def evaluation_settings_snapshot(config: Config) -> dict[str, Any]:
    """The settings that determine evaluation results, apart from the weights."""
    return {
        "evaluation": config.raw["evaluation"],
        "renderer": config.raw["renderer"],
        "test_data": config.raw["data"]["test"],
    }


def profiling_settings_snapshot(config: Config) -> dict[str, Any]:
    """The settings that determine profiling results."""
    return {
        "profiling": config.raw["profiling"],
        "renderer": config.raw["renderer"],
        "model": config.raw["model"],
        "source_views": config.evaluation.source_view_sets[0],
        "test_data": config.raw["data"]["test"],
    }


def _write_json(content: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(content, file, indent=2)
    logger.info("Wrote %s", path)
