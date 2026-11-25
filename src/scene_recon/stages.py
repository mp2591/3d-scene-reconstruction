"""The stages of an experiment, each as one function.

    run_training    -> checkpoint and TensorBoard logs
    run_evaluation  -> evaluation.json and preview renders
    run_profiling   -> profile.json

The single-run commands (``scene-recon train``, ``evaluate``, ``profile``)
and the suite runner both call these functions, so a run produces the same
outputs, and follows the same rules, whichever way it is started.

A checkpoint is only ever used with the settings it was trained with, and
every results file records the checkpoint (by content hash), the settings
and the software environment it was made with. See ``scene_recon.provenance``.

``check_run_can_start`` builds and checks everything a run's stages will
need, without training, so that mistakes surface in seconds.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import torch

from scene_recon.build import (
    apply_float32_matmul_precision,
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_seeded_model,
    resolve_device,
)
from scene_recon.checkpoints import TrainingProgress, load_checkpoint
from scene_recon.config import Config, save_config
from scene_recon.data import TrainingSchedule
from scene_recon.data.base import SceneDataset
from scene_recon.environment import environment, evaluation_software
from scene_recon.evaluation import Evaluator, check_protocol_feasible, save_preview
from scene_recon.models.base import pretrained_weights_of
from scene_recon.models.contract import check_model_contract
from scene_recon.optim import build_optimizer_and_scheduler
from scene_recon.profiling import count_parameters, measure_rendering_speed
from scene_recon.provenance import (
    check_checkpoint_matches_config,
    untrained_model_record,
    check_training_data_unchanged,
    checkpoint_record,
    evaluation_settings,
    profiling_settings,
)
from scene_recon.registry import ConfigError
from scene_recon.rendering import VolumeRenderer
from scene_recon.run_files import RunFiles, write_atomically, write_json
from scene_recon.training import Trainer, check_training_settings

logger = logging.getLogger(__name__)

STAGES = ("train", "evaluate", "profile")
"""All stages, in the order they are executed."""


def run_training(config: Config, resume: bool = False) -> None:
    """Train the model of ``config``, optionally resuming from the run's latest checkpoint."""
    Trainer(config, resume=resume).train()


def run_evaluation(config: Config, checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Evaluate a trained model on the test split with every configured source-view set.

    Writes ``evaluation.json`` and the preview renders into the run directory,
    and saves the config it was evaluated with as the run's ``config.yaml``, so
    that evaluating with that file repeats this evaluation (e.g. on every test
    scene, after training was last run on fewer).

    A model without trainable parameters (frozen, or analytic) cannot be
    trained, so it never has a checkpoint: it is evaluated as built, and the
    results record a hash of its weights instead of a checkpoint. Any other
    model must have a checkpoint, so random untrained weights are never
    measured by mistake.

    Args:
        config: The run to evaluate.
        checkpoint_path: Weights to load. Defaults to the run's latest checkpoint.

    Returns:
        The content written to ``evaluation.json``.

    Raises:
        ConfigError: If the checkpoint was trained with other settings than ``config``.
        FileNotFoundError: If a model with trainable parameters has no checkpoint.
        RuntimeError: If a metric comes out as NaN or infinite; nothing is written then.
    """
    apply_float32_matmul_precision(config)
    files = RunFiles(config.run_dir)
    weights_record: dict[str, Any]
    if checkpoint_path is None and not files.checkpoint.exists() and not _has_trainable_parameters(config):
        model = build_seeded_model(config).to(resolve_device(config.experiment.device))
        weights_record = {"checkpoint": None, "untrained_model": untrained_model_record(config, model)}
    else:
        checkpoint_path = checkpoint_path or files.checkpoint
        model, progress = _load_trained_model(config, checkpoint_path)
        weights_record = {"checkpoint": checkpoint_record(checkpoint_path, progress)}

    settings = config.evaluation
    device = resolve_device(config.experiment.device)
    # Read before scoring, so that reading them can never cost a finished evaluation.
    recorded_environment, software = environment(), evaluation_software(device)
    evaluator = Evaluator(model, VolumeRenderer(config.renderer), build_metrics(settings.metrics), device)
    test_scenes = build_dataset(config, "test")
    check_evaluation_settings(config, test_scenes)

    protocols, previews = [], {}
    for source_views in settings.source_view_sets:
        logger.info("Evaluating with source views %s", source_views)
        result = evaluator.evaluate(
            test_scenes,
            source_views,
            num_scenes=settings.num_scenes,
            num_target_views=settings.num_target_views,
            num_preview_scenes=settings.num_preview_scenes,
            image_dir=files.renders(len(source_views)) if settings.save_images else None,
        )
        logger.info("Source views %s: %s", source_views, result.mean)
        non_finite = [name for name, value in result.mean.items() if not math.isfinite(value)]
        if non_finite:
            raise RuntimeError(
                f"Evaluation with source views {source_views} produced non-finite {non_finite}; "
                "the model's output is probably NaN. No results were written."
            )
        previews[len(source_views)] = result.previews
        protocols.append(
            {
                "source_views": source_views,
                "num_source_views": len(source_views),
                "mean": result.mean,
                "per_scene": result.per_scene,
                "preview_scene_ids": [preview.scene_id for preview in result.previews],
            }
        )

    # Replace the previews of an earlier evaluation, so none are left over.
    for num_source_views, protocol_previews in previews.items():
        preview_dir = files.previews(num_source_views)
        shutil.rmtree(preview_dir, ignore_errors=True)
        for preview in protocol_previews:
            save_preview(preview, preview_dir / preview.scene_id)

    report = {
        **weights_record,
        "settings": evaluation_settings(config),
        "environment": recorded_environment,
        "software": software,
        "protocols": protocols,
    }
    write_json(report, files.evaluation)
    write_atomically(files.config, lambda path: save_config(config, path))
    logger.info("Wrote %s", files.evaluation)
    return report


def run_profiling(config: Config, checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Count parameters and time full-frame rendering on the first test scene.

    Writes ``profile.json`` into the run directory. Speed does not depend on
    the weights, so an untrained model can be profiled too: a checkpoint is
    loaded only if one is given or the run has one.

    Returns:
        The content written to ``profile.json``.
    """
    apply_float32_matmul_precision(config)
    files = RunFiles(config.run_dir)
    device = resolve_device(config.experiment.device)
    checkpoint_path = checkpoint_path or (files.checkpoint if files.checkpoint.exists() else None)
    if checkpoint_path is not None:
        model, _ = _load_trained_model(config, checkpoint_path)
    else:
        model = build_model(config).to(device)

    test_scenes = build_dataset(config, "test")
    check_profiling_settings(config, test_scenes)
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
        min_free_gpu_memory=config.profiling.min_free_gpu_memory,
    )
    height, width = views.image_size
    report = {
        "image_size": [height, width],
        "num_source_views": num_source_views,
        "settings": profiling_settings(config),
        "environment": environment(),
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
    write_json(report, files.profile)
    return report


def check_model(
    config: Config, image_size: tuple[int, int] | None = None, stages: Sequence[str] = STAGES
) -> None:
    """Build the model of ``config`` and verify it against the pipeline's contract.

    Args:
        config: The run whose model to check.
        image_size: Size of the random test images for every check. By
            default each split the stages read is checked at the size of its
            own images (see ``_check_contract_where_used``).
        stages: The stages the model will be used in; training checks
            (backward pass, gradients, parameter groups) only apply to "train".
    """
    splits = _how_splits_are_used(config, stages)
    if image_size is None:
        image_sizes = {split: dataset_image_size(build_dataset(config, split)) for split in splits}
    else:
        image_sizes = {split: tuple(image_size) for split in splits}
    apply_float32_matmul_precision(config)  # the contract is checked with the arithmetic of the real run
    device = resolve_device(config.experiment.device)
    model = build_model(config).to(device)
    _check_contract_where_used(model, config, image_sizes, stages, device)
    logger.info("Model '%s' satisfies the pipeline contract.", config.model["name"])


def check_run_can_start(config: Config, stages: list[str]) -> None:
    """Build and check everything the given stages of a run will need, without training.

    Only what the stages use is built: a profile-only run needs no training
    data, loss or metrics. Checks: the datasets open and have the requested
    views; an existing checkpoint was trained with these settings and, when
    training is requested (even if it is already finished and will be
    skipped), on the data the training split holds now; the loss,
    metrics (including optional packages such as LPIPS) and optimiser can be
    built; the model satisfies the contract with the configured sample counts.

    Raises:
        Exception: The first problem found, as raised by the failing component.
    """
    files = RunFiles(config.run_dir)
    if files.checkpoint.exists():
        check_checkpoint_matches_config(files.checkpoint, config)

    image_sizes = {}  # split -> (height, width) of its images
    if "train" in stages:
        train_scenes = build_dataset(config, "train")
        validation_scenes = build_dataset(config, "val")
        check_training_settings(config, validation_scenes)
        TrainingSchedule(
            train_scenes,
            seed=config.experiment.seed,
            batch_size=config.training.batch_size,
            views_per_scene=config.training.views_per_scene,
            num_source_views=config.training.num_source_views,
        ).scene_plan(0)  # raises if training scenes have too few views
        if files.checkpoint.exists():
            # Also for a finished run: its checkpoint is reused in the comparison
            # as if it had been trained on the current data.
            check_training_data_unchanged(files.checkpoint, train_scenes)
        build_loss(config)
        build_metrics(config.validation.metrics)
        image_sizes["train"] = dataset_image_size(train_scenes)
        image_sizes["val"] = dataset_image_size(validation_scenes)
    if "evaluate" in stages or "profile" in stages:
        test_scenes = build_dataset(config, "test")
        image_sizes["test"] = dataset_image_size(test_scenes)
    if "evaluate" in stages:
        check_evaluation_settings(config, test_scenes)
        build_metrics(config.evaluation.metrics)
    if "profile" in stages:
        check_profiling_settings(config, test_scenes)

    apply_float32_matmul_precision(config)  # the contract is checked with the arithmetic of the real run
    device = resolve_device(config.experiment.device)
    model = build_model(config).to(device)
    _check_contract_where_used(model, config, image_sizes, stages, device)
    trainable = any(parameter.requires_grad for parameter in model.parameters())
    if "evaluate" in stages and "train" not in stages and not files.checkpoint.exists() and trainable:
        raise ConfigError(
            f"{config.run_dir} has no checkpoint to evaluate, and its model has trainable parameters, so "
            "evaluating it as built would measure untrained weights. Add the 'train' stage, or train the run first."
        )
    if "train" in stages:
        build_optimizer_and_scheduler(model, config.optimization)


def check_evaluation_settings(config: Config, test_scenes: SceneDataset) -> None:
    """Make sure every evaluation protocol can run on every test scene it will use."""
    for source_views in config.evaluation.source_view_sets:
        check_protocol_feasible(
            test_scenes, source_views, config.evaluation.num_scenes, where="evaluation.source_view_sets"
        )


def check_profiling_settings(config: Config, test_scenes: SceneDataset) -> None:
    """Make sure profiling can run: it renders one view of the first test scene from the first source-view set.

    Only that scene and that set matter; later scenes and protocols, which
    only evaluation uses, do not affect profiling.
    """
    check_protocol_feasible(
        test_scenes,
        config.evaluation.source_view_sets[0],
        num_scenes=1,
        where="profiling (first test scene, evaluation.source_view_sets[0])",
    )


def dataset_image_size(scenes: SceneDataset) -> tuple[int, int]:
    """(height, width) of a dataset's images, read from its first view."""
    return scenes.load_scene(0, [0]).image_size


def _has_trainable_parameters(config: Config) -> bool:
    return any(parameter.requires_grad for parameter in build_model(config).parameters())


def _load_trained_model(config: Config, checkpoint_path: Path) -> tuple[Any, TrainingProgress]:
    """Build the model and load a checkpoint into it, after checking that the two belong together."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Train the model first, or pass --checkpoint.")
    check_checkpoint_matches_config(checkpoint_path, config)
    model = build_model(config).to(resolve_device(config.experiment.device))
    progress = load_checkpoint(checkpoint_path, model)
    return model, progress


def _how_splits_are_used(config: Config, stages: Sequence[str]) -> dict[str, tuple[set[int], list[int] | None]]:
    """How the given stages use the model on each split they read.

    Returns, for each split: the numbers of source views it is run with in
    evaluation mode, and those it is trained with (None if it is not trained on).
    """
    uses = {}
    if "train" in stages:
        uses["train"] = (set(), list(config.training.num_source_views))
        uses["val"] = ({len(config.validation.source_views)}, None)  # validation runs in evaluation mode
    test_view_counts = set()
    if "evaluate" in stages:
        test_view_counts |= {len(views) for views in config.evaluation.source_view_sets}
    if "profile" in stages:
        test_view_counts.add(len(config.evaluation.source_view_sets[0]))  # profiling uses the first set only
    if test_view_counts:
        uses["test"] = (test_view_counts, None)
    return uses


def _check_contract_where_used(
    model: Any,
    config: Config,
    image_sizes: dict[str, tuple[int, int]],
    stages: Sequence[str],
    device: torch.device,
) -> None:
    """Check the model as each split uses it, at the size of that split's images.

    The training split is checked in training mode (with a backward pass),
    the validation and test splits in evaluation mode, each with its own
    numbers of source views. Splits whose images have the same size are
    checked together, so when all splits share one size (the normal case)
    the model is checked once and an error passes through unchanged. When
    the sizes differ, a failure names the splits of the size that failed:
    some encoders accept only some sizes, such as multiples of their stride.
    """
    if "train" in stages:
        pretrained_weights_of(model)  # raises if a part's record could not be stored in a checkpoint
    uses = _how_splits_are_used(config, stages)
    sizes = sorted({image_sizes[split] for split in uses})
    for size in sizes:
        splits = [split for split in uses if image_sizes[split] == size]
        inference_view_counts = set().union(*(uses[split][0] for split in splits))
        training_view_counts = uses["train"][1] if "train" in splits else None
        try:
            check_model_contract(
                model,
                image_size=size,
                num_coarse_samples=config.renderer.num_coarse_samples,
                num_fine_samples=config.renderer.num_fine_samples,
                inference_view_counts=sorted(inference_view_counts),
                training_view_counts=training_view_counts,
                device=device,
            )
        except Exception as error:
            if len(sizes) == 1:
                raise
            raise ConfigError(
                f"The splits' images differ in size, and the model fails on the {' and '.join(splits)} images "
                f"({size[0]}x{size[1]}): {type(error).__name__}: {error}"
            ) from error
