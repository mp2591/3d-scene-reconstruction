"""Command-line interface: ``scene-recon <command> --config <file>``.

Commands:

    train        Train the model described by a config.
    evaluate     Score a trained model on the test split.
    profile      Report parameter counts and rendering speed.
    check-model  Verify that the configured model satisfies the pipeline's contract.

Any config value can be changed from the command line with ``--set``, e.g.
``--set training.batch_size=8 experiment.run_name=my_run``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from scene_recon.build import build_dataset, build_metrics, build_model, resolve_device
from scene_recon.checkpoints import load_checkpoint
from scene_recon.config import Config, load_config
from scene_recon.evaluation import Evaluator
from scene_recon.models.contract import check_model_contract
from scene_recon.profiling import count_parameters, measure_rendering_speed
from scene_recon.registry import ConfigError
from scene_recon.rendering import VolumeRenderer
from scene_recon.training import CHECKPOINT_NAME, Trainer

logger = logging.getLogger(__name__)


def main(arguments: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    parser = _build_parser()
    args = parser.parse_args(arguments)
    # A mistake in a config is the user's to fix, so it is reported as a
    # one-line message rather than as a stack trace.
    try:
        config = load_config(args.config, args.set)
        args.run(config, args)
    except ConfigError as error:
        raise SystemExit(f"Config error: {error}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scene-recon", description="Train, evaluate and profile sparse-view 3D reconstruction models."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    # Options shared by every command.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", required=True, help="Path to a YAML config file.")
    shared.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values, e.g. training.batch_size=8."
    )

    # Options shared by the commands that work on a trained model.
    uses_checkpoint = argparse.ArgumentParser(add_help=False)
    uses_checkpoint.add_argument(
        "--checkpoint", help=f"Checkpoint to load. Defaults to checkpoints/{CHECKPOINT_NAME} in the run directory."
    )

    train = commands.add_parser("train", parents=[shared], help="Train a model.")
    train.add_argument("--resume", action="store_true", help="Continue from the run's latest checkpoint.")
    train.set_defaults(run=_train)

    evaluate = commands.add_parser(
        "evaluate", parents=[shared, uses_checkpoint], help="Score a trained model on the test split."
    )
    evaluate.set_defaults(run=_evaluate)

    profile = commands.add_parser(
        "profile", parents=[shared, uses_checkpoint], help="Report parameter counts and rendering speed."
    )
    profile.add_argument("--num-frames", type=int, default=20, help="Number of frames to time.")
    profile.set_defaults(run=_profile)

    check = commands.add_parser(
        "check-model", parents=[shared], help="Verify the configured model against the pipeline's contract."
    )
    check.add_argument("--image-size", type=int, default=64, help="Size of the random test images.")
    check.set_defaults(run=_check_model)

    return parser


def _train(config: Config, args: argparse.Namespace) -> None:
    Trainer(config, resume=args.resume).train()


def _evaluate(config: Config, args: argparse.Namespace) -> None:
    """Run every configured source-view protocol on the test split and save the results as JSON."""
    device = resolve_device(config.experiment.device)
    model = build_model(config).to(device)
    checkpoint_path = _checkpoint_path(config, args)
    progress = load_checkpoint(checkpoint_path, model)

    settings = config.evaluation
    evaluator = Evaluator(model, VolumeRenderer(config.renderer), build_metrics(settings.metrics), device)
    test_scenes = build_dataset(config, "test")

    protocols = []
    for source_views in settings.source_view_sets:
        logger.info("Evaluating with source views %s", source_views)
        image_dir = None
        if settings.save_images:
            image_dir = config.run_dir / "renders" / f"{len(source_views)}_source_views"
        result = evaluator.evaluate(
            test_scenes,
            source_views,
            num_scenes=settings.num_scenes,
            num_target_views=settings.num_target_views,
            image_dir=image_dir,
        )
        logger.info("Source views %s: %s", source_views, result.mean)
        protocols.append({"source_views": source_views, "mean": result.mean, "per_scene": result.per_scene})

    report = {
        "checkpoint": str(checkpoint_path),
        "training_step": progress.step,
        "training_hours": progress.training_seconds / 3600,
        "protocols": protocols,
    }
    _write_json(report, config.run_dir / "evaluation.json")


def _profile(config: Config, args: argparse.Namespace) -> None:
    """Count parameters and time full-frame rendering on the first test scene."""
    device = resolve_device(config.experiment.device)
    model = build_model(config).to(device)

    # Speed does not depend on the weights, so an untrained model can be
    # profiled too. A checkpoint is loaded only if one is given or exists.
    checkpoint_path = _checkpoint_path(config, args)
    if args.checkpoint or checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model)

    test_scenes = build_dataset(config, "test")
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
        num_frames=args.num_frames,
    )
    height, width = views.image_size
    report = {
        "device": str(device),
        "image_size": [height, width],
        "num_source_views": num_source_views,
        "parameters": count_parameters(model),
        "rendering": speed,
    }
    logger.info("\n%s", json.dumps(report, indent=2))
    _write_json(report, config.run_dir / "profile.json")


def _check_model(config: Config, args: argparse.Namespace) -> None:
    model = build_model(config).to(resolve_device(config.experiment.device))
    check_model_contract(model, image_size=args.image_size)
    logger.info("Model '%s' satisfies the pipeline contract.", config.model["name"])


def _checkpoint_path(config: Config, args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    return config.run_dir / "checkpoints" / CHECKPOINT_NAME


def _write_json(content: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(content, file, indent=2)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
