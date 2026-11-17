"""Command-line interface: ``scene-recon <command> --config <file>``.

Commands for a single run (``--config`` is an experiment config):

    train        Train the model described by a config.
    evaluate     Score a trained model on the test split.
    profile      Report parameter counts and rendering speed.
    check-model  Verify that the configured model satisfies the pipeline's contract.

Command for a set of runs (``--config`` is a suite file):

    run-suite    Train, evaluate and profile every run of a suite, then write a comparison report.

Any config value can be changed from the command line with ``--set``, e.g.
``--set training.batch_size=8 experiment.run_name=my_run``. For ``run-suite``
the overrides apply to every run of the suite.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from scene_recon.config import load_config
from scene_recon.registry import ConfigError
from scene_recon.stages import check_model, run_evaluation, run_profiling, run_training
from scene_recon.suite import PreflightError, load_suite, run_suite

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s | %(message)s"


def main(arguments: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt="%H:%M:%S")

    args = _build_parser().parse_args(arguments)
    # Mistakes in a config are the user's to fix, so they are reported as a
    # short message rather than as a stack trace.
    try:
        args.run(args)
    except ConfigError as error:
        raise SystemExit(f"Config error: {error}") from error
    except PreflightError as error:
        raise SystemExit(f"Suite not started. {error}") from error


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
        "--checkpoint", type=Path, help="Checkpoint to load. Defaults to the run's checkpoints/last.pt."
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
    profile.set_defaults(run=_profile)

    check = commands.add_parser(
        "check-model", parents=[shared], help="Verify the configured model against the pipeline's contract."
    )
    check.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Size of the random test images. Defaults to the size of the test split's images.",
    )
    check.set_defaults(run=_check_model)

    suite = commands.add_parser(
        "run-suite",
        parents=[shared],
        help="Run every experiment of a suite file in order and write a comparison report.",
    )
    suite.set_defaults(run=_run_suite)

    return parser


def _train(args: argparse.Namespace) -> None:
    run_training(load_config(args.config, args.set), resume=args.resume)


def _evaluate(args: argparse.Namespace) -> None:
    run_evaluation(load_config(args.config, args.set), args.checkpoint)


def _profile(args: argparse.Namespace) -> None:
    run_profiling(load_config(args.config, args.set), args.checkpoint)


def _check_model(args: argparse.Namespace) -> None:
    image_size = tuple(args.image_size) if args.image_size else None
    check_model(load_config(args.config, args.set), image_size)


def _run_suite(args: argparse.Namespace) -> None:
    suite = load_suite(args.config, args.set)

    # Keep a complete log next to the results; a suite can run for days.
    suite.directory.mkdir(parents=True, exist_ok=True)
    log_file = logging.FileHandler(suite.directory / "suite.log")
    log_file.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(log_file)

    try:
        logger.info("Suite '%s': %d runs, stages %s", suite.name, len(suite.runs), suite.stages)
        failures = run_suite(suite)
    finally:
        logging.getLogger().removeHandler(log_file)
        log_file.close()

    if failures:
        summary = "\n".join(f"  {name}: {message}" for name, message in failures.items())
        raise SystemExit(f"{len(failures)} run(s) failed (details in {suite.directory / 'suite.log'}):\n{summary}")
    logger.info("Suite finished. Report: %s", suite.directory / "summary.md")


if __name__ == "__main__":
    main()
