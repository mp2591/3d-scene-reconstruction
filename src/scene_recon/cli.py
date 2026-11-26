"""Command-line interface: ``scene-recon <command> --config <file>``.

Commands for a single run (``--config`` is an experiment config):

    train        Train the model described by a config.
    evaluate     Score a trained model on the test split.
    profile      Report parameter counts and rendering speed.
    check-model  Verify that the configured model satisfies the pipeline's contract.

Command for a set of runs (``--config`` is a suite file):

    run-suite    Train, evaluate and profile every run of a suite, then write a comparison report.

Command for the learning-rate pilot (see ``scene_recon.pilot``):

    choose-encoder-rates  Choose each encoder's learning rate from the pilot's scores, write the
                          rates into the comparison's suite file, and continue the chosen runs.
    estimate-time         Time training and scoring on this machine, and estimate how long the
                          pilot and the comparison take for a few step budgets.

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
from scene_recon.models.contract import ModelContractError
from scene_recon.pilot import check_pilot, choose_and_continue
from scene_recon.registry import ConfigError
from scene_recon.stages import STAGES, check_model, run_evaluation, run_profiling, run_training
from scene_recon.run_files import SuiteFiles
from scene_recon.suite import PreflightError, SuiteReportError, load_suite, run_suite
from scene_recon.time_estimate import estimate_time

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
    except ModelContractError as error:
        raise SystemExit(f"Model check failed: {error}") from error
    except PreflightError as error:
        raise SystemExit(f"Suite not started. {error}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scene-recon", description="Train, evaluate and profile sparse-view 3D reconstruction models."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    # Options shared by every command.
    overrides = argparse.ArgumentParser(add_help=False)
    overrides.add_argument(
        "--set",
        nargs="*",
        action="extend",  # every --set counts: adding one never drops the overrides of another
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. training.batch_size=8. May be given more than once; later values win.",
    )
    shared = argparse.ArgumentParser(add_help=False, parents=[overrides])
    shared.add_argument("--config", required=True, help="Path to a YAML config file.")

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
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="Stages the model will be used in. Training checks only apply to 'train'. Default: all.",
    )
    check.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "Size of the random test images, for every check. Default: each split the stages read is "
            "checked at the size of its own images (train and val for 'train', test for 'evaluate' and 'profile')."
        ),
    )
    check.set_defaults(run=_check_model)

    suite = commands.add_parser(
        "run-suite",
        parents=[shared],
        help="Run every experiment of a suite file in order and write a comparison report.",
    )
    suite.set_defaults(run=_run_suite)

    choose = commands.add_parser(
        "choose-encoder-rates",
        parents=[overrides],
        help="Choose each encoder's learning rate from the pilot and continue the chosen runs.",
    )
    choose.add_argument("--pilot", required=True, type=Path, help="The pilot's suite file.")
    choose.add_argument("--comparison", required=True, type=Path, help="The comparison's suite file.")
    choose.add_argument(
        "--check-only",
        action="store_true",
        help="Only check what can be checked before the pilot trains (see scene_recon.pilot); choose nothing.",
    )
    _add_comparison_overrides(choose)
    choose.set_defaults(run=_choose_encoder_rates)

    estimate = commands.add_parser(
        "estimate-time",
        parents=[overrides],
        help="Time training and scoring on this machine, and estimate how long the comparison takes.",
    )
    estimate.add_argument("--pilot", required=True, type=Path, help="The pilot's suite file.")
    estimate.add_argument("--comparison", required=True, type=Path, help="The comparison's suite file.")
    estimate.add_argument(
        "--steps",
        nargs="+",
        type=int,
        required=True,
        metavar="STEPS",
        help="Training steps per comparison run to estimate the time of, e.g. 100000 200000.",
    )
    estimate.add_argument(
        "--first-scenes",
        type=int,
        help="Test scenes the comparison is scored on first, before all of them (as scripts/run_comparison.sh does).",
    )
    _add_comparison_overrides(estimate)
    estimate.set_defaults(run=_estimate_time)

    return parser


def _add_comparison_overrides(command: argparse.ArgumentParser) -> None:
    """``--comparison-set``, for a command that loads both the pilot and the comparison."""
    command.add_argument(
        "--comparison-set",
        nargs="*",
        action="extend",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override config values of the comparison only, after --set, e.g. its step budget "
            "training.num_steps=200000, which the pilot does not share."
        ),
    )


def _train(args: argparse.Namespace) -> None:
    run_training(load_config(args.config, args.set), resume=args.resume)


def _evaluate(args: argparse.Namespace) -> None:
    run_evaluation(load_config(args.config, args.set), args.checkpoint)


def _profile(args: argparse.Namespace) -> None:
    run_profiling(load_config(args.config, args.set), args.checkpoint)


def _check_model(args: argparse.Namespace) -> None:
    image_size = tuple(args.image_size) if args.image_size else None
    check_model(load_config(args.config, args.set), image_size, stages=args.stages)


def _run_suite(args: argparse.Namespace) -> None:
    suite = load_suite(args.config, args.set)
    files = SuiteFiles(suite.directory)

    # Keep a complete log next to the results; a suite can run for days.
    suite.directory.mkdir(parents=True, exist_ok=True)
    log_file = logging.FileHandler(files.log)
    log_file.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(log_file)

    report_error = None
    try:
        logger.info("Suite '%s': %d runs, stages %s", suite.name, len(suite.runs), suite.stages)
        failures = run_suite(suite)
    except SuiteReportError as error:
        failures, report_error = error.failures, str(error)
    finally:
        logging.getLogger().removeHandler(log_file)
        log_file.close()

    problems = [f"  {name}: {message}" for name, message in failures.items()]
    if report_error is not None:
        problems.append(f"  {report_error}")
    if problems:
        raise SystemExit(f"The suite did not finish cleanly (details in {files.log}):\n" + "\n".join(problems))
    logger.info("Suite finished. Report: %s", files.summary_markdown)


def _choose_encoder_rates(args: argparse.Namespace) -> None:
    if args.check_only:
        comparison_overrides = [*args.set, *args.comparison_set]
        pilot, comparison = load_suite(args.pilot, args.set), load_suite(args.comparison, comparison_overrides)
        tuned = check_pilot(pilot, comparison, args.comparison, comparison_overrides)
        logger.info("The pilot fits the comparison: it tunes %s.", ", ".join(run.name for run in tuned))
        return
    choose_and_continue(args.pilot, args.comparison, args.set, args.comparison_set)


def _estimate_time(args: argparse.Namespace) -> None:
    pilot = load_suite(args.pilot, args.set)
    comparison = load_suite(args.comparison, [*args.set, *args.comparison_set])
    print(estimate_time(pilot, comparison, args.steps, args.first_scenes))


if __name__ == "__main__":
    main()
