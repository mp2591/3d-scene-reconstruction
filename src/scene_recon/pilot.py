"""Choosing each pretrained encoder's learning rate from the pilot, then continuing the chosen runs.

The learning-rate pilot (``configs/suites/encoder_lr_pilot.yaml``) trains each
model of the comparison (``configs/suites/proposal_comparison.yaml``) for a
short while with a few encoder learning rates, and scores every one on the
validation split. Each pilot run trains exactly like its comparison run, apart
from that rate: same model, seed, data and learning-rate schedule
(``optimization.decay_steps`` included), the same everything in its training
identity (see ``scene_recon.provenance``). So a pilot run is the first steps
of the comparison run that uses its rate, and the comparison can continue it
instead of training those steps again.

``scene-recon choose-encoder-rates`` does, for every model:

1. Score each rate: the mean PSNR of its evaluation, averaged over the
   evaluation's protocols (1 and 2 input views).
2. Take the lowest rate whose score is within ``TOLERANCE_DB`` of the best
   one. A lower rate changes the pretrained weights less.
3. Write that rate into the comparison's suite file.
4. Copy the chosen pilot run's checkpoint and training curves into the
   comparison run's directory, where the comparison continues them. The copy
   is recorded in that directory (``TRANSFER_FILE``).

The rule was decided before any pilot ran, and is fixed here rather than in a
setting, so that it cannot be adjusted after seeing the scores. The choice is
saved as ``chosen_rates.json`` next to the pilot's report.

Everything is checked before anything changes: the suite file is rewritten and
checkpoints are copied only once every pilot run is scored and every copy is
known to fit. With ``--check-only``, only what can be checked before the pilot
trains is checked (``check_pilot``): run it first, so that a mismatch is found
before hours of training.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from scene_recon.build import build_dataset, resolve_device
from scene_recon.checkpoints import read_checkpoint_starting_weights, read_progress
from scene_recon.config import Config, save_config
from scene_recon.environment import evaluation_software
from scene_recon.provenance import (
    check_checkpoint_matches_config,
    check_training_fingerprint,
    checkpoint_digest,
    evaluation_status,
    training_identity,
)
from scene_recon.registry import ConfigError
from scene_recon.report import SOFTWARE_PROBLEMS, read_software
from scene_recon.run_files import RunFiles, read_json, write_json
from scene_recon.suite import Suite, load_suite

logger = logging.getLogger(__name__)

METRIC = "psnr"
"""The metric that ranks the rates (higher is better)."""

TOLERANCE_DB = 0.1
"""A rate within this many dB of the best score counts as just as good."""

ENCODER_GROUP = "encoder"
"""The parameter group whose learning rate the pilot varies."""

ENCODER_RATE = f"optimization.group_overrides.{ENCODER_GROUP}.learning_rate"
"""The same setting as a dotted name."""

CHOICE_FILE = "chosen_rates.json"
"""Written into the pilot's suite directory."""

TRANSFER_FILE = "continued_from.json"
"""Written into a comparison run's directory when a pilot run is copied into it: which one, at which step."""


@dataclass(frozen=True)
class Candidate:
    """One pilot run: a comparison run's training with one encoder learning rate."""

    key: str
    """The pilot run's key in its suite."""

    rate: float
    config: Config


@dataclass(frozen=True)
class TunedRun:
    """A comparison run whose encoder learning rate the pilot chooses."""

    name: str
    """The run's name in the comparison's suite file, e.g. ``"hiera_mlp"``."""

    key: str
    """Its key in the comparison suite that the pilot runs train like, e.g. ``"hiera_mlp/seed_0"``."""

    candidates: tuple[Candidate, ...]
    """The pilot runs that try a rate for it, lowest rate first."""


@dataclass(frozen=True)
class Choice:
    """The rate chosen for one comparison run, with the scores it was chosen from."""

    run: TunedRun
    scores: dict[float, float]
    """Rate -> score (``METRIC`` averaged over the evaluation protocols)."""

    rate: float

    @property
    def winner(self) -> Candidate:
        return next(candidate for candidate in self.run.candidates if candidate.rate == self.rate)


# ---------------------------------------------------------------- what can be checked before the pilot


def check_pilot(
    pilot: Suite, comparison: Suite, comparison_path: Path, overrides: Sequence[str] = ()
) -> list[TunedRun]:
    """Everything about the pilot that can be checked before it trains (``--check-only``).

    The pairing (``pair_pilot_with_comparison``); that the pilot trains and
    evaluates with ``METRIC``; that the software of its evaluations can be
    recorded on this machine (``choose_rates`` compares it); and that a rate
    written into the comparison's suite file takes effect for every tuned run.

    Raises:
        ConfigError: Listing every problem.
    """
    tuned = pair_pilot_with_comparison(pilot, comparison)
    problems = []
    if not {"train", "evaluate"} <= set(pilot.stages):
        problems.append(f"the pilot's stages must include train and evaluate, got {pilot.stages}")
    for key, config in pilot.runs.items():
        names = [entry if isinstance(entry, str) else entry.get("name") for entry in config.evaluation.metrics]
        if METRIC not in names:
            problems.append(f"pilot run '{key}' is not evaluated with {METRIC}, which ranks the rates")
    for device in sorted({config.experiment.device for config in pilot.runs.values()}):
        _, reasons = read_software(evaluation_software(resolve_device(device)))
        if reasons:
            problems.append(
                f"the software of an evaluation on device '{device}' cannot be recorded on this machine, so the "
                "pilot's scores could not be compared: " + "; ".join(SOFTWARE_PROBLEMS[reason] for reason in reasons)
            )

    # Write a rate other than the current one into every tuned run, and check that each takes effect.
    text = comparison_path.read_text(encoding="utf-8")
    lines_found = True
    for run in tuned:
        try:
            _with_encoder_rate(text, run.name, run.candidates[0].rate, comparison_path)
        except ConfigError as error:
            problems.append(str(error))
            lines_found = False
    if lines_found:
        probe = {}
        for run in tuned:
            current = _encoder_rate(comparison.runs[run.key])
            probe[run.name] = (run.key, next(c.rate for c in run.candidates if c.rate != current))
        try:
            _suite_with_rates(comparison_path, probe, overrides)
        except ConfigError as error:
            problems.append(str(error))

    if problems:
        raise ConfigError("The pilot cannot choose the rates:\n  " + "\n  ".join(problems))
    return tuned


def pair_pilot_with_comparison(pilot: Suite, comparison: Suite) -> list[TunedRun]:
    """Find, for every pilot run, the comparison run it is the beginning of.

    Raises:
        ConfigError: Listing every problem: a pilot run that reads the
            comparison's test split, or is scored on another split than the
            comparison's validation split; one that trains like no comparison
            run or like several, or sets no encoder rate or writes it as text;
            a tuned run with fewer than two distinct rates, or other rates than
            another tuned run; pilot runs with different step budgets or seeds,
            or more steps than the comparison.
    """
    problems = []
    trainings = {key: _training_without_encoder_rate(config) for key, config in comparison.runs.items()}
    name_of = {key: name for name, replicates in comparison.grouped_runs().items() for key in replicates}
    reference = next(iter(comparison.runs.values()))
    validation, test = reference.raw["data"]["val"], reference.raw["data"]["test"]
    for split in ("train", "val"):
        if _reads_same_scenes(reference.raw["data"][split], test):
            problems.append(f"the comparison's data.{split} reads its test scenes")

    found: dict[str, list[Candidate]] = {}
    for key, config in pilot.runs.items():
        if not _same_data(config.raw["data"]["test"], validation):
            problems.append(
                f"pilot run '{key}' is scored on {config.data['test'].get('split_dir', config.data['test'])}, "
                "not on the comparison's validation split; its data.test must be the comparison's data.val"
            )
        for split in ("train", "val", "test"):  # trained on, validated during training, scored after it
            if _reads_same_scenes(config.raw["data"][split], test):
                problems.append(f"pilot run '{key}' reads the comparison's test scenes in its data.{split}")
        written = config.raw.get("optimization", {}).get("group_overrides", {}).get(ENCODER_GROUP, {}).get("learning_rate")
        if isinstance(written, str):
            # The checkpoint would record the text, the comparison the number: the copy would not match.
            problems.append(
                f"pilot run '{key}' writes its {ENCODER_RATE} as text ({written!r}); write it as a number "
                "such as 1.0e-5 (YAML reads 1e-5, without the dot, as text)"
            )
            continue
        rate = _encoder_rate(config)
        if rate is None:
            problems.append(f"pilot run '{key}' sets no {ENCODER_RATE}")
            continue
        matches = [other for other, training in trainings.items() if training == _training_without_encoder_rate(config)]
        if len(matches) != 1:
            problems.append(
                f"pilot run '{key}' trains like {len(matches)} runs of the comparison, not exactly one "
                f"({', '.join(matches) or 'none'}); apart from {ENCODER_RATE}, its training settings must be "
                "those of its comparison run"
            )
            continue
        found.setdefault(matches[0], []).append(Candidate(key, rate, config))

    # One budget and one seed for the whole pilot: every model is tuned the same way, and the
    # suite file holds one rate per run, so a run cannot be tuned once per seed.
    budgets = {config.training.num_steps for config in pilot.runs.values()}
    if len(budgets) > 1:
        problems.append(f"the pilot runs must all train the same number of steps, got {sorted(budgets)}")
    seeds = {config.experiment.seed for config in pilot.runs.values()}
    if len(seeds) > 1:
        problems.append(f"the pilot runs must all use one seed, got {sorted(seeds)}")

    tuned = []
    for key, candidates in found.items():
        rates = sorted(candidate.rate for candidate in candidates)
        if len(set(rates)) != len(rates) or len(rates) < 2:
            problems.append(f"comparison run '{key}' needs at least two different rates in the pilot, got {rates}")
        comparison_steps = comparison.runs[key].training.num_steps
        for candidate in candidates:
            if candidate.config.training.num_steps > comparison_steps:
                problems.append(
                    f"pilot run '{candidate.key}' trains {candidate.config.training.num_steps:,} steps, more than "
                    f"the {comparison_steps:,} of comparison run '{key}' that would continue it"
                )
        tuned.append(TunedRun(name_of[key], key, tuple(sorted(candidates, key=lambda candidate: candidate.rate))))

    rate_sets = {run.name: [candidate.rate for candidate in run.candidates] for run in tuned}
    if len({tuple(rates) for rates in rate_sets.values()}) > 1:
        problems.append(f"every tuned run must try the same rates, got {rate_sets}")

    if problems:
        raise ConfigError("The pilot does not fit the comparison:\n  " + "\n  ".join(problems))
    return tuned


def _encoder_rate(config: Config) -> float | None:
    """The encoder's learning rate as the config parsed it."""
    return config.optimization.group_overrides.get(ENCODER_GROUP, {}).get("learning_rate")


def _training_without_encoder_rate(config: Config) -> dict[str, Any]:
    """Everything that shapes the trained weights, apart from the encoder's learning rate."""
    identity = training_identity(config.raw)
    identity.pop(ENCODER_RATE, None)  # a comparison run may leave its encoder at the base rate
    return identity


def _resolved(section: dict[str, Any]) -> dict[str, Any]:
    """A data section with its split directory, if it has one, as an absolute path without links."""
    if "split_dir" not in section:
        return section
    return {**section, "split_dir": str(Path(section["split_dir"]).resolve())}


def _same_data(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether two data sections are the same, however their split directories are written."""
    return _resolved(first) == _resolved(second)


def _reads_same_scenes(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether two data sections read the same scenes: the same split directory, whatever else differs.

    Data without a split directory (generated scenes) is the same only if its settings are.
    """
    if "split_dir" in first and "split_dir" in second:
        return _resolved(first)["split_dir"] == _resolved(second)["split_dir"]
    return _same_data(first, second)


class _TrainingData:
    """Checks checkpoints against their training split, computing each split's fingerprint once.

    A fingerprint reads the metadata of every file of the split (for SRN Cars,
    about 250,000 files), and one call of the chooser checks every pilot and
    comparison checkpoint. Made anew for each call, so that it sees the files as they are.
    """

    def __init__(self) -> None:
        self._fingerprints: dict[str, dict] = {}

    def check(self, checkpoint: Path, config: Config) -> None:
        """Raises ``ConfigError`` if the checkpoint was trained on other data than the config's training split."""
        check_training_fingerprint(checkpoint, lambda: self._fingerprint(config))

    def _fingerprint(self, config: Config) -> dict:
        key = json.dumps(_resolved(config.raw["data"]["train"]), sort_keys=True, default=str)
        if key not in self._fingerprints:
            self._fingerprints[key] = build_dataset(config, "train").fingerprint()
        return self._fingerprints[key]


# ---------------------------------------------------------------- choosing


def choose_rates(tuned_runs: Sequence[TunedRun], training_data: _TrainingData | None = None) -> list[Choice]:
    """Apply the rule of the module docstring to every tuned run.

    Raises:
        ConfigError: If any pilot run has not trained exactly its steps, has a
            checkpoint trained with other settings or data than its config
            asks for, has no evaluation made from that checkpoint, or one whose
            software was not fully recorded; or if the candidates of one run
            started from different pretrained weights or were evaluated with
            different software.
    """
    training_data = training_data or _TrainingData()
    problems, choices = [], []
    for run in tuned_runs:
        scores, starting_weights, software = {}, {}, {}
        for candidate in run.candidates:
            try:
                scores[candidate.rate], starting_weights[candidate.key], software[candidate.key] = _score(
                    candidate, training_data
                )
            except ConfigError as error:
                problems.append(str(error))
        if len(scores) != len(run.candidates):
            continue
        first = run.candidates[0].key
        if any(weights != starting_weights[first] for weights in starting_weights.values()):
            problems.append(f"the pilot runs of '{run.name}' started from different pretrained weights; train again")
        elif any(record != software[first] for record in software.values()):
            evaluations = [str(RunFiles(candidate.config.run_dir).evaluation) for candidate in run.candidates]
            problems.append(
                f"the pilot runs of '{run.name}' were evaluated with different software (a package changed between "
                f"their evaluations). {_EVALUATE_AGAIN.format(files=', '.join(evaluations))}"
            )
        else:
            best = max(scores.values())
            rate = min(rate for rate, score in scores.items() if score >= best - TOLERANCE_DB)
            choices.append(Choice(run, scores, rate))
    if problems:
        raise ConfigError("The rates cannot be chosen yet:\n  " + "\n  ".join(problems))
    return choices


_EVALUATE_AGAIN = (
    "To score them again with the software installed now, delete {files} and run the same command again: the "
    "pilot's evaluate stage scores every run that has no evaluation.json (scripts/run_comparison.sh runs it first)."
)


def _score(candidate: Candidate, training_data: _TrainingData) -> tuple[float, Any, dict]:
    """The candidate's score, the pretrained weights it started from, and the software it was evaluated with."""
    config = candidate.config
    checkpoint = RunFiles(config.run_dir).checkpoint
    progress = read_progress(checkpoint)
    if progress is None or progress.step != config.training.num_steps:
        reached = 0 if progress is None else progress.step
        raise ConfigError(f"'{candidate.key}' has trained {reached:,} steps, not its {config.training.num_steps:,}")
    try:
        check_checkpoint_matches_config(checkpoint, config)
        training_data.check(checkpoint, config)
    except ConfigError as error:
        raise ConfigError(f"'{candidate.key}': {error}") from error
    status = evaluation_status(config)
    if not status.is_current:
        raise ConfigError(f"'{candidate.key}' has no evaluation of its final checkpoint ({status.reason})")
    software, reasons = read_software(status.content.get("software"))
    if reasons:
        raise ConfigError(
            f"'{candidate.key}': the software of its evaluation cannot be compared with that of the other rates: "
            + "; ".join(SOFTWARE_PROBLEMS[reason] for reason in reasons)
            + ". "
            + _EVALUATE_AGAIN.format(files=RunFiles(config.run_dir).evaluation)
        )
    means = [protocol["mean"][METRIC] for protocol in status.content["protocols"]]
    return sum(means) / len(means), read_checkpoint_starting_weights(checkpoint), software


# ---------------------------------------------------------------- applying the choice


def write_rates_into_suite(
    path: Path,
    choices: Sequence[Choice],
    overrides: Sequence[str] = (),
    check: Callable[[Suite], None] = lambda suite: None,
) -> Suite:
    """Write each chosen rate into the comparison's suite file, keeping its comments.

    Each run's block must contain one line of the form
    ``encoder: {learning_rate: <rate>, ...}``; only the number on it changes.
    The new file is loaded, and must have the chosen rates and pass ``check``,
    before it replaces the old one.

    Returns:
        The comparison suite as now written.

    Raises:
        ConfigError: If a run has no such line, the result does not load with
            the chosen rates, or ``check`` refuses it. The file is unchanged then.
    """
    original = path.read_text(encoding="utf-8")
    updated, written = _suite_with_rates(path, {c.run.name: (c.run.key, c.rate) for c in choices}, overrides)
    check(written)
    if updated != original:
        temporary = _temporary_copy(path)
        temporary.write_text(updated, encoding="utf-8", newline="")
        os.replace(temporary, path)
        logger.info("Wrote the chosen encoder learning rates into %s", path)
    return load_suite(path, overrides)


def _suite_with_rates(path: Path, rates: dict[str, tuple[str, float]], overrides: Sequence[str]) -> tuple[str, Suite]:
    """The suite file's text with rates written in, and the suite that text loads as.

    Args:
        rates: Run name in the file -> (its run key in the suite, the rate).

    Raises:
        ConfigError: If a run has no line for its rate, or a written rate does
            not take effect (another setting of the file overrides it).
    """
    text = path.read_text(encoding="utf-8")
    for name, (_, rate) in rates.items():
        text = _with_encoder_rate(text, name, rate, path)
    temporary = _temporary_copy(path)
    temporary.write_text(text, encoding="utf-8", newline="")
    try:
        suite = load_suite(temporary, overrides)
    finally:
        temporary.unlink(missing_ok=True)
    wrong = [name for name, (key, rate) in rates.items() if _encoder_rate(suite.runs[key]) != rate]
    if wrong:
        raise ConfigError(
            f"A rate written into {path} does not take effect for {', '.join(wrong)}: another setting in the "
            f"file overrides the line 'encoder: {{learning_rate: ...}}'."
        )
    return text, suite


def _temporary_copy(path: Path) -> Path:
    """Where a new version of a suite file is written first: next to it, so that its relative base paths resolve."""
    return path.with_name(f".{path.name}.new")


_RATE_LINE = re.compile(r"^(\s+encoder:\s*\{[^}#]*?\blearning_rate:\s*)([^,}\s]+)")


def _with_encoder_rate(text: str, run_name: str, rate: float, path: Path) -> str:
    """``text`` with the encoder learning rate of ``run_name``'s block set to ``rate``."""
    lines = text.splitlines(keepends=True)
    runs = [index for index, line in enumerate(lines) if re.match(r"runs:\s*(#.*)?$", line.rstrip())]
    first = runs[0] + 1 if len(runs) == 1 else len(lines)
    starts = [
        index
        for index in range(first, len(lines))
        if re.match(rf"  {re.escape(run_name)}:\s*(#.*)?$", lines[index].rstrip())
    ]
    found = []
    if len(starts) == 1:
        # The block ends at the next line indented by two spaces or less: the next run, or a comment between runs.
        for index in range(starts[0] + 1, len(lines)):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip(" ")) <= 2:
                break
            if _RATE_LINE.match(line):
                found.append(index)
    if len(found) != 1:
        raise ConfigError(
            f"run '{run_name}' of {path} needs one line 'encoder: {{learning_rate: ..., ...}}' in its own block, "
            "where the chosen rate is written"
        )
    index = found[0]
    lines[index] = _RATE_LINE.sub(lambda match: match.group(1) + _yaml_float(rate), lines[index], count=1)
    return "".join(lines)


def _yaml_float(value: float) -> str:
    """The shortest text YAML reads back as exactly ``value``, in the style of the suite files: 3.0e-5."""
    for digits in range(17):
        mantissa, exponent = f"{value:.{digits}e}".split("e")
        if float(f"{mantissa}e{exponent}") == value:
            break
    if "." not in mantissa:
        mantissa += ".0"
    return f"{mantissa}e{int(exponent):+d}"  # YAML 1.1 reads an exponent as a number only with its sign


def check_transfers(comparison: Suite, choices: Sequence[Choice], training_data: _TrainingData | None = None) -> None:
    """Check, before anything changes, that every chosen run can be continued by the comparison.

    ``comparison`` is the suite with the chosen rates. A comparison run that
    already has a checkpoint must have been trained with its settings and its
    training data, and not past its number of steps (it is kept); otherwise
    the chosen pilot run's checkpoint must fit them.

    Raises:
        ConfigError: Naming the runs that do not fit.
    """
    training_data = training_data or _TrainingData()
    problems = []
    for choice in choices:
        config = comparison.runs[choice.run.key]
        destination = RunFiles(config.run_dir).checkpoint
        source = destination if destination.exists() else RunFiles(choice.winner.config.run_dir).checkpoint
        try:
            check_checkpoint_matches_config(source, config)
            training_data.check(source, config)
        except ConfigError as error:
            problems.append(f"'{choice.run.key}': {error}")
            continue
        progress = read_progress(source)
        if progress is not None and progress.step > config.training.num_steps:
            # It would be kept as finished, and the comparison would compare models trained for different lengths.
            problems.append(
                f"'{choice.run.key}' has trained {progress.step:,} steps, more than the {config.training.num_steps:,} "
                "the comparison asks for; ask for at least as many (training.num_steps)"
            )
    if problems:
        raise ConfigError("Nothing was changed:\n  " + "\n  ".join(problems))


def continue_chosen_runs(comparison: Suite, choices: Sequence[Choice]) -> None:
    """Copy each chosen pilot run's checkpoint and training curves into its comparison run.

    ``comparison`` is the suite with the chosen rates. A comparison run that
    already has a checkpoint trained with its settings keeps it (for example
    after an earlier call). Each copy is recorded in the run's ``TRANSFER_FILE``.

    Raises:
        ConfigError: If a checkpoint does not fit the comparison run's settings.
    """
    for choice in choices:
        config = comparison.runs[choice.run.key]
        destination = RunFiles(config.run_dir)
        if destination.checkpoint.exists():
            check_checkpoint_matches_config(destination.checkpoint, config)
            save_config(config, destination.config)  # in case an earlier call stopped before saving it
            logger.info("'%s' already has a checkpoint trained with its settings; keeping it.", choice.run.key)
            continue

        source = RunFiles(choice.winner.config.run_dir)
        config.run_dir.mkdir(parents=True, exist_ok=True)
        if source.tensorboard.is_dir():
            shutil.copytree(source.tensorboard, destination.tensorboard, dirs_exist_ok=True)
        checkpoints = destination.checkpoint.parent
        partial = checkpoints.with_name(f"{checkpoints.name}.partial")
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(source.checkpoint.parent, partial)
        copy = partial / destination.checkpoint.name
        check_checkpoint_matches_config(copy, config)
        record = {"pilot_run": choice.winner.key, "step": read_progress(copy).step}
        write_json({**record, "checkpoint_sha256": checkpoint_digest(copy)}, config.run_dir / TRANSFER_FILE)
        save_config(config, destination.config)  # also when the copy already has every step the run asks for
        # The checkpoint comes last and appears at once: a checkpoint in place means its record and config are too.
        if checkpoints.is_dir():
            shutil.rmtree(checkpoints)  # holds no checkpoint (checked above)
        os.replace(partial, checkpoints)
        logger.info("'%s' continues from pilot run '%s'.", choice.run.key, choice.winner.key)


def save_choices(pilot: Suite, comparison: Suite, choices: Sequence[Choice]) -> Path:
    """Write the choice and the scores it was made from into the pilot's suite directory.

    ``continued_from`` is the comparison run's ``TRANSFER_FILE``: the pilot run
    copied into it, at which step, and the checkpoint's SHA-256; ``None`` if
    the comparison run trained its checkpoint itself.
    """
    path = pilot.directory / CHOICE_FILE
    runs = {}
    for choice in choices:
        transfer = comparison.runs[choice.run.key].run_dir / TRANSFER_FILE
        runs[choice.run.name] = {
            "chosen_rate": choice.rate,
            "chosen_pilot_run": choice.winner.key,
            "comparison_run": choice.run.key,
            "continued_from": read_json(transfer) if transfer.is_file() else None,
            "candidates": [
                {"pilot_run": candidate.key, "rate": candidate.rate, METRIC: choice.scores[candidate.rate]}
                for candidate in choice.run.candidates
            ],
        }
    rule = f"lowest rate whose mean {METRIC} over the evaluation protocols is within {TOLERANCE_DB} dB of the best"
    write_json({"rule": rule, "runs": runs}, path)
    return path


def choose_and_continue(
    pilot_path: Path,
    comparison_path: Path,
    overrides: Sequence[str] = (),
    comparison_overrides: Sequence[str] = (),
) -> list[Choice]:
    """The whole command: choose every rate, write it, continue the chosen runs, and save the choice.

    Args:
        overrides: Applied to both suites, e.g. the learning-rate schedule
            (``optimization.decay_steps``), which the pilot shares with the
            comparison.
        comparison_overrides: Applied to the comparison only, after
            ``overrides``: its step budget (``training.num_steps``), which the
            pilot does not share.
    """
    pilot = load_suite(pilot_path, overrides)
    overrides = [*overrides, *comparison_overrides]  # from here on, only the comparison is loaded
    tuned = check_pilot(pilot, load_suite(comparison_path, overrides), comparison_path, overrides)
    training_data = _TrainingData()
    choices = choose_rates(tuned, training_data)
    for choice in choices:
        scores = ", ".join(
            f"{rate:g}: {score:.3f}" + (" (chosen)" if rate == choice.rate else "")
            for rate, score in sorted(choice.scores.items())
        )
        logger.info("%s, mean %s by encoder learning rate: %s", choice.run.name, METRIC, scores)

    comparison = write_rates_into_suite(
        comparison_path, choices, overrides, check=lambda written: check_transfers(written, choices, training_data)
    )
    continue_chosen_runs(comparison, choices)
    logger.info("Saved the choice to %s", save_choices(pilot, comparison, choices))
    return choices
