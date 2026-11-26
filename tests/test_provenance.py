"""A result must describe the experiment that produced it.

Each test reproduces a scenario from the repository review in which a result
could be attributed to the wrong training settings, checkpoint or device.
"""

import zipfile
from pathlib import Path

import pytest
import torch

from scene_recon import evaluation as evaluation_module
from scene_recon import provenance as provenance_module
from scene_recon.checkpoints import COMPATIBILITY_VERSION
from scene_recon.config import Config, parse_config
from scene_recon.data.synthetic import SyntheticSpheresDataset
from scene_recon.provenance import (
    evaluation_status,
    profile_status,
    profiling_settings,
    training_identity,
)
from scene_recon.registry import ConfigError
from scene_recon.run_files import RunFiles
from scene_recon.stages import run_evaluation, run_profiling
from scene_recon.training import Trainer

from conftest import READ_CPU_MODEL, write_srn_image


def _changed(config: Config, section: str, **changes) -> Config:
    """A copy of ``config`` with some settings of one section replaced."""
    return parse_config({**config.raw, section: {**config.raw[section], **changes}})


def _with_normalization(config: Config, name: str) -> Config:
    model = {**config.raw["model"], "encoder": {**config.raw["model"]["encoder"], "input_normalization": name}}
    return parse_config({**config.raw, "model": model})


# ---------------------------------------------------------------- training identity (F1)


@pytest.mark.parametrize(
    "change",
    [
        lambda config: _changed(config, "optimization", learning_rate=0.01),
        lambda config: _changed(config, "experiment", seed=99),
        lambda config: _changed(config, "training", rays_per_scene=64),
        lambda config: _changed(config, "renderer", num_coarse_samples=16),
        lambda config: _with_normalization(config, "imagenet"),
    ],
    ids=["learning_rate", "seed", "rays_per_scene", "renderer", "input_normalization"],
)
def test_resuming_with_other_training_settings_is_refused(smoke_config: Config, change):
    Trainer(smoke_config).train()
    files = RunFiles(smoke_config.run_dir)
    checkpoint_before, config_before = files.checkpoint.read_bytes(), files.config.read_text()
    changed = _changed(change(smoke_config), "training", num_steps=smoke_config.training.num_steps + 2)

    with pytest.raises(ConfigError, match="was trained with different settings"):
        Trainer(changed, resume=True)

    # Nothing about the run was rewritten to claim the new settings.
    assert files.checkpoint.read_bytes() == checkpoint_before
    assert files.config.read_text() == config_before


def test_continuation_settings_may_change(smoke_config: Config):
    Trainer(smoke_config).train()
    longer = _changed(smoke_config, "training", num_steps=smoke_config.training.num_steps + 2, log_every=1)
    longer = _changed(longer, "evaluation", metrics=["psnr"])

    trainer = Trainer(longer, resume=True)
    trainer.train()

    assert trainer.progress.step == longer.training.num_steps


def test_evaluating_a_checkpoint_with_other_model_settings_is_refused(smoke_config: Config):
    Trainer(smoke_config).train()

    with pytest.raises(ConfigError, match="model.encoder.input_normalization: trained with none, requested imagenet"):
        run_evaluation(_with_normalization(smoke_config, "imagenet"))


def test_training_identity_ignores_only_continuation_settings(smoke_config: Config):
    identity = training_identity(smoke_config.raw)

    assert "optimization.learning_rate" in identity and "experiment.seed" in identity
    assert "model.decoder.hidden_width" in identity and "data.train.seed" in identity
    assert "training.num_steps" not in identity and "evaluation.metrics" not in identity
    assert "experiment.device" not in identity and "data.test.seed" not in identity


# ---------------------------------------------------------------- evaluation freshness (F2)


def test_an_evaluation_of_another_checkpoint_at_the_same_step_is_not_reused(smoke_config: Config, tmp_path: Path):
    Trainer(smoke_config).train()
    files = RunFiles(smoke_config.run_dir)

    # A second checkpoint of this run at the same step, with different weights.
    state = torch.load(files.checkpoint, weights_only=True)
    state["model"] = {name: value + 0.05 if value.is_floating_point() else value for name, value in state["model"].items()}
    alternate = tmp_path / "alternate.pt"
    torch.save(state, alternate)

    run_evaluation(smoke_config)
    assert evaluation_status(smoke_config).is_current

    run_evaluation(smoke_config, alternate)  # writes its results into the run directory
    status = evaluation_status(smoke_config)

    assert status.state == "stale" and "different checkpoint" in status.reason


def test_an_evaluation_with_other_settings_is_stale(smoke_config: Config):
    Trainer(smoke_config).train()
    run_evaluation(smoke_config)

    status = evaluation_status(_changed(smoke_config, "evaluation", source_view_sets=[[0, 6]]))

    assert status.state == "stale" and "different evaluation" in status.reason


def test_a_damaged_evaluation_file_is_reported_as_unreadable(smoke_config: Config):
    Trainer(smoke_config).train()
    run_evaluation(smoke_config)
    RunFiles(smoke_config.run_dir).evaluation.write_text('{"checkpoint":')

    assert evaluation_status(smoke_config).state == "unreadable"


def test_nan_metrics_are_never_written_as_results(smoke_config: Config, monkeypatch):
    Trainer(smoke_config).train()
    monkeypatch.setattr(evaluation_module.Evaluator, "evaluate", _evaluate_with_nan(evaluation_module.Evaluator.evaluate))

    with pytest.raises(RuntimeError, match="non-finite"):
        run_evaluation(smoke_config)
    assert not RunFiles(smoke_config.run_dir).evaluation.exists()


def _evaluate_with_nan(real_evaluate):
    def evaluate(self, *args, **kwargs):
        result = real_evaluate(self, *args, **kwargs)
        result.mean["psnr"] = float("nan")
        return result

    return evaluate


# ---------------------------------------------------------------- profiling freshness (F9)


def test_a_profile_from_another_device_is_stale(smoke_config: Config):
    run_profiling(smoke_config)
    assert profile_status(smoke_config).is_current

    on_gpu = _changed(smoke_config, "experiment", device="cuda:0")

    assert profiling_settings(on_gpu) != profiling_settings(smoke_config)
    assert profile_status(on_gpu).state == "stale"


# ---------------------------------------------------------------- dataset identity (I4)


def _pretend_the_data_changed(monkeypatch) -> None:
    """Same config, different content: as if the dataset directory had been replaced."""
    real_fingerprint = SyntheticSpheresDataset.fingerprint

    def changed_fingerprint(self):
        fingerprint = real_fingerprint(self)
        return {**fingerprint, "seed": fingerprint["seed"] + 1000}  # same format, other content

    monkeypatch.setattr(SyntheticSpheresDataset, "fingerprint", changed_fingerprint)


def test_an_evaluation_on_other_test_data_is_stale(smoke_config: Config, monkeypatch):
    Trainer(smoke_config).train()
    run_evaluation(smoke_config)

    _pretend_the_data_changed(monkeypatch)

    status = evaluation_status(smoke_config)
    assert status.state == "stale" and "other test data" in status.reason


def test_resuming_on_other_training_data_is_refused(smoke_config: Config, monkeypatch):
    Trainer(smoke_config).train()
    _pretend_the_data_changed(monkeypatch)

    with pytest.raises(ConfigError, match="trained on different data"):
        Trainer(_changed(smoke_config, "training", num_steps=smoke_config.training.num_steps + 2), resume=True)


def test_results_record_the_data_they_came_from(smoke_config: Config):
    Trainer(smoke_config).train()
    report = run_evaluation(smoke_config)
    state = torch.load(RunFiles(smoke_config.run_dir).checkpoint, weights_only=True)

    expected = {"dataset": "synthetic_spheres", "num_scenes": 32, "num_views": 12, "image_size": 16, "seed": 0}
    assert state["training_data"] == expected
    assert report["settings"]["test_data_fingerprint"]["seed"] == 2


# ---------------------------------------------------------------- profiling workload (R8)


def test_a_profile_is_stale_after_the_images_are_replaced_by_larger_ones(srn_config: Config):
    """The review's reproduction: 8x8 images replaced by 16x16 ones at the same paths, config unchanged."""
    run_profiling(srn_config)
    assert profile_status(srn_config).is_current

    test_dir = Path(srn_config.data["test"]["split_dir"])
    for image in test_dir.glob("*/rgb/*.png"):
        write_srn_image(image, image_size=16, grey_level=50)

    status = profile_status(srn_config)
    assert status.state == "stale" and "test data" in status.reason
    assert run_profiling(srn_config)["image_size"] == [16, 16]
    assert profile_status(srn_config).is_current


# ---------------------------------------------------------------- older fingerprint formats (#3 compatibility note)


def test_a_checkpoint_with_an_older_fingerprint_format_is_explained_not_called_different_data(srn_config: Config):
    from scene_recon.stages import check_run_can_start

    Trainer(srn_config).train()
    checkpoint = RunFiles(srn_config.run_dir).checkpoint
    state = torch.load(checkpoint, weights_only=True)
    # The format of the previous version: names and sizes only, which cannot confirm the content.
    state["training_data"] = {"dataset": "srn", "num_scenes": 2, "files_sha256": "0" * 64}
    torch.save(state, checkpoint)

    with pytest.raises(ConfigError, match="older fingerprint format") as error:
        check_run_can_start(srn_config, ["train"])
    assert "different data" not in str(error.value)
    with pytest.raises(ConfigError, match="older fingerprint format"):
        Trainer(_changed(srn_config, "training", num_steps=srn_config.training.num_steps + 1), resume=True)

    run_evaluation(srn_config)  # evaluation does not need the training data
    assert evaluation_status(srn_config).is_current


# ---------------------------------------------------------------- code compatibility (review F19)


def _rewrite_checkpoint_version(checkpoint: Path, version: int | None) -> None:
    """Make a checkpoint look as if it was written by another version of the code (None: before versions existed)."""
    state = torch.load(checkpoint, weights_only=True)
    if version is None:
        del state["compatibility_version"]
    else:
        state["compatibility_version"] = version
    torch.save(state, checkpoint)


@pytest.mark.parametrize("version", [None, COMPATIBILITY_VERSION - 1], ids=["no_version", "older_version"])
def test_a_checkpoint_from_older_code_is_refused_everywhere(srn_config: Config, version):
    from scene_recon.stages import check_run_can_start

    Trainer(srn_config).train()
    checkpoint = RunFiles(srn_config.run_dir).checkpoint
    _rewrite_checkpoint_version(checkpoint, version)

    with pytest.raises(ConfigError, match="older version of the code") as error:
        check_run_can_start(srn_config, ["train", "evaluate", "profile"])
    assert "Train it again under a new name" in str(error.value) and "delete" not in str(error.value)
    with pytest.raises(ConfigError, match="older version of the code"):
        Trainer(_changed(srn_config, "training", num_steps=srn_config.training.num_steps + 1), resume=True)
    with pytest.raises(ConfigError, match="older version of the code"):
        run_evaluation(srn_config)
    with pytest.raises(ConfigError, match="older version of the code"):
        run_profiling(srn_config)


@pytest.mark.parametrize("version", [2, 3, 4])
def test_a_checkpoint_of_a_retired_version_is_refused(smoke_config: Config, version: int):
    """Version 2 (review F64) and 3 (the first object-box rule, review F196) trained differently."""
    from scene_recon.stages import check_run_can_start

    assert COMPATIBILITY_VERSION > 4
    Trainer(smoke_config).train()
    _rewrite_checkpoint_version(RunFiles(smoke_config.run_dir).checkpoint, version)

    with pytest.raises(ConfigError, match=f"compatibility version {version}"):
        check_run_can_start(smoke_config, ["train"])


def test_a_checkpoint_from_newer_code_is_refused(smoke_config: Config):
    Trainer(smoke_config).train()
    _rewrite_checkpoint_version(RunFiles(smoke_config.run_dir).checkpoint, COMPATIBILITY_VERSION + 1)

    with pytest.raises(ConfigError, match="newer version of the code"):
        run_evaluation(smoke_config)


def test_results_from_older_code_are_out_of_date(smoke_config: Config, monkeypatch):
    Trainer(smoke_config).train()
    run_evaluation(smoke_config)
    run_profiling(smoke_config)
    assert evaluation_status(smoke_config).is_current and profile_status(smoke_config).is_current

    # The code moves on to the next version; the results on disk were made with the previous one.
    monkeypatch.setattr(provenance_module, "COMPATIBILITY_VERSION", COMPATIBILITY_VERSION + 1)

    for status in (evaluation_status(smoke_config), profile_status(smoke_config)):
        assert status.state == "stale" and "older version of the code" in status.reason


# ---------------------------------------------------------------- starting weights (review B(b))


def test_checkpoints_record_the_pretrained_weights_the_run_started_from(smoke_config: Config, monkeypatch):
    from scene_recon.checkpoints import read_checkpoint_starting_weights
    from scene_recon.models.testing.tiny_encoder import TinyConvEncoder

    checkpoint = RunFiles(smoke_config.run_dir).checkpoint
    first = {"source": "weights/encoder.pth", "sha256": "1" * 64}
    monkeypatch.setattr(TinyConvEncoder, "pretrained_weights", first)
    Trainer(smoke_config).train()
    assert read_checkpoint_starting_weights(checkpoint) == {"encoder": first}

    # The weights file is replaced before the run is resumed; the run still started from the first one.
    monkeypatch.setattr(TinyConvEncoder, "pretrained_weights", {"source": "weights/encoder.pth", "sha256": "2" * 64})
    Trainer(_changed(smoke_config, "training", num_steps=smoke_config.training.num_steps + 2), resume=True).train()
    assert read_checkpoint_starting_weights(checkpoint) == {"encoder": first}


@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_a_weights_record_a_checkpoint_could_not_hold_is_refused_before_training(
    smoke_config: Config, monkeypatch, part: str
):
    """Review F44 and F58: suite preflight and check-model find it in any part of the model."""
    from scene_recon.models.testing.tiny_decoder import TinyPooledMLPDecoder
    from scene_recon.models.testing.tiny_encoder import TinyConvEncoder
    from scene_recon.stages import check_model, check_run_can_start

    owner = {"encoder": TinyConvEncoder, "decoder": TinyPooledMLPDecoder}[part]
    monkeypatch.setattr(owner, "pretrained_weights", {"source": Path("weights/part.pth"), "sha256": "1" * 64}, raising=False)

    with pytest.raises(ConfigError, match="has pretrained_weights"):
        check_run_can_start(smoke_config, ["train"])
    with pytest.raises(ConfigError, match="has pretrained_weights"):
        check_model(smoke_config, stages=["train"])
    assert not RunFiles(smoke_config.run_dir).checkpoint.exists()


def test_a_model_that_starts_from_random_values_records_no_pretrained_weights(smoke_config: Config):
    from scene_recon.checkpoints import read_checkpoint_starting_weights

    Trainer(smoke_config).train()

    assert read_checkpoint_starting_weights(RunFiles(smoke_config.run_dir).checkpoint) == {}


# ---------------------------------------------------------------- software versions (review F13)


def test_the_evaluation_software_record_lists_every_installed_package():
    """Review F66 and F67: also packages nobody names, such as SciPy inside scikit-image's SSIM."""
    from scene_recon.environment import evaluation_software

    software = evaluation_software(torch.device("cpu"))

    assert {"torch", "numpy", "scipy", "scikit-image", "timm"} <= software["packages"].keys()
    assert software["python"] and software["cuda"] and software["device_type"] == "cpu"
    assert software["device"].endswith(f", {torch.get_num_threads()} threads")  # the CPU's model comes first


class _BrokenDistribution:
    """An installed package whose metadata cannot be read."""

    _path = "site-packages/broken-1.0.dist-info"

    def __init__(self, error: Exception | None = None, name: str | None = "broken", version: str | None = "1.0"):
        self.error, self.name, self.version = error, name, version

    @property
    def metadata(self):
        if self.error:
            raise self.error
        return {"Name": self.name}


NOT_UTF8 = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")


@pytest.mark.parametrize(
    "broken",
    [
        _BrokenDistribution(NOT_UTF8),
        _BrokenDistribution(OSError("unreadable")),
        _BrokenDistribution(zipfile.BadZipFile("Bad CRC-32")),  # review F86: a zipped package
        _BrokenDistribution(name=None),
        _BrokenDistribution(name="not a name"),
        _BrokenDistribution(version="\n"),  # review F83: a blank continuation line after "Version:"
        _BrokenDistribution(version=None),
    ],
    ids=["not_utf8", "unreadable", "bad_archive", "no_name", "invalid_name", "blank_version", "no_version"],
)
def test_an_unreadable_package_makes_the_package_list_unknown(monkeypatch, caplog, broken):
    """Review F77, F83 and F86: the package list is marked unknown, and the package named."""
    from scene_recon import environment as environment_module

    monkeypatch.setattr(environment_module.importlib.metadata, "distributions", lambda: [broken])

    assert environment_module.installed_packages() is None
    assert "broken-1.0.dist-info cannot be read" in caplog.text


@pytest.mark.parametrize("packages_listed_first", [0, 1], ids=["at_once", "after_a_healthy_package"])
def test_a_failing_list_of_installed_packages_makes_the_package_list_unknown(
    monkeypatch, caplog, packages_listed_first
):
    """Review F93: the warning blames no package, not even the healthy one listed just before."""
    from scene_recon import environment as environment_module

    healthy = _BrokenDistribution(name="healthy")
    healthy._path = "site-packages/healthy-1.0.dist-info"

    def distributions():
        yield from [healthy][:packages_listed_first]
        raise OSError("a folder on the search path cannot be listed")

    monkeypatch.setattr(environment_module.importlib.metadata, "distributions", distributions)

    assert environment_module.installed_packages() is None
    assert "installed packages cannot be listed (OSError" in caplog.text
    assert "healthy" not in caplog.text and "Reinstall or remove" not in caplog.text


class _HiddenFolder(_BrokenDistribution):
    """A package whose metadata can be read, but not the folder it lies in."""

    @property
    def _path(self):
        raise OSError("the folder cannot be named")


def test_a_package_is_listed_even_if_its_folder_cannot_be_named(monkeypatch, caplog):
    """Review F100: the folder is looked up only for a warning, and the lookup never fails."""
    from scene_recon import environment as environment_module

    monkeypatch.setattr(environment_module.importlib.metadata, "distributions", lambda: [_HiddenFolder()])
    assert environment_module.installed_packages() == {"broken": "1.0"}

    monkeypatch.setattr(environment_module.importlib.metadata, "distributions", lambda: [_HiddenFolder(name=None)])
    assert environment_module.installed_packages() is None
    assert "installed package in an unknown folder cannot be read" in caplog.text


def test_a_package_finder_that_lists_nothing_does_not_cut_the_list_short(monkeypatch, caplog):
    """Review F101: None from a broken finder makes the package list unknown, never shorter."""
    from scene_recon import environment as environment_module

    monkeypatch.setattr(environment_module.importlib.metadata, "distributions", lambda: [_BrokenDistribution(), None])

    assert environment_module.installed_packages() is None
    assert "installed packages cannot be listed (TypeError" in caplog.text


def test_an_evaluation_is_written_when_package_metadata_cannot_be_read(smoke_config: Config, monkeypatch):
    """Review F77 and F85: the scores are kept; only the package list is unknown."""
    from scene_recon import environment as environment_module

    Trainer(smoke_config).train()

    def unreadable(name: str):
        raise NOT_UTF8

    monkeypatch.setattr(environment_module.importlib.metadata, "distributions", lambda: [_BrokenDistribution(NOT_UTF8)])
    monkeypatch.setattr(environment_module.importlib.metadata, "version", unreadable)
    environment_module.environment.cache_clear()  # read again, with the damaged metadata
    try:
        report = run_evaluation(smoke_config)
    finally:
        environment_module.environment.cache_clear()

    assert report["software"]["packages"] is None and report["environment"]["timm"] is None
    assert RunFiles(smoke_config.run_dir).evaluation.exists() and report["protocols"][0]["mean"]


def test_the_cpu_model_is_read_from_the_operating_system():
    """On Windows, macOS and Linux on x86 the model is named; elsewhere it may be unknown, but reading never fails."""
    model = READ_CPU_MODEL()

    assert model is None or (model and model == " ".join(model.split()))


def test_a_device_whose_model_cannot_be_named_is_recorded_as_unknown(monkeypatch):
    """Review F76: never a label several devices could share."""
    from scene_recon import environment as environment_module

    assert environment_module.device_model(torch.device("mps")) is None
    monkeypatch.setattr(environment_module, "cpu_model", lambda: None)
    assert environment_module.evaluation_software(torch.device("cpu"))["device"] is None
    monkeypatch.setattr(environment_module, "cpu_model", lambda: "Some CPU")
    assert environment_module.device_model(torch.device("cpu")) == f"Some CPU, {torch.get_num_threads()} threads"


def test_the_timm_version_is_recorded_and_decides_whether_a_profile_is_current(smoke_config: Config, monkeypatch):
    import timm

    from scene_recon import environment as environment_module

    assert environment_module.environment()["timm"] == timm.__version__
    run_profiling(smoke_config)
    assert profile_status(smoke_config).is_current

    real_version = environment_module.importlib.metadata.version
    monkeypatch.setattr(
        environment_module.importlib.metadata, "version", lambda name: "0.0.1" if name == "timm" else real_version(name)
    )

    assert profile_status(smoke_config).state == "stale"


@pytest.fixture
def damaged_timm_metadata(monkeypatch):
    """The metadata of timm cannot be read. The cached environment record is read again before and after,
    so that neither the real nor the damaged version leaks into another test."""
    from scene_recon import environment as environment_module

    real_version = environment_module.importlib.metadata.version

    def version(name: str) -> str:
        if name == "timm":
            raise NOT_UTF8
        return real_version(name)

    monkeypatch.setattr(environment_module.importlib.metadata, "version", version)
    environment_module.environment.cache_clear()
    yield
    environment_module.environment.cache_clear()  # the next test reads the real metadata again


def test_an_update_of_timm_is_seen_when_its_metadata_cannot_be_read(
    smoke_config: Config, monkeypatch, damaged_timm_metadata
):
    """Review F92: the version comes from the module itself, so an update still makes the saved speed stale."""
    import timm

    run_profiling(smoke_config)
    assert profile_status(smoke_config).is_current

    monkeypatch.setattr(timm, "__version__", "0.0.1")  # updated, and its metadata still damaged

    assert profile_status(smoke_config).state == "stale"


def test_a_speed_is_measured_again_while_the_version_of_timm_cannot_be_read(
    smoke_config: Config, monkeypatch, damaged_timm_metadata
):
    """Review F92: neither the metadata nor the module gives a version, so no saved speed can be checked."""
    from scene_recon import environment as environment_module

    real_import = environment_module.importlib.import_module

    def import_module(name: str, *arguments):
        if name == "timm":
            raise ImportError("timm cannot be imported")
        return real_import(name, *arguments)

    monkeypatch.setattr(environment_module.importlib, "import_module", import_module)
    assert environment_module.loaded_version("timm") == environment_module.UNREADABLE_VERSION

    run_profiling(smoke_config)
    status = profile_status(smoke_config)

    assert status.state == "stale" and "version of timm cannot be read" in status.reason
    assert environment_module.loaded_version("not-an-installed-package") is None
