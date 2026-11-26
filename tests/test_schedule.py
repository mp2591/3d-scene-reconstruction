"""The training data must depend only on the seed and the step number.

The unit tests check the schedule itself; the end-to-end tests check the
guarantees that matter for comparing models: two different models see the
same data, and an interrupted run ends exactly where an uninterrupted one does.
"""

import hashlib
import random
import shutil
import sys
import time
from types import SimpleNamespace

import pytest
import torch

import scene_recon.training as training_module
from scene_recon.config import Config, parse_config
from scene_recon.data import TrainingSchedule, loading, training_batches
from scene_recon.data.synthetic import SyntheticSpheresDataset
from scene_recon.training import Trainer


def _schedule(seed: int = 0, num_scenes: int = 10, batch_size: int = 3) -> TrainingSchedule:
    scenes = SyntheticSpheresDataset(num_scenes=num_scenes, num_views=8, image_size=8, seed=0)
    return TrainingSchedule(scenes, seed=seed, batch_size=batch_size, views_per_scene=4, num_source_views=[1, 2, 3])


class TestSchedule:
    def test_same_seed_and_step_give_the_same_data(self):
        first, second = _schedule(seed=7), _schedule(seed=7)

        for step in (0, 5, 1234):
            assert first.scene_plan(step) == second.scene_plan(step)
            assert first.source_view_indices(step) == second.source_view_indices(step)
            assert torch.equal(
                torch.rand(5, generator=first.generator("rays", step)),
                torch.rand(5, generator=second.generator("rays", step)),
            )

    def test_different_seeds_give_different_data(self):
        assert _schedule(seed=1).scene_plan(0) != _schedule(seed=2).scene_plan(0)

    def test_each_epoch_visits_every_scene_at_most_once(self):
        schedule = _schedule(num_scenes=10, batch_size=3)  # 3 full batches per epoch, one scene left out

        for epoch in range(3):
            steps = range(epoch * 3, epoch * 3 + 3)
            scenes = [scene for step in steps for scene, _ in schedule.scene_plan(step)]
            assert len(scenes) == len(set(scenes)) == 9

    def test_fewer_scenes_than_batch_size_uses_all_of_them(self):
        schedule = _schedule(num_scenes=2, batch_size=4)

        assert sorted(scene for scene, _ in schedule.scene_plan(0)) == [0, 1]

    def test_views_and_source_views_are_valid(self):
        schedule = _schedule()

        for step in range(20):
            for _, views in schedule.scene_plan(step):
                assert len(views) == len(set(views)) == 4 and all(0 <= view < 8 for view in views)
            source = schedule.source_view_indices(step)
            assert 1 <= len(source) <= 3 and all(0 <= index < 4 for index in source)

    def test_global_random_state_has_no_influence(self):
        schedule = _schedule()
        torch.manual_seed(0)
        random.seed(0)
        before = (schedule.scene_plan(3), schedule.source_view_indices(3))
        torch.manual_seed(99)
        random.seed(99)
        torch.rand(1000)  # e.g. a model initialising its weights

        assert (schedule.scene_plan(3), schedule.source_view_indices(3)) == before

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_loader_yields_exactly_the_planned_batches(self, num_workers: int):
        schedule = _schedule()
        batches = training_batches(schedule, start_step=4, num_workers=num_workers)

        for step in (4, 5, 6):
            batch = next(batches)
            expected = [schedule.scenes.load_scene(scene, views) for scene, views in schedule.scene_plan(step)]
            assert torch.equal(batch.images, torch.cat([scene.images for scene in expected]))
        batches.close()


class TestSharedMemory:
    """Worker processes hand batches over through /dev/shm, which a container may have little of."""

    @pytest.fixture
    def linux(self, monkeypatch):
        monkeypatch.setattr(loading, "_hands_batches_over_in_shared_memory", lambda: True)

    @staticmethod
    def _free(monkeypatch, free_bytes: int) -> None:
        monkeypatch.setattr(loading.shutil, "disk_usage", lambda path: SimpleNamespace(free=free_bytes))

    @pytest.mark.parametrize("page", [4096, 65536])  # most processors; the GH200's Grace with 64 KB pages
    def test_needs_whole_pages_for_the_batches_on_their_way_and_the_locks(self, linux, monkeypatch, tmp_path, page):
        monkeypatch.setattr(loading.mmap, "PAGESIZE", page)
        schedule = _schedule(batch_size=3)  # 3 scenes, 4 views of 3x8x8 float32 images, poses, intrinsics, bounds
        tensor_bytes = [3 * 4 * 3 * 8 * 8 * 4, 3 * 4 * 16 * 4, 3 * 4 * 9 * 4, 3 * 4, 3 * 4]
        batch_pages = sum(-(-size // page) for size in tensor_bytes)
        workers = 2
        batches = workers * loading.BATCHES_AHEAD_PER_WORKER + 1
        needed = (batches * batch_pages + 3 * workers + 8) * page

        self._free(monkeypatch, needed)
        loading.check_shared_memory(schedule, start_step=0, num_workers=workers, folder=tmp_path)
        self._free(monkeypatch, needed - 1)
        with pytest.raises(RuntimeError, match=r"2 worker processes .* lower training.num_workers"):
            loading.check_shared_memory(schedule, start_step=0, num_workers=workers, folder=tmp_path)

    def test_a_linux_machine_without_the_folder_is_refused(self, linux, tmp_path):
        with pytest.raises(RuntimeError, match=r"has no .*missing.* lower training.num_workers"):
            loading.check_shared_memory(_schedule(), start_step=0, num_workers=2, folder=tmp_path / "missing")

    def test_nothing_is_checked_on_other_systems(self, monkeypatch, tmp_path):
        monkeypatch.setattr(loading, "_hands_batches_over_in_shared_memory", lambda: False)
        self._free(monkeypatch, 0)

        loading.check_shared_memory(_schedule(), start_step=0, num_workers=2, folder=tmp_path)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="workers use /dev/shm only on Linux")
    def test_a_closed_loader_gives_its_shared_memory_back(self):
        # The runs of a suite train one after another in one process, so each must find it free again.
        scenes = SyntheticSpheresDataset(num_scenes=10, num_views=8, image_size=64, seed=0)
        schedule = TrainingSchedule(scenes, seed=0, batch_size=3, views_per_scene=4, num_source_views=[1, 2, 3])
        batch_bytes = 3 * 4 * 3 * 64 * 64 * 4  # the images alone

        def used() -> int:
            return shutil.disk_usage(loading.SHARED_MEMORY).used

        before = used()
        for _ in range(2):
            batches = training_batches(schedule, start_step=0, num_workers=2)
            for _ in range(3):
                next(batches)
            time.sleep(1)  # each worker has loaded its next batch by now
            while_loading = used()
            batches.close()

            assert while_loading - before >= 2 * batch_bytes
            assert used() - before < batch_bytes

    def test_the_loader_checks_only_when_it_starts_workers(self, monkeypatch):
        checked = []
        monkeypatch.setattr(loading, "check_shared_memory", lambda *arguments: checked.append(arguments))

        for num_workers in (0, 2):
            batches = training_batches(_schedule(), start_step=0, num_workers=num_workers)
            next(batches)
            batches.close()

        assert [arguments[2] for arguments in checked] == [2]


# ---------------------------------------------------------------- end to end


def _config(smoke_config: Config, run_name: str, **training_changes) -> Config:
    raw = {
        **smoke_config.raw,
        "experiment": {**smoke_config.raw["experiment"], "run_name": run_name},
        "training": {**smoke_config.raw["training"], "validate_every": 1000, **training_changes},
    }
    return parse_config(raw)


def _record_training_data(monkeypatch) -> list[str]:
    """Record a fingerprint of every step's training data: loaded images, source views and supervised rays."""
    fingerprints: list[str] = []
    real_sample_training_rays = training_module.sample_training_rays

    def recording_sample_training_rays(batch, rays_per_scene, generator):
        rays, target_rgb = real_sample_training_rays(batch, rays_per_scene, generator)
        content = torch.cat([batch.images.flatten(), rays.directions.flatten(), target_rgb.flatten()])
        fingerprints.append(hashlib.sha256(content.cpu().numpy().tobytes()).hexdigest())
        return rays, target_rgb

    monkeypatch.setattr(training_module, "sample_training_rays", recording_sample_training_rays)
    return fingerprints


def _with_decoder_width(config: Config, hidden_width: int) -> Config:
    model = {**config.raw["model"], "decoder": {**config.raw["model"]["decoder"], "hidden_width": hidden_width}}
    return parse_config({**config.raw, "model": model})


def test_different_models_are_trained_on_identical_data(smoke_config: Config, monkeypatch):
    fingerprints = _record_training_data(monkeypatch)

    Trainer(_with_decoder_width(_config(smoke_config, "small"), hidden_width=8)).train()
    data_seen_by_small = list(fingerprints)
    fingerprints.clear()

    # A larger model draws far more random numbers to initialise its weights.
    Trainer(_with_decoder_width(_config(smoke_config, "large"), hidden_width=96)).train()

    assert len(data_seen_by_small) == smoke_config.training.num_steps
    assert fingerprints == data_seen_by_small


def _with_dropout(config: Config) -> Config:
    model = {**config.raw["model"], "decoder": {**config.raw["model"]["decoder"], "dropout": 0.3}}
    return parse_config({**config.raw, "model": model})


@pytest.mark.parametrize("num_workers", [0, 2])
def test_an_interrupted_run_ends_exactly_like_an_uninterrupted_one(smoke_config: Config, num_workers: int):
    """Holds for a model with dropout, i.e. randomness inside the model, and with worker processes."""

    def config(run_name: str, num_steps: int) -> Config:
        return _with_dropout(_config(smoke_config, run_name, num_steps=num_steps, num_workers=num_workers))

    uninterrupted = Trainer(config("uninterrupted", 6))
    uninterrupted.train()

    Trainer(config("interrupted", 3)).train()
    torch.manual_seed(12345)  # whatever happened to the global random state in between
    random.seed(12345)
    resumed = Trainer(config("interrupted", 6), resume=True)
    resumed.train()

    for name, weights in uninterrupted.model.state_dict().items():
        assert torch.equal(weights, resumed.model.state_dict()[name]), name
