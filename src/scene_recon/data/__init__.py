"""Datasets. Importing this package registers every built-in dataset."""

from scene_recon.data import srn, synthetic  # noqa: F401  (imported for registration)
from scene_recon.data.base import SceneDataset
from scene_recon.data.loading import training_batches
from scene_recon.data.schedule import TrainingSchedule

__all__ = ["SceneDataset", "TrainingSchedule", "training_batches"]
