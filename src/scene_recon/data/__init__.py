"""Datasets. Importing this package registers every built-in dataset."""

from scene_recon.data import srn, synthetic  # noqa: F401  (imported for registration)
from scene_recon.data.base import SceneDataset
from scene_recon.data.loading import endless_training_batches

__all__ = ["SceneDataset", "endless_training_batches"]
