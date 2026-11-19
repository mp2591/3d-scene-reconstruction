"""Statistics for comparing runs: spread over training seeds, and paired per-scene differences.

Two sources of noise blur a comparison between two architectures:

* **Training randomness.** The same architecture trained with another seed
  ends somewhat differently. A suite with several ``seeds`` measures this:
  ``summarize`` gives the mean and standard deviation of a run's score over
  its seeds.
* **Scene sampling.** The test set is a sample of scenes, and models differ
  more on some scenes than on others. Because every model is evaluated on
  the same scenes, the fair way to compare two models is per scene:
  ``paired_difference`` averages the per-scene differences and gives a 95%
  confidence interval for that average.

A difference between two runs is convincing only if its confidence interval
excludes zero and it is larger than the spread between seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

CONFIDENCE = 0.95


@dataclass(frozen=True)
class Summary:
    """A score summarised over the seeds of one run."""

    mean: float
    std: float
    """Sample standard deviation over seeds; 0 for a single seed."""

    count: int
    """Number of seeds with a result."""


@dataclass(frozen=True)
class PairedDifference:
    """The average per-scene difference between a run and a reference run."""

    mean: float
    low: float
    high: float
    """Bounds of the 95% confidence interval of ``mean``."""

    num_scenes: int

    def verdict(self, higher_is_better: bool | None) -> str:
        """``"better"`` or ``"worse"`` if the interval excludes zero, otherwise ``"no clear difference"``."""
        if higher_is_better is None or self.low <= 0 <= self.high:
            return "no clear difference"
        improved = self.mean > 0 if higher_is_better else self.mean < 0
        return "better" if improved else "worse"


def summarize(values: Sequence[float]) -> Summary | None:
    """Mean and sample standard deviation of a run's score over its seeds."""
    if not values:
        return None
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return Summary(mean=float(np.mean(values)), std=std, count=len(values))


def average_over_seeds(per_seed: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Per-scene scores averaged over seeds, for the scenes every seed has a score for."""
    if not per_seed:
        return {}
    common = set.intersection(*(set(scores) for scores in per_seed))
    return {scene: float(np.mean([scores[scene] for scores in per_seed])) for scene in sorted(common)}


def paired_difference(candidate: Mapping[str, float], reference: Mapping[str, float]) -> PairedDifference | None:
    """Average of ``candidate[scene] - reference[scene]`` over shared scenes, with a 95% confidence interval.

    The interval is the Student-t interval of the mean of the per-scene
    differences. Returns ``None`` with fewer than two shared scenes.
    """
    scenes = sorted(set(candidate) & set(reference))
    if len(scenes) < 2:
        return None
    differences = np.array([candidate[scene] - reference[scene] for scene in scenes])
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1)) / math.sqrt(len(scenes))
    half_width = _t_quantile(len(scenes) - 1) * standard_error
    return PairedDifference(mean=mean, low=mean - half_width, high=mean + half_width, num_scenes=len(scenes))


def _t_quantile(degrees_of_freedom: int) -> float:
    from scipy.stats import t  # imported here: only needed when a report is written

    return float(t.ppf(0.5 + CONFIDENCE / 2, degrees_of_freedom))
