"""Streaming attention probability scores and predeclared matchup slices.

Training support is supplied by the caller; this module never derives groups
from validation outcomes. Legacy log-loss, Brier, and ECE definitions are reused.
"""

from dataclasses import dataclass
from math import isfinite

from clash_sos.domain.model_artifact import (
    ProbabilityMetricAccumulator,
    ProbabilityMetrics,
    log_loss,
)

RELIABILITY_BINS = 10
PROBABILITY_BANDS = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class ReliabilityBin:
    """Observed frequency and confidence for one fixed-width probability bin."""

    count: int
    mean_prediction: float | None
    mean_outcome: float | None


@dataclass(frozen=True)
class MatchupScoreSummary:
    """Probability quality, symmetric tie accuracy, and reliability counts."""

    metrics: ProbabilityMetrics
    accuracy_half_ties: float
    tie_count: int
    reliability: tuple[ReliabilityBin, ...]


@dataclass(frozen=True)
class MatchupEvaluation:
    """Full population plus nonempty slices defined before outcome inspection."""

    overall: MatchupScoreSummary
    groups: dict[str, MatchupScoreSummary]


class _ScoreAccumulator:
    """Accumulate legacy metrics and additional counts in one streaming pass."""

    def __init__(self) -> None:
        self.metrics = ProbabilityMetricAccumulator(bins=RELIABILITY_BINS)
        self.bin_count = [0] * RELIABILITY_BINS
        self.bin_probabilities = [0.0] * RELIABILITY_BINS
        self.bin_outcomes = [0.0] * RELIABILITY_BINS
        self.correct = 0
        self.ties = 0
        self.count = 0

    def update(self, label: int, probability: float) -> None:
        """Use the legacy bin boundaries, including a closed final bin."""
        self.metrics.update(label, probability)
        index = min(RELIABILITY_BINS - 1, int(probability * RELIABILITY_BINS))
        self.bin_count[index] += 1
        self.bin_probabilities[index] += probability
        self.bin_outcomes[index] += label
        self.ties += int(probability == 0.5)
        self.correct += int(
            (probability > 0.5 and label == 1) or (probability < 0.5 and label == 0)
        )
        self.count += 1

    def finalize(self) -> MatchupScoreSummary:
        """Return means only for occupied reliability bins."""
        reliability = tuple(
            ReliabilityBin(
                count=count,
                mean_prediction=self.bin_probabilities[index] / count if count else None,
                mean_outcome=self.bin_outcomes[index] / count if count else None,
            )
            for index, count in enumerate(self.bin_count)
        )
        return MatchupScoreSummary(
            metrics=self.metrics.finalize(),
            accuracy_half_ties=(self.correct + 0.5 * self.ties) / self.count,
            tie_count=self.ties,
            reliability=reliability,
        )


class MatchupEvaluationAccumulator:
    """Partition every row by novelty, support, and predicted probability."""

    def __init__(self) -> None:
        self._overall = _ScoreAccumulator()
        self._groups: dict[str, _ScoreAccumulator] = {}

    def update(
        self,
        *,
        label: int,
        probability: float,
        deck_a_support: int,
        deck_b_support: int,
        unordered_pair_support: int,
        comparator_probability: float | None = None,
    ) -> float | None:
        """Return paired per-row loss delta when a comparator is supplied."""
        if label not in (0, 1) or not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("evaluation requires a binary label and finite probability")
        if min(deck_a_support, deck_b_support, unordered_pair_support) < 0:
            raise ValueError("training support counts must be nonnegative")
        if comparator_probability is not None and (
            not isfinite(comparator_probability) or not 0 <= comparator_probability <= 1
        ):
            raise ValueError("comparator probability must be finite and in [0, 1]")

        known = int(deck_a_support > 0) + int(deck_b_support > 0)
        novelty = ("both_unseen", "one_unseen", "both_seen")[known]
        pair = "seen_pair" if unordered_pair_support else "unseen_pair"
        minimum = min(deck_a_support, deck_b_support)
        support = (
            "support_0"
            if minimum == 0
            else "support_1_3"
            if minimum < 4
            else "support_4_19"
            if minimum < 20
            else "support_20_plus"
        )
        band = next(
            index
            for index, upper in enumerate(PROBABILITY_BANDS)
            if probability < upper or index == len(PROBABILITY_BANDS) - 1
        )
        groups = (
            f"deck:{novelty}",
            f"pair:{pair}",
            f"support:{support}",
            f"probability:band_{band}",
        )
        self._overall.update(label, probability)
        for group in groups:
            self._groups.setdefault(group, _ScoreAccumulator()).update(label, probability)
        if comparator_probability is None:
            return None
        return log_loss((label,), (probability,)) - log_loss((label,), (comparator_probability,))

    def finalize(self) -> MatchupEvaluation:
        """Reject empty populations and return sorted, nonempty slice reports."""
        overall = self._overall.finalize()
        groups = {key: self._groups[key].finalize() for key in sorted(self._groups)}
        for prefix in ("deck:", "pair:", "support:", "probability:"):
            counted = sum(
                item.metrics.row_count for key, item in groups.items() if key.startswith(prefix)
            )
            if counted != overall.metrics.row_count:
                raise RuntimeError(f"evaluation {prefix} slices do not partition rows")
        return MatchupEvaluation(overall=overall, groups=groups)
