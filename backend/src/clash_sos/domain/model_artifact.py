"""Versioned matchup-baseline artifact contracts and probability metrics."""

from collections.abc import Sequence
from math import log
from typing import Literal, Self

from pydantic import Field, model_validator

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel, RelativePath, Sha256, validate_relative_path
from clash_sos.domain.matchup_baseline import (
    DEFAULT_MIRROR_SEED,
    DEFAULT_SMOOTHING_ALPHA,
    PROBABILITY_FLOOR,
)

DEFAULT_MODEL_VERSION = "kaggle-v6-ranked16-card-logodds-v1"
DEFAULT_PAIR_MODEL_VERSION = "kaggle-v6-ranked16-card-pair-v3"
MODEL_ARTIFACT_TYPE = "matchup_baseline"
REQUIRED_FILE_KINDS = ("card_catalog", "evaluation", "feature_schema", "predictor")
EVALUATION_SPLITS = ("temporal", "player_disjoint")
EVALUATION_PARTITIONS = ("train", "validation", "test")


def _require_aligned(labels: Sequence[int], probabilities: Sequence[float]) -> None:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not labels:
        raise ValueError("metrics require at least one observation")


def log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mean binary log loss with probabilities clipped away from 0 and 1."""
    _require_aligned(labels, probabilities)
    total = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        if label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        clipped = min(max(probability, PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)
        total += -(label * log(clipped) + (1 - label) * log(1.0 - clipped))
    return total / len(labels)


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mean squared error between labels and predicted probabilities."""
    _require_aligned(labels, probabilities)
    total = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        if label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        total += (probability - label) ** 2
    return total / len(labels)


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    """Weighted absolute gap between accuracy and confidence in equal-width bins."""
    _require_aligned(labels, probabilities)
    if bins < 1:
        raise ValueError("bins must be positive")
    totals = [0] * bins
    weighted_labels = [0.0] * bins
    weighted_probabilities = [0.0] * bins
    for label, probability in zip(labels, probabilities, strict=True):
        if label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        index = min(bins - 1, max(0, int(probability * bins)))
        totals[index] += 1
        weighted_labels[index] += label
        weighted_probabilities[index] += probability
    count = len(labels)
    error = 0.0
    for index, total in enumerate(totals):
        if total == 0:
            continue
        accuracy = weighted_labels[index] / total
        confidence = weighted_probabilities[index] / total
        error += abs(accuracy - confidence) * (total / count)
    return error


class ProbabilityMetrics(ManifestModel):
    """Held-out probability scores for one baseline on one split partition."""

    log_loss: float = Field(ge=0)
    brier_score: float = Field(ge=0)
    expected_calibration_error: float = Field(ge=0)
    row_count: int = Field(ge=0)


class ProbabilityMetricAccumulator:
    """Online log-loss, Brier, and ECE so scoring need not retain every row."""

    def __init__(self, *, bins: int = 10) -> None:
        if bins < 1:
            raise ValueError("bins must be positive")
        self._bins = bins
        self._log_loss_total = 0.0
        self._brier_total = 0.0
        self._count = 0
        self._bin_totals = [0] * bins
        self._bin_labels = [0.0] * bins
        self._bin_probabilities = [0.0] * bins

    def update(self, label: int, probability: float) -> None:
        """Add one labeled probability. Labels must be 0 or 1."""
        if label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        clipped = min(max(probability, PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)
        self._log_loss_total += -(label * log(clipped) + (1 - label) * log(1.0 - clipped))
        self._brier_total += (probability - label) ** 2
        index = min(self._bins - 1, max(0, int(probability * self._bins)))
        self._bin_totals[index] += 1
        self._bin_labels[index] += label
        self._bin_probabilities[index] += probability
        self._count += 1

    def finalize(self) -> ProbabilityMetrics:
        """Return mean metrics. Raises if no observations were added."""
        if self._count == 0:
            raise ValueError("metrics require at least one observation")
        error = 0.0
        for index, total in enumerate(self._bin_totals):
            if total == 0:
                continue
            accuracy = self._bin_labels[index] / total
            confidence = self._bin_probabilities[index] / total
            error += abs(accuracy - confidence) * (total / self._count)
        return ProbabilityMetrics(
            log_loss=self._log_loss_total / self._count,
            brier_score=self._brier_total / self._count,
            expected_calibration_error=error,
            row_count=self._count,
        )


class SplitEvaluation(ManifestModel):
    """Prior, exact-matchup, card-log-odds, and optional card-pair metrics for one slice."""

    split: Literal["temporal", "player_disjoint"]
    partition: Literal["train", "validation", "test"]
    prior: ProbabilityMetrics
    exact_matchup: ProbabilityMetrics
    card_log_odds: ProbabilityMetrics
    card_pair: ProbabilityMetrics | None = None


class EvaluationReport(ManifestModel):
    """Frozen evaluation output written beside the promoted predictor."""

    report_version: Literal[1] = 1
    promoted_model: Literal["card_log_odds", "card_pair"] = "card_log_odds"
    splits: tuple[SplitEvaluation, ...]

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        if not self.splits:
            raise ValueError("evaluation report requires at least one split")
        if self.promoted_model == "card_pair" and any(
            split.card_pair is None for split in self.splits
        ):
            raise ValueError("card_pair reports require card_pair metrics on every split")
        return self


class ModelOutputFile(ManifestModel):
    """Inventoried artifact file excluding manifest.json itself."""

    path: RelativePath
    kind: Literal["card_catalog", "evaluation", "feature_schema", "predictor"]
    size_bytes: int = Field(ge=0)
    sha256: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        validate_relative_path(self.path)
        return self


class ModelArtifactManifest(ManifestModel):
    """Immutable metadata for a published card-log-odds matchup artifact."""

    manifest_type: Literal["matchup_baseline"] = "matchup_baseline"
    manifest_version: Literal[1] = 1
    model_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    catalog_version: str = Field(min_length=1)
    balance_era_id: str = Field(min_length=1)
    smoothing_alpha: float = Field(default=DEFAULT_SMOOTHING_ALPHA, gt=0)
    mirror_seed: int = DEFAULT_MIRROR_SEED
    fit_split: Literal["temporal"] = "temporal"
    fit_partition: Literal["train"] = "train"
    files: tuple[ModelOutputFile, ...]

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        paths = tuple(file.path for file in self.files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ValueError("model files must have unique, sorted paths")
        for path in paths:
            if path == "manifest.json" or path.endswith("/manifest.json"):
                raise ValueError("model manifest must not inventory manifest.json")
        kinds = tuple(sorted(file.kind for file in self.files))
        if kinds != REQUIRED_FILE_KINDS:
            raise ValueError("model artifact must inventory each required file kind once")
        return self


def dump_model_manifest(manifest: ModelArtifactManifest) -> bytes:
    """Serialize a model artifact manifest as canonical JSON plus a newline."""
    return canonical_json_bytes(manifest.model_dump(mode="python")) + b"\n"


def dump_evaluation_report(report: EvaluationReport) -> bytes:
    """Serialize an evaluation report as canonical JSON plus a newline."""
    return canonical_json_bytes(report.model_dump(mode="python")) + b"\n"
