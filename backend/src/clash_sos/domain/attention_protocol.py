"""Immutable row-slice contracts for attention experiments, without model dependencies.

Cache builders resolve the declared counts and row digests against Parquet before
training; this module validates the protocol structure and fit provenance.
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel, Sha256

PROTOCOL_VERSION = "attention-protocol:v1"
RowKey = tuple[datetime, str, str, int]
Partition = Literal["train", "validation", "test"]


def digest_row_keys(rows: Iterable[RowKey]) -> str:
    """Hash sorted, unique canonical row keys as a canonical JSON array."""
    digest = sha256()
    digest.update(b"[")
    previous: RowKey | None = None
    for row in rows:
        if row[0].tzinfo is None or row[0].utcoffset() is None:
            raise ValueError("row key timestamp must be timezone-aware")
        normalized = (row[0].astimezone(UTC), row[1], row[2], row[3])
        if previous is not None and normalized <= previous:
            raise ValueError("row keys must be unique and sorted")
        if previous is not None:
            digest.update(b",")
        digest.update(canonical_json_bytes(normalized))
        previous = normalized
    digest.update(b"]")
    return digest.hexdigest()


class AttentionSlice(ManifestModel):
    """Declared nonempty row set; its count and digest require data verification."""

    partition: Partition
    start: datetime
    end: datetime
    row_count: int = Field(gt=0)
    row_keys_sha256: Sha256

    @field_validator("start", "end")
    @classmethod
    def require_aware_utc(cls, value: datetime) -> datetime:
        """Normalize aware bounds so serialized protocols have one time basis."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("slice bounds must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Reject empty or reversed half-open intervals."""
        if self.start >= self.end:
            raise ValueError("slice start must precede end")
        return self


class AttentionProtocol(ManifestModel):
    """One fit population and its permitted evaluation stages."""

    protocol_version: Literal["attention-protocol:v1"] = PROTOCOL_VERSION
    family: Literal["temporal", "player_disjoint"]
    dataset_version: str = Field(min_length=1)
    balance_era_id: str = Field(min_length=1)
    processed_manifest_sha256: Sha256
    canonical_sha256: Sha256
    split_file: Literal["splits-temporal.parquet", "splits-player-disjoint.parquet"]
    split_sha256: Sha256
    encoding_sha256: Sha256
    mirror_seed: int = Field(ge=0)
    selection_fit: AttentionSlice
    watch: AttentionSlice
    refit: AttentionSlice
    development: AttentionSlice
    calibration: AttentionSlice | None = None
    reporting: AttentionSlice | None = None

    @model_validator(mode="after")
    def validate_stages(self) -> Self:
        """Enforce partition roles and the sole intentional refit overlap."""
        expected_split = (
            "splits-temporal.parquet"
            if self.family == "temporal"
            else "splits-player-disjoint.parquet"
        )
        if self.split_file != expected_split:
            raise ValueError("split file does not match protocol family")
        fit, watch, refit = self.selection_fit, self.watch, self.refit
        if any(item.partition != "train" for item in (fit, watch, refit)):
            raise ValueError("fit, watch, and refit must use the train partition")
        if fit.end != watch.start or (refit.start, refit.end) != (fit.start, watch.end):
            raise ValueError("refit must span contiguous fit and watch slices")
        if refit.row_count != fit.row_count + watch.row_count:
            raise ValueError("refit count must equal fit plus watch")
        if self.family == "player_disjoint" and self.development.partition != "validation":
            raise ValueError("player development must use player-validation")
        if self.development.partition == "test":
            raise ValueError("development cannot use the test partition")
        if self.family == "temporal" and refit.end > self.development.start:
            raise ValueError("temporal development must follow refit")
        if self.calibration is not None:
            if self.calibration.partition != self.development.partition:
                raise ValueError("calibration and development must share a partition")
            if self.calibration.end > self.development.start:
                raise ValueError("calibration must precede development")
            if self.calibration.partition == "train" and self.calibration.start < refit.end:
                raise ValueError("calibration cannot overlap refit")
        if self.reporting is not None:
            if self.reporting.partition != "test":
                raise ValueError("reporting must use the test partition")
            if self.family == "temporal" and self.reporting.start < self.development.end:
                raise ValueError("temporal reporting must follow development")
        return self

    def fit_sha256(self) -> str:
        """Identify weights' permitted source, excluding evaluation-only slices."""
        payload = self.model_dump(
            mode="python", exclude={"development", "calibration", "reporting"}
        )
        return sha256(canonical_json_bytes(payload)).hexdigest()


def require_matching_fit(fitted: AttentionProtocol, scored: AttentionProtocol) -> None:
    """Reject evaluation metadata that claims a different training population."""
    if fitted.fit_sha256() != scored.fit_sha256():
        raise ValueError("evaluation protocol does not match model fit")
