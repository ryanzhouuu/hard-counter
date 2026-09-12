"""Processed-dataset manifest contracts distinct from raw audit manifests."""

from datetime import datetime
from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, model_validator

from clash_sos.domain.canonical import PlayerId, RecordIssue, RecordState
from clash_sos.domain.canonical_dataset import (
    CANONICAL_SCHEMA_VERSION,
    canonical_json_bytes,
)
from clash_sos.domain.manifests import ManifestModel, RelativePath, Sha256, validate_relative_path

FINALIZATION_ORDER = ("data_artifacts", "verification_report", "manifest")

PROCESSED_MANIFEST_TYPE = "processed"
DEFAULT_DATASET_VERSION = "kaggle-v6-ranked16-v1"
PREPARATION_POLICY_VERSION = "kaggle-v6-ranked16-policy:v1"
DEFAULT_PLAYER_HASH_SEED = 0
DEFAULT_PLAYER_TRAIN_MAX = 0.70
DEFAULT_PLAYER_VALIDATION_MAX = 0.85
TEMPORAL_PARTITIONS = ("train", "validation", "test")
PLAYER_PARTITIONS = ("train", "validation", "test")

INGESTION_ISSUES: tuple[RecordIssue, ...] = (
    RecordIssue.CONFLICTING_BATTLE,
    RecordIssue.DRAW_OUTCOME,
    RecordIssue.DUPLICATE_BATTLE,
    RecordIssue.IDENTICAL_PLAYERS,
    RecordIssue.INCOMPLETE_DECK,
    RecordIssue.INCOMPATIBLE_SOURCE_SCHEMA,
    RecordIssue.MALFORMED_ROW,
    RecordIssue.MALFORMED_TIMESTAMP,
    RecordIssue.NON_MAX_CARD_LEVEL,
    RecordIssue.REPEATED_CARD,
    RecordIssue.STALE_BALANCE_ERA,
    RecordIssue.UNSUPPORTED_MODE,
    RecordIssue.UNKNOWN_CARD,
)

RECORD_STATES: tuple[RecordState, ...] = (
    RecordState.INSUFFICIENT_DATA,
    RecordState.INVALID,
    RecordState.QUARANTINED,
    RecordState.UNSUPPORTED,
    RecordState.VALID,
)


class StateCount(ManifestModel):
    state: RecordState
    count: int = Field(ge=0)


class IssueCount(ManifestModel):
    issue: RecordIssue
    count: int = Field(ge=0)


class EraCount(ManifestModel):
    era_id: str = Field(min_length=1)
    count: int = Field(ge=0)


class DispositionSummary(ManifestModel):
    source_row_count: int = Field(ge=0)
    states: tuple[StateCount, ...]
    issues: tuple[IssueCount, ...]
    duplicate_row_count: int = Field(ge=0)
    conflict_row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        state_keys = tuple(count.state for count in self.states)
        if state_keys != RECORD_STATES:
            raise ValueError("disposition states must list every record state once, sorted")
        issue_keys = tuple(count.issue for count in self.issues)
        if issue_keys != INGESTION_ISSUES:
            raise ValueError("disposition issues must list every ingestion issue once, sorted")
        if sum(count.count for count in self.states) != self.source_row_count:
            raise ValueError("disposition state counts must sum to source_row_count")
        issue_map = {count.issue: count.count for count in self.issues}
        if self.duplicate_row_count != issue_map[RecordIssue.DUPLICATE_BATTLE]:
            raise ValueError("duplicate_row_count must match duplicate_battle issue count")
        if self.conflict_row_count != issue_map[RecordIssue.CONFLICTING_BATTLE]:
            raise ValueError("conflict_row_count must match conflicting_battle issue count")
        return self


class AcceptedSummary(ManifestModel):
    row_count: int = Field(ge=0)
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None
    eras: tuple[EraCount, ...]

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        era_ids = [era.era_id for era in self.eras]
        if era_ids != sorted(set(era_ids)):
            raise ValueError("accepted era counts must have unique, sorted era IDs")
        if sum(era.count for era in self.eras) != self.row_count:
            raise ValueError("accepted era counts must sum to row_count")
        for timestamp in (self.timestamp_min, self.timestamp_max):
            if timestamp is not None and (
                timestamp.tzinfo is None or timestamp.utcoffset() is None
            ):
                raise ValueError("accepted timestamps must be timezone-aware")
        if self.row_count == 0:
            if self.timestamp_min is not None or self.timestamp_max is not None:
                raise ValueError("empty accepted summary cannot include timestamp bounds")
        else:
            if self.timestamp_min is None or self.timestamp_max is None:
                raise ValueError("accepted summary with rows requires timestamp bounds")
            if self.timestamp_min > self.timestamp_max:
                raise ValueError("timestamp_min must not be later than timestamp_max")
        return self


class TemporalPartitionSummary(ManifestModel):
    partition: Literal["train", "validation", "test"]
    row_count: int = Field(ge=0)
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        for timestamp in (self.timestamp_min, self.timestamp_max):
            if timestamp is not None and (
                timestamp.tzinfo is None or timestamp.utcoffset() is None
            ):
                raise ValueError("temporal partition timestamps must be timezone-aware")
        if self.row_count == 0:
            if self.timestamp_min is not None or self.timestamp_max is not None:
                raise ValueError("empty temporal partition cannot include timestamp bounds")
        else:
            if self.timestamp_min is None or self.timestamp_max is None:
                raise ValueError("temporal partition with rows requires timestamp bounds")
            if self.timestamp_min > self.timestamp_max:
                raise ValueError(
                    "temporal partition timestamp_min must not be later than timestamp_max"
                )
        return self


class TemporalSplitManifest(ManifestModel):
    train_end: datetime
    validation_end: datetime
    semantics: Literal["half_open"] = "half_open"
    partitions: tuple[TemporalPartitionSummary, ...]

    @model_validator(mode="after")
    def validate_split(self) -> Self:
        for timestamp in (self.train_end, self.validation_end):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("temporal cutovers must be timezone-aware")
        if self.train_end >= self.validation_end:
            raise ValueError("train_end must be earlier than validation_end")
        labels = tuple(partition.partition for partition in self.partitions)
        if labels != TEMPORAL_PARTITIONS:
            raise ValueError("temporal partitions must be train, validation, and test in order")
        return self


class PlayerDisjointPartitionSummary(ManifestModel):
    partition: Literal["train", "validation", "test"]
    row_count: int = Field(ge=0)


class PlayerDisjointSplitManifest(ManifestModel):
    hash_algorithm: Literal["sha256"] = "sha256"
    seed: int
    train_max: float = Field(gt=0, lt=1)
    validation_max: float = Field(gt=0, lt=1)
    excluded_bridge_rows: int = Field(ge=0)
    partitions: tuple[PlayerDisjointPartitionSummary, ...]

    @model_validator(mode="after")
    def validate_split(self) -> Self:
        if self.train_max >= self.validation_max:
            raise ValueError("train_max must be less than validation_max")
        labels = tuple(partition.partition for partition in self.partitions)
        if labels != PLAYER_PARTITIONS:
            raise ValueError(
                "player-disjoint partitions must be train, validation, and test in order"
            )
        return self


class ProcessingConfiguration(ManifestModel):
    memory_limit: str = Field(min_length=1)
    threads: int = Field(ge=1)
    chunk_size: int = Field(ge=1)
    temp_policy: Literal["same_filesystem_required"] = "same_filesystem_required"
    parquet_compression: Literal["zstd"] = "zstd"
    parquet_row_group_rows: Literal[131072] = 131072
    final_write_threads: Literal[1] = 1


class ProcessedOutputFile(ManifestModel):
    path: RelativePath
    kind: Literal[
        "canonical",
        "disposition",
        "temporal_split",
        "player_disjoint_split",
        "verification_report",
    ]
    size_bytes: int = Field(ge=0)
    sha256: Sha256
    logical_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_path_and_hash(self) -> Self:
        validate_relative_path(self.path)
        if self.kind == "canonical":
            if self.logical_sha256 is None:
                raise ValueError("canonical output files require logical_sha256")
        elif self.logical_sha256 is not None:
            raise ValueError("only canonical output files may include logical_sha256")
        return self


class ProcessedValidation(ManifestModel):
    status: Literal["passed"] = "passed"
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_warnings(self) -> Self:
        if self.warnings:
            raise ValueError("processed validation warnings are not allowed in v1")
        return self


class VerificationCheckResult(ManifestModel):
    check_id: str = Field(min_length=1)
    passed: Literal[True] = True


class VerificationReport(ManifestModel):
    report_version: Literal[1] = 1
    checks: tuple[VerificationCheckResult, ...]

    @model_validator(mode="after")
    def validate_checks(self) -> Self:
        check_ids = [check.check_id for check in self.checks]
        if check_ids != sorted(set(check_ids)):
            raise ValueError("verification checks must have unique, sorted check IDs")
        return self


def player_hash_fraction(player_id: str, *, seed: int = DEFAULT_PLAYER_HASH_SEED) -> float:
    normalized = PlayerId(player_id).value
    digest = sha256(f"{seed}:{normalized}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def player_partition(
    fraction: float,
    *,
    train_max: float = DEFAULT_PLAYER_TRAIN_MAX,
    validation_max: float = DEFAULT_PLAYER_VALIDATION_MAX,
) -> Literal["train", "validation", "test"]:
    if fraction < train_max:
        return "train"
    if fraction < validation_max:
        return "validation"
    return "test"


class ProcessedDatasetManifest(ManifestModel):
    manifest_type: Literal["processed"] = "processed"
    manifest_version: Literal[1] = 1
    dataset_version: str = Field(min_length=1)
    preparation_policy_version: Literal["kaggle-v6-ranked16-policy:v1"]
    canonical_schema_version: Literal["kaggle-v6-ranked16-schema:v1"]
    catalog_version: str = Field(min_length=1)
    era_registry_version: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    raw_audit_manifest_sha256: Sha256
    archive_sha256: Sha256
    source_schema_fingerprint: Sha256
    source_members: tuple[RelativePath, ...]
    dispositions: DispositionSummary
    accepted: AcceptedSummary
    files: tuple[ProcessedOutputFile, ...]
    processing: ProcessingConfiguration
    temporal_split: TemporalSplitManifest
    player_disjoint_split: PlayerDisjointSplitManifest
    validation: ProcessedValidation

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        members = tuple(self.source_members)
        if not members or members != tuple(sorted(set(members))):
            raise ValueError("source members must have unique, sorted paths")
        for member in members:
            validate_relative_path(member)
        paths = tuple(file.path for file in self.files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ValueError("processed files must have unique, sorted paths")
        for path in paths:
            if path == "manifest.json" or path.endswith("/manifest.json"):
                raise ValueError("processed manifest must not inventory itself")
        valid_count = next(
            count.count for count in self.dispositions.states if count.state is RecordState.VALID
        )
        if self.accepted.row_count != valid_count:
            raise ValueError("accepted row_count must equal the valid disposition count")
        temporal_rows = sum(partition.row_count for partition in self.temporal_split.partitions)
        if temporal_rows != self.accepted.row_count:
            raise ValueError("temporal partition row counts must sum to accepted row_count")
        player_rows = sum(
            partition.row_count for partition in self.player_disjoint_split.partitions
        )
        if player_rows + self.player_disjoint_split.excluded_bridge_rows != self.accepted.row_count:
            raise ValueError(
                "player-disjoint partitions plus bridge exclusions must equal accepted row_count"
            )
        kind_counts = {
            "canonical": 0,
            "disposition": 0,
            "temporal_split": 0,
            "player_disjoint_split": 0,
            "verification_report": 0,
        }
        for file in self.files:
            kind_counts[file.kind] += 1
        if kind_counts["canonical"] != 1:
            raise ValueError("processed manifest must inventory exactly one canonical file")
        if kind_counts["temporal_split"] != 1:
            raise ValueError("processed manifest must inventory exactly one temporal split file")
        if kind_counts["player_disjoint_split"] != 1:
            raise ValueError(
                "processed manifest must inventory exactly one player-disjoint split file"
            )
        if kind_counts["verification_report"] != 1:
            raise ValueError("processed manifest must inventory exactly one verification report")
        if kind_counts["disposition"] < 1:
            raise ValueError("processed manifest must inventory at least one disposition file")
        if self.canonical_schema_version != CANONICAL_SCHEMA_VERSION:
            raise ValueError("canonical_schema_version must match the frozen schema literal")
        return self


def dump_processed_manifest(manifest: ProcessedDatasetManifest) -> bytes:
    return canonical_json_bytes(manifest.model_dump(mode="python")) + b"\n"


def dump_verification_report(report: VerificationReport) -> bytes:
    """Serialize a passed-only verification report as canonical JSON plus a newline."""
    return canonical_json_bytes(report.model_dump(mode="python")) + b"\n"
