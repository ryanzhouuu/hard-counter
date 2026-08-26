from datetime import UTC, datetime
from hashlib import sha256

import pytest
from pydantic import ValidationError

from clash_sos.domain.canonical import RecordIssue, RecordState
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.processed_manifest import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_PLAYER_TRAIN_MAX,
    DEFAULT_PLAYER_VALIDATION_MAX,
    FINALIZATION_ORDER,
    INGESTION_ISSUES,
    PREPARATION_POLICY_VERSION,
    PROCESSED_MANIFEST_TYPE,
    RECORD_STATES,
    AcceptedSummary,
    DispositionSummary,
    EraCount,
    IssueCount,
    PlayerDisjointPartitionSummary,
    PlayerDisjointSplitManifest,
    ProcessedDatasetManifest,
    ProcessedOutputFile,
    ProcessedValidation,
    ProcessingConfiguration,
    StateCount,
    TemporalPartitionSummary,
    TemporalSplitManifest,
    VerificationCheckResult,
    VerificationReport,
    dump_processed_manifest,
    player_hash_fraction,
    player_partition,
)
from clash_sos.infrastructure.kaggle_v6.balance_eras import KAGGLE_V6_ERA_REGISTRY_VERSION
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.source import (
    KAGGLE_V6_ARCHIVE_SHA256,
    KAGGLE_V6_SOURCE_ID,
)

SHA256 = "a" * 64
LOGICAL_SHA256 = "b" * 64
TRAIN_END = datetime(2026, 6, 10, tzinfo=UTC)
VALIDATION_END = datetime(2026, 6, 20, tzinfo=UTC)
TIMESTAMP_MIN = datetime(2026, 6, 2, tzinfo=UTC)
TIMESTAMP_MAX = datetime(2026, 6, 25, tzinfo=UTC)


def zero_issue_counts() -> tuple[IssueCount, ...]:
    return tuple(IssueCount(issue=issue, count=0) for issue in INGESTION_ISSUES)


def disposition_summary(**overrides: object) -> DispositionSummary:
    payload: dict[str, object] = {
        "source_row_count": 100,
        "states": (
            StateCount(state=RecordState.INSUFFICIENT_DATA, count=0),
            StateCount(state=RecordState.INVALID, count=10),
            StateCount(state=RecordState.QUARANTINED, count=20),
            StateCount(state=RecordState.UNSUPPORTED, count=65),
            StateCount(state=RecordState.VALID, count=5),
        ),
        "issues": zero_issue_counts(),
        "duplicate_row_count": 0,
        "conflict_row_count": 0,
    }
    payload.update(overrides)
    return DispositionSummary.model_validate(payload)


def accepted_summary(**overrides: object) -> AcceptedSummary:
    payload: dict[str, object] = {
        "row_count": 5,
        "timestamp_min": TIMESTAMP_MIN,
        "timestamp_max": TIMESTAMP_MAX,
        "eras": (EraCount(era_id="2026-06", count=5),),
    }
    payload.update(overrides)
    return AcceptedSummary.model_validate(payload)


def temporal_split(**overrides: object) -> TemporalSplitManifest:
    payload: dict[str, object] = {
        "train_end": TRAIN_END,
        "validation_end": VALIDATION_END,
        "partitions": (
            TemporalPartitionSummary(
                partition="train",
                row_count=2,
                timestamp_min=TIMESTAMP_MIN,
                timestamp_max=datetime(2026, 6, 9, tzinfo=UTC),
            ),
            TemporalPartitionSummary(
                partition="validation",
                row_count=2,
                timestamp_min=datetime(2026, 6, 10, tzinfo=UTC),
                timestamp_max=datetime(2026, 6, 19, tzinfo=UTC),
            ),
            TemporalPartitionSummary(
                partition="test",
                row_count=1,
                timestamp_min=datetime(2026, 6, 20, tzinfo=UTC),
                timestamp_max=TIMESTAMP_MAX,
            ),
        ),
    }
    payload.update(overrides)
    return TemporalSplitManifest.model_validate(payload)


def player_disjoint_split(**overrides: object) -> PlayerDisjointSplitManifest:
    payload: dict[str, object] = {
        "seed": 0,
        "train_max": DEFAULT_PLAYER_TRAIN_MAX,
        "validation_max": DEFAULT_PLAYER_VALIDATION_MAX,
        "excluded_bridge_rows": 1,
        "partitions": (
            PlayerDisjointPartitionSummary(partition="train", row_count=2),
            PlayerDisjointPartitionSummary(partition="validation", row_count=1),
            PlayerDisjointPartitionSummary(partition="test", row_count=1),
        ),
    }
    payload.update(overrides)
    return PlayerDisjointSplitManifest.model_validate(payload)


def processed_files() -> tuple[ProcessedOutputFile, ...]:
    return (
        ProcessedOutputFile(
            path="canonical.parquet",
            kind="canonical",
            size_bytes=100,
            sha256=SHA256,
            logical_sha256=LOGICAL_SHA256,
        ),
        ProcessedOutputFile(
            path="dispositions/member-a.parquet",
            kind="disposition",
            size_bytes=50,
            sha256=SHA256,
        ),
        ProcessedOutputFile(
            path="splits-player-disjoint.parquet",
            kind="player_disjoint_split",
            size_bytes=20,
            sha256=SHA256,
        ),
        ProcessedOutputFile(
            path="splits-temporal.parquet",
            kind="temporal_split",
            size_bytes=20,
            sha256=SHA256,
        ),
        ProcessedOutputFile(
            path="verification-report.json",
            kind="verification_report",
            size_bytes=10,
            sha256=SHA256,
        ),
    )


def processed_manifest() -> ProcessedDatasetManifest:
    return ProcessedDatasetManifest(
        dataset_version=DEFAULT_DATASET_VERSION,
        preparation_policy_version=PREPARATION_POLICY_VERSION,
        canonical_schema_version="kaggle-v6-ranked16-schema:v1",
        catalog_version=KAGGLE_V6_CARDS.version,
        era_registry_version=KAGGLE_V6_ERA_REGISTRY_VERSION,
        source_id=KAGGLE_V6_SOURCE_ID,
        raw_audit_manifest_sha256=SHA256,
        archive_sha256=KAGGLE_V6_ARCHIVE_SHA256,
        source_schema_fingerprint=SHA256,
        source_members=("a.parquet", "b.parquet"),
        dispositions=disposition_summary(),
        accepted=accepted_summary(),
        files=processed_files(),
        processing=ProcessingConfiguration(
            memory_limit="1GB",
            threads=2,
            chunk_size=8_388_608,
        ),
        temporal_split=temporal_split(),
        player_disjoint_split=player_disjoint_split(),
        validation=ProcessedValidation(),
    )


def test_version_literals() -> None:
    assert PROCESSED_MANIFEST_TYPE == "processed"
    assert DEFAULT_DATASET_VERSION == "kaggle-v6-ranked16-v1"
    assert PREPARATION_POLICY_VERSION == "kaggle-v6-ranked16-policy:v1"
    assert FINALIZATION_ORDER == ("data_artifacts", "verification_report", "manifest")


def test_disposition_summary_requires_full_sorted_inventories() -> None:
    summary = disposition_summary()
    assert tuple(count.state for count in summary.states) == RECORD_STATES
    assert tuple(count.issue for count in summary.issues) == INGESTION_ISSUES

    with pytest.raises(ValidationError, match="record state"):
        disposition_summary(
            states=(
                StateCount(state=RecordState.VALID, count=100),
                StateCount(state=RecordState.INVALID, count=0),
                StateCount(state=RecordState.QUARANTINED, count=0),
                StateCount(state=RecordState.UNSUPPORTED, count=0),
                StateCount(state=RecordState.INSUFFICIENT_DATA, count=0),
            )
        )


def test_disposition_summary_reconciles_duplicate_and_conflict_counts() -> None:
    issues = list(zero_issue_counts())
    issues[INGESTION_ISSUES.index(RecordIssue.DUPLICATE_BATTLE)] = IssueCount(
        issue=RecordIssue.DUPLICATE_BATTLE, count=3
    )
    issues[INGESTION_ISSUES.index(RecordIssue.CONFLICTING_BATTLE)] = IssueCount(
        issue=RecordIssue.CONFLICTING_BATTLE, count=2
    )
    with pytest.raises(ValidationError, match="duplicate_row_count"):
        disposition_summary(issues=tuple(issues), duplicate_row_count=0, conflict_row_count=2)
    summary = disposition_summary(issues=tuple(issues), duplicate_row_count=3, conflict_row_count=2)
    assert summary.duplicate_row_count == 3
    assert summary.conflict_row_count == 2


def test_accepted_summary_requires_timestamp_bounds_when_non_empty() -> None:
    with pytest.raises(ValidationError, match="timestamp bounds"):
        accepted_summary(timestamp_min=None)
    with pytest.raises(ValidationError, match="timestamp_min"):
        accepted_summary(
            timestamp_min=TIMESTAMP_MAX,
            timestamp_max=TIMESTAMP_MIN,
        )


def test_temporal_split_requires_half_open_cutovers() -> None:
    split = temporal_split()
    assert split.semantics == "half_open"
    assert tuple(partition.partition for partition in split.partitions) == (
        "train",
        "validation",
        "test",
    )
    with pytest.raises(ValidationError, match="train_end"):
        temporal_split(train_end=VALIDATION_END, validation_end=TRAIN_END)


def test_temporal_partition_bounds_follow_row_count() -> None:
    empty = TemporalPartitionSummary(partition="train", row_count=0)
    assert empty.timestamp_min is None
    assert empty.timestamp_max is None
    with pytest.raises(ValidationError, match="timestamp bounds"):
        TemporalPartitionSummary(
            partition="train",
            row_count=0,
            timestamp_min=TIMESTAMP_MIN,
            timestamp_max=TIMESTAMP_MIN,
        )
    with pytest.raises(ValidationError, match="timestamp bounds"):
        TemporalPartitionSummary(partition="train", row_count=1)


def test_processed_manifest_allows_zero_accepted_rows() -> None:
    payload = processed_manifest().model_dump()
    payload["dispositions"] = disposition_summary(
        source_row_count=100,
        states=(
            StateCount(state=RecordState.INSUFFICIENT_DATA, count=0),
            StateCount(state=RecordState.INVALID, count=10),
            StateCount(state=RecordState.QUARANTINED, count=20),
            StateCount(state=RecordState.UNSUPPORTED, count=70),
            StateCount(state=RecordState.VALID, count=0),
        ),
    ).model_dump()
    payload["accepted"] = AcceptedSummary(row_count=0, eras=()).model_dump()
    payload["temporal_split"] = temporal_split(
        partitions=(
            TemporalPartitionSummary(partition="train", row_count=0),
            TemporalPartitionSummary(partition="validation", row_count=0),
            TemporalPartitionSummary(partition="test", row_count=0),
        )
    ).model_dump()
    payload["player_disjoint_split"] = player_disjoint_split(
        excluded_bridge_rows=0,
        partitions=(
            PlayerDisjointPartitionSummary(partition="train", row_count=0),
            PlayerDisjointPartitionSummary(partition="validation", row_count=0),
            PlayerDisjointPartitionSummary(partition="test", row_count=0),
        ),
    ).model_dump()
    manifest = ProcessedDatasetManifest.model_validate(payload)
    assert manifest.accepted.row_count == 0
    assert all(partition.row_count == 0 for partition in manifest.temporal_split.partitions)


def test_processed_manifest_allows_single_accepted_row() -> None:
    payload = processed_manifest().model_dump()
    payload["dispositions"] = disposition_summary(
        source_row_count=100,
        states=(
            StateCount(state=RecordState.INSUFFICIENT_DATA, count=0),
            StateCount(state=RecordState.INVALID, count=10),
            StateCount(state=RecordState.QUARANTINED, count=20),
            StateCount(state=RecordState.UNSUPPORTED, count=69),
            StateCount(state=RecordState.VALID, count=1),
        ),
    ).model_dump()
    payload["accepted"] = accepted_summary(
        row_count=1,
        timestamp_min=TIMESTAMP_MIN,
        timestamp_max=TIMESTAMP_MIN,
        eras=(EraCount(era_id="2026-06", count=1),),
    ).model_dump()
    payload["temporal_split"] = temporal_split(
        partitions=(
            TemporalPartitionSummary(
                partition="train",
                row_count=1,
                timestamp_min=TIMESTAMP_MIN,
                timestamp_max=TIMESTAMP_MIN,
            ),
            TemporalPartitionSummary(partition="validation", row_count=0),
            TemporalPartitionSummary(partition="test", row_count=0),
        )
    ).model_dump()
    payload["player_disjoint_split"] = player_disjoint_split(
        excluded_bridge_rows=0,
        partitions=(
            PlayerDisjointPartitionSummary(partition="train", row_count=1),
            PlayerDisjointPartitionSummary(partition="validation", row_count=0),
            PlayerDisjointPartitionSummary(partition="test", row_count=0),
        ),
    ).model_dump()
    manifest = ProcessedDatasetManifest.model_validate(payload)
    assert manifest.accepted.row_count == 1
    assert manifest.temporal_split.partitions[0].row_count == 1
    assert manifest.temporal_split.partitions[1].row_count == 0


@pytest.mark.parametrize(
    "path",
    ["/absolute.parquet", "../escape.parquet", "nested/../file.parquet"],
)
def test_source_members_reject_non_canonical_paths(path: str) -> None:
    payload = processed_manifest().model_dump()
    payload["source_members"] = [path]
    with pytest.raises(ValidationError, match="file path must"):
        ProcessedDatasetManifest.model_validate(payload)


def test_player_disjoint_thresholds_and_partition_assignment() -> None:
    expected = int.from_bytes(sha256(b"0:#PLAYER").digest()[:8], "big") / 2**64
    assert player_hash_fraction("#PLAYER", seed=0) == expected
    assert player_partition(0.0) == "train"
    assert player_partition(DEFAULT_PLAYER_TRAIN_MAX - 1e-12) == "train"
    assert player_partition(DEFAULT_PLAYER_TRAIN_MAX) == "validation"
    assert player_partition(DEFAULT_PLAYER_VALIDATION_MAX - 1e-12) == "validation"
    assert player_partition(DEFAULT_PLAYER_VALIDATION_MAX) == "test"
    with pytest.raises(ValidationError, match="train_max"):
        PlayerDisjointSplitManifest.model_validate(
            {
                "seed": 0,
                "train_max": 0.85,
                "validation_max": 0.70,
                "excluded_bridge_rows": 0,
                "partitions": [
                    {"partition": "train", "row_count": 0},
                    {"partition": "validation", "row_count": 0},
                    {"partition": "test", "row_count": 0},
                ],
            }
        )


@pytest.mark.parametrize(
    "path",
    ["/absolute.parquet", "../escape.parquet", "nested/../file.parquet"],
)
def test_processed_output_file_rejects_non_canonical_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="file path must"):
        ProcessedOutputFile(
            path=path,
            kind="disposition",
            size_bytes=1,
            sha256=SHA256,
        )


def test_processed_output_file_requires_logical_hash_only_for_canonical() -> None:
    with pytest.raises(ValidationError, match="logical_sha256"):
        ProcessedOutputFile(
            path="canonical.parquet",
            kind="canonical",
            size_bytes=1,
            sha256=SHA256,
        )
    with pytest.raises(ValidationError, match="only canonical"):
        ProcessedOutputFile(
            path="splits-temporal.parquet",
            kind="temporal_split",
            size_bytes=1,
            sha256=SHA256,
            logical_sha256=LOGICAL_SHA256,
        )


def test_processed_validation_rejects_warnings() -> None:
    with pytest.raises(ValidationError, match="warnings"):
        ProcessedValidation(warnings=("unexpected",))


def test_verification_report_has_no_manifest_hash_field() -> None:
    assert not any("manifest" in name for name in VerificationReport.model_fields)
    report = VerificationReport(
        checks=(
            VerificationCheckResult(check_id="canonical_identity"),
            VerificationCheckResult(check_id="disposition_reconciliation"),
        )
    )
    assert report.checks[0].check_id == "canonical_identity"


def test_processed_manifest_round_trips_through_json() -> None:
    manifest = processed_manifest()
    serialized = manifest.model_dump_json()
    restored = ProcessedDatasetManifest.model_validate_json(serialized)
    assert restored == manifest


def test_dump_processed_manifest_is_byte_stable() -> None:
    manifest = processed_manifest()
    first = dump_processed_manifest(manifest)
    second = dump_processed_manifest(manifest)
    assert first == second
    assert first.endswith(b"\n")
    assert first == canonical_json_bytes(manifest.model_dump(mode="python")) + b"\n"


def test_manifest_does_not_inventory_itself() -> None:
    files = list(processed_files())
    files[2] = files[2].model_copy(update={"path": "manifest.json"})
    payload = processed_manifest().model_dump()
    payload["files"] = [file.model_dump(mode="python") for file in files]
    with pytest.raises(ValidationError, match="itself"):
        ProcessedDatasetManifest.model_validate(payload)


def test_manifest_rejects_mismatched_accepted_and_valid_counts() -> None:
    payload = processed_manifest().model_dump()
    payload["accepted"] = accepted_summary(
        row_count=4,
        eras=(EraCount(era_id="2026-06", count=4),),
    ).model_dump()
    with pytest.raises(ValidationError, match="valid disposition"):
        ProcessedDatasetManifest.model_validate(payload)


def test_manifest_rejects_duplicate_file_paths() -> None:
    payload = processed_manifest().model_dump()
    files = list(payload["files"])
    files.append(files[0])
    payload["files"] = files
    with pytest.raises(ValidationError, match="unique, sorted"):
        ProcessedDatasetManifest.model_validate(payload)


def test_manifest_rejects_invalid_sha256() -> None:
    with pytest.raises(ValidationError, match="pattern"):
        ProcessedDatasetManifest.model_validate(
            {
                **processed_manifest().model_dump(),
                "raw_audit_manifest_sha256": "not-a-checksum",
            }
        )


def test_manifest_rejects_mismatched_split_row_counts() -> None:
    payload = processed_manifest().model_dump()
    payload["temporal_split"] = temporal_split(
        partitions=(
            TemporalPartitionSummary(
                partition="train",
                row_count=1,
                timestamp_min=TIMESTAMP_MIN,
                timestamp_max=datetime(2026, 6, 9, tzinfo=UTC),
            ),
            TemporalPartitionSummary(
                partition="validation",
                row_count=1,
                timestamp_min=datetime(2026, 6, 10, tzinfo=UTC),
                timestamp_max=datetime(2026, 6, 19, tzinfo=UTC),
            ),
            TemporalPartitionSummary(
                partition="test",
                row_count=1,
                timestamp_min=datetime(2026, 6, 20, tzinfo=UTC),
                timestamp_max=TIMESTAMP_MAX,
            ),
        )
    ).model_dump()
    with pytest.raises(ValidationError, match="temporal partition"):
        ProcessedDatasetManifest.model_validate(payload)


def test_manifest_rejects_run_varying_extra_fields() -> None:
    payload = processed_manifest().model_dump()
    payload["prepared_at"] = TIMESTAMP_MIN.isoformat()
    with pytest.raises(ValidationError, match="prepared_at"):
        ProcessedDatasetManifest.model_validate(payload)
