from datetime import UTC, datetime
from json import loads
from os import stat_result
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import duckdb
import pytest

from clash_sos.application.dataset_prepare import (
    KaggleV6PrepareError,
    assemble_processed_manifest,
    execute_with_prepare_cleanup,
    summarize_accepted,
    write_processed_manifest,
    write_verification_report,
)
from clash_sos.application.dataset_splits import write_player_disjoint_split, write_temporal_split
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.canonical import RecordState
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.dataset_splits import VERIFICATION_CHECK_IDS
from clash_sos.domain.processed_manifest import (
    INGESTION_ISSUES,
    RECORD_STATES,
    DispositionSummary,
    IssueCount,
    PlayerDisjointSplitManifest,
    StateCount,
    TemporalSplitManifest,
    VerificationCheckResult,
    VerificationReport,
    dump_processed_manifest,
    dump_verification_report,
)
from clash_sos.infrastructure.kaggle_v6.publish_io import (
    KaggleV6PublishError,
    publish_processed_version,
)
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_ARCHIVE_SHA256

CONFIG = StagingConfig(threads=1, memory_limit="256MB")
TRAIN_END = datetime(2026, 6, 10, tzinfo=UTC)
VALIDATION_END = datetime(2026, 6, 20, tzinfo=UTC)
FINGERPRINT = "a" * 64
LOGICAL_SHA256 = "b" * 64
SCHEMA_FINGERPRINT = "c" * 64
TRAIN_A = "#P0001"
TRAIN_B = "#P0002"
VALIDATION_A = "#P0011"


def _report() -> VerificationReport:
    return VerificationReport(
        checks=tuple(
            VerificationCheckResult(check_id=check_id) for check_id in VERIFICATION_CHECK_IDS
        )
    )


def test_dump_verification_report_is_byte_stable() -> None:
    report = _report()
    first = dump_verification_report(report)
    second = dump_verification_report(report)
    assert first == second
    assert first.endswith(b"\n")
    assert first == canonical_json_bytes(report.model_dump(mode="python")) + b"\n"
    payload = loads(first)
    assert [check["check_id"] for check in payload["checks"]] == list(VERIFICATION_CHECK_IDS)
    assert not any("manifest" in key for key in payload)


def test_write_verification_report_rejects_duplicate_identity(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE identities (
                timestamp TIMESTAMPTZ,
                fingerprint VARCHAR,
                archive_member VARCHAR,
                row_number BIGINT
            )
            """
        )
        connection.executemany(
            "INSERT INTO identities VALUES (?, ?, ?, ?)",
            [
                (datetime(2026, 6, 9, tzinfo=UTC), FINGERPRINT, "a.parquet", 0),
                (datetime(2026, 6, 9, tzinfo=UTC), FINGERPRINT, "a.parquet", 0),
            ],
        )
        connection.execute("COPY identities TO ? (FORMAT PARQUET)", [str(canonical)])
    finally:
        connection.close()
    output = tmp_path / "verification-report.json"
    with pytest.raises(KaggleV6PrepareError, match="identity"):
        write_verification_report(
            output,
            canonical_path=canonical,
            disposition_files=(),
            temporal_split_path=tmp_path / "missing-temporal.parquet",
            player_split_path=tmp_path / "missing-player.parquet",
            train_end=TRAIN_END,
            validation_end=VALIDATION_END,
            excluded_bridge_rows=0,
            config=CONFIG,
            temp_directory=tmp_path / "tmp",
        )
    assert not output.exists()


def test_summarize_accepted_matches_duckdb_aggregates(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    early = datetime(2026, 6, 9, tzinfo=UTC)
    late = datetime(2026, 6, 21, tzinfo=UTC)
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE accepted (
                timestamp TIMESTAMPTZ,
                balance_era_id VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO accepted VALUES (?, ?)",
            [(early, "2026-06"), (late, "2026-07"), (late, "2026-06")],
        )
        connection.execute("COPY accepted TO ? (FORMAT PARQUET)", [str(canonical)])
        expected = connection.execute(
            """
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM read_parquet(?)
            """,
            [str(canonical)],
        ).fetchone()
        eras = connection.execute(
            """
            SELECT balance_era_id, COUNT(*)
            FROM read_parquet(?)
            GROUP BY 1
            ORDER BY 1
            """,
            [str(canonical)],
        ).fetchall()
    finally:
        connection.close()
    summary = summarize_accepted(
        canonical,
        config=CONFIG,
        temp_directory=tmp_path / "tmp",
    )
    assert expected is not None
    assert summary.row_count == expected[0]
    assert summary.timestamp_min == expected[1]
    assert summary.timestamp_max == expected[2]
    assert [(era.era_id, era.count) for era in summary.eras] == [
        (str(era_id), int(count)) for era_id, count in eras
    ]


def _disposition_summary(valid: int) -> DispositionSummary:
    return DispositionSummary(
        source_row_count=valid,
        states=tuple(
            StateCount(state=state, count=valid if state is RecordState.VALID else 0)
            for state in RECORD_STATES
        ),
        issues=tuple(IssueCount(issue=issue, count=0) for issue in INGESTION_ISSUES),
        duplicate_row_count=0,
        conflict_row_count=0,
    )


def _write_workspace(
    tmp_path: Path,
) -> tuple[Path, TemporalSplitManifest, PlayerDisjointSplitManifest]:
    output = tmp_path / "output"
    output.mkdir()
    canonical = output / "canonical.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE battles (
                timestamp TIMESTAMPTZ,
                fingerprint VARCHAR,
                archive_member VARCHAR,
                row_number BIGINT,
                side_a_player_id VARCHAR,
                side_b_player_id VARCHAR,
                balance_era_id VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO battles VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    datetime(2026, 6, 9, tzinfo=UTC),
                    FINGERPRINT,
                    "a.parquet",
                    0,
                    TRAIN_A,
                    TRAIN_B,
                    "2026-06",
                ),
                (TRAIN_END, "b" * 64, "a.parquet", 1, TRAIN_B, TRAIN_A, "2026-06"),
                (VALIDATION_END, "c" * 64, "b.parquet", 2, TRAIN_A, VALIDATION_A, "2026-06"),
            ],
        )
        connection.execute("COPY battles TO ? (FORMAT PARQUET)", [str(canonical)])
        connection.execute(
            """
            CREATE TABLE dispositions (
                archive_member VARCHAR,
                row_number BIGINT,
                state VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO dispositions VALUES (?, ?, ?)",
            [("a.parquet", 0, "valid"), ("a.parquet", 1, "valid"), ("b.parquet", 2, "valid")],
        )
        (output / "dispositions").mkdir()
        connection.execute(
            "COPY dispositions TO ? (FORMAT PARQUET)",
            [str(output / "dispositions" / "a.parquet")],
        )
    finally:
        connection.close()
    temporal_manifest = write_temporal_split(
        canonical,
        output / "splits-temporal.parquet",
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
        config=CONFIG,
        temp_directory=tmp_path / "tmp-temporal",
    )
    player_manifest = write_player_disjoint_split(
        canonical,
        output / "splits-player-disjoint.parquet",
        config=CONFIG,
        temp_directory=tmp_path / "tmp-player",
    )
    (output / "verification-report.json").write_bytes(dump_verification_report(_report()))
    return output, temporal_manifest, player_manifest


def test_assemble_processed_manifest_is_byte_stable_and_sorted(tmp_path: Path) -> None:
    output, temporal_manifest, player_manifest = _write_workspace(tmp_path)
    raw_manifest = tmp_path / "raw.json"
    raw_manifest.write_bytes(b"{}\n")
    accepted = summarize_accepted(
        output / "canonical.parquet",
        config=CONFIG,
        temp_directory=tmp_path / "tmp-accepted",
    )
    manifest = assemble_processed_manifest(
        output,
        config=CONFIG,
        temporal_split=temporal_manifest,
        player_disjoint_split=player_manifest,
        dispositions=_disposition_summary(3),
        accepted=accepted,
        canonical_logical_sha256=LOGICAL_SHA256,
        raw_manifest_path=raw_manifest,
        archive_sha256=KAGGLE_V6_ARCHIVE_SHA256,
        source_schema_fingerprint=SCHEMA_FINGERPRINT,
        source_members=("a.parquet", "b.parquet"),
    )
    first = dump_processed_manifest(manifest)
    assert first == dump_processed_manifest(manifest)
    assert [file.path for file in manifest.files] == [
        "canonical.parquet",
        "dispositions/a.parquet",
        "splits-player-disjoint.parquet",
        "splits-temporal.parquet",
        "verification-report.json",
    ]
    kinds = {file.path: file for file in manifest.files}
    assert kinds["canonical.parquet"].logical_sha256 == LOGICAL_SHA256
    assert all(
        file.logical_sha256 is None for path, file in kinds.items() if path != "canonical.parquet"
    )
    written = write_processed_manifest(output, manifest)
    assert written.read_bytes() == first
    connection = duckdb.connect()
    try:
        temporal_rows = connection.execute(
            "SELECT partition, COUNT(*) FROM read_parquet(?) GROUP BY 1",
            [str(output / "splits-temporal.parquet")],
        ).fetchall()
        player_row = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(output / "splits-player-disjoint.parquet")],
        ).fetchone()
    finally:
        connection.close()
    temporal_counts = {str(partition): int(count) for partition, count in temporal_rows}
    assert {
        partition.partition: partition.row_count for partition in manifest.temporal_split.partitions
    } == {
        "train": temporal_counts["train"],
        "validation": temporal_counts["validation"],
        "test": temporal_counts["test"],
    }
    retained = 0 if player_row is None else int(player_row[0])
    assert retained + manifest.player_disjoint_split.excluded_bridge_rows == accepted.row_count


def test_publish_processed_version_renames_and_refuses_overwrite(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    destination = tmp_path / "destination"
    workspace.mkdir()
    (workspace / "canonical.parquet").write_text("payload", encoding="utf-8")
    published = publish_processed_version(workspace, destination)
    assert published == destination
    assert not workspace.exists()
    assert (destination / "canonical.parquet").read_text(encoding="utf-8") == "payload"
    leftover = tmp_path / "leftover"
    leftover.mkdir()
    with pytest.raises(KaggleV6PublishError, match="already exists"):
        publish_processed_version(leftover, destination)
    assert leftover.exists()


def test_publish_processed_version_requires_same_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "left" / "workspace"
    destination = tmp_path / "right" / "destination"
    workspace.parent.mkdir()
    destination.parent.mkdir()
    workspace.mkdir()
    original = Path.stat

    def fake_stat(self: Path, *args: object, **kwargs: object) -> stat_result:
        result = original(self, *args, **kwargs)
        if self == destination.parent:
            return cast(stat_result, SimpleNamespace(st_dev=result.st_dev + 1))
        return result

    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(KaggleV6PublishError, match="filesystem"):
        publish_processed_version(workspace, destination)
    assert workspace.exists()
    assert not destination.exists()


def test_forced_failure_after_splits_removes_workspaces(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    output_workspace = tmp_path / "output"
    staging_workspace = tmp_path / "staging"

    def body() -> None:
        output_workspace.mkdir()
        staging_workspace.mkdir()
        (output_workspace / "splits-temporal.parquet").write_text("split", encoding="utf-8")
        raise RuntimeError("forced")

    with pytest.raises(RuntimeError, match="forced"):
        execute_with_prepare_cleanup(
            destination=destination,
            output_workspace=output_workspace,
            staging_workspace=staging_workspace,
            body=body,
        )
    assert not output_workspace.exists()
    assert not staging_workspace.exists()
    assert not destination.exists()
