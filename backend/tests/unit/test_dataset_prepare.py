from datetime import UTC, datetime
from json import loads
from pathlib import Path

import duckdb
import pytest

from clash_sos.application.dataset_prepare import (
    KaggleV6PrepareError,
    summarize_accepted,
    write_verification_report,
)
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.dataset_splits import VERIFICATION_CHECK_IDS
from clash_sos.domain.processed_manifest import (
    VerificationCheckResult,
    VerificationReport,
    dump_verification_report,
)

CONFIG = StagingConfig(threads=1, memory_limit="256MB")
TRAIN_END = datetime(2026, 6, 10, tzinfo=UTC)
VALIDATION_END = datetime(2026, 6, 20, tzinfo=UTC)
FINGERPRINT = "a" * 64


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
