"""Build accepted-row summaries and serialize processed verification reports.

Accepted summaries are DuckDB aggregates over `canonical.parquet`. Frozen
checks raise unless every artifact passes; the report is written only after.
"""

from datetime import datetime
from pathlib import Path

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.dataset_splits import VERIFICATION_CHECK_IDS
from clash_sos.domain.processed_manifest import (
    AcceptedSummary,
    EraCount,
    VerificationCheckResult,
    VerificationReport,
    dump_verification_report,
)
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb, python_cell
from clash_sos.infrastructure.kaggle_v6.verify_io import (
    KaggleV6VerifyError,
    verify_processed_artifacts,
)


class KaggleV6PrepareError(ValueError):
    pass


def _as_int(value: object) -> int:
    cell = python_cell(value)
    if type(cell) is not int:
        raise KaggleV6PrepareError("count must be an integer")
    return cell


def _as_datetime(value: object) -> datetime:
    cell = python_cell(value)
    if not isinstance(cell, datetime):
        raise KaggleV6PrepareError("timestamp bounds must be datetimes")
    if cell.tzinfo is None or cell.utcoffset() is None:
        raise KaggleV6PrepareError("accepted timestamps must be timezone-aware")
    return cell


def summarize_accepted(
    canonical_path: Path,
    *,
    config: StagingConfig,
    temp_directory: Path,
) -> AcceptedSummary:
    """Return era counts and timestamp bounds from canonical Parquet aggregates."""
    if not canonical_path.is_file():
        raise KaggleV6PrepareError("canonical parquet is required")
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = connect_staging_duckdb(
        memory_limit=config.memory_limit,
        threads=1,
        temp_directory=temp_directory,
    )
    try:
        bounds = connection.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM read_parquet(?)",
            [str(canonical_path)],
        ).fetchone()
        if bounds is None:
            raise KaggleV6PrepareError("accepted summary is unavailable")
        row_count = _as_int(bounds[0])
        eras = tuple(
            EraCount(era_id=str(era_id), count=_as_int(count))
            for era_id, count in connection.execute(
                """
                SELECT balance_era_id, COUNT(*)
                FROM read_parquet(?)
                GROUP BY 1
                ORDER BY 1
                """,
                [str(canonical_path)],
            ).fetchall()
        )
    finally:
        connection.close()
    if row_count == 0:
        return AcceptedSummary(row_count=0, timestamp_min=None, timestamp_max=None, eras=())
    return AcceptedSummary(
        row_count=row_count,
        timestamp_min=_as_datetime(bounds[1]),
        timestamp_max=_as_datetime(bounds[2]),
        eras=eras,
    )


def write_verification_report(
    output_path: Path,
    *,
    canonical_path: Path,
    disposition_files: tuple[Path, ...],
    temporal_split_path: Path,
    player_split_path: Path,
    train_end: datetime,
    validation_end: datetime,
    excluded_bridge_rows: int,
    config: StagingConfig,
    temp_directory: Path,
) -> VerificationReport:
    """Write verification-report.json after every frozen check passes; otherwise raise."""
    if output_path.exists():
        raise KaggleV6PrepareError("verification report already exists")
    if not canonical_path.is_file():
        raise KaggleV6PrepareError("canonical parquet is required")
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = connect_staging_duckdb(
        memory_limit=config.memory_limit,
        threads=1,
        temp_directory=temp_directory,
    )
    try:
        verify_processed_artifacts(
            connection,
            canonical_path=canonical_path,
            disposition_files=disposition_files,
            temporal_split_path=temporal_split_path,
            player_split_path=player_split_path,
            train_end=train_end,
            validation_end=validation_end,
            excluded_bridge_rows=excluded_bridge_rows,
        )
    except KaggleV6VerifyError as error:
        raise KaggleV6PrepareError(str(error)) from error
    finally:
        connection.close()
    report = VerificationReport(
        checks=tuple(
            VerificationCheckResult(check_id=check_id) for check_id in VERIFICATION_CHECK_IDS
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(dump_verification_report(report))
    return report
