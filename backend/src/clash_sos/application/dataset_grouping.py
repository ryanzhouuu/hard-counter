"""Final disposition assignment for staged workspaces after identity grouping.

Adapter and staging records are observations. Python helpers encode duplicate,
conflict, and population precedence for one event group. `group_staged_dataset`
applies the same rules with DuckDB over staged Parquet and never loads the
full corpus into Python objects.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.canonical import RecordIssue, RecordState
from clash_sos.domain.disposition_ledger import DispositionRow, IngestionState
from clash_sos.domain.processed_manifest import DispositionSummary
from clash_sos.domain.staged_dataset import StagedBattleRow, UnadaptableRow
from clash_sos.infrastructure.kaggle_v6.grouping_io import (
    KaggleV6GroupingError as KaggleV6GroupingError,
)
from clash_sos.infrastructure.kaggle_v6.grouping_io import (
    create_ledger,
    list_parquet_files,
    summary_from_ledger,
    write_disposition_files,
)
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb


def _sorted_issues(*groups: Sequence[RecordIssue]) -> tuple[RecordIssue, ...]:
    unique = {issue for group in groups for issue in group}
    return tuple(sorted(unique, key=lambda issue: issue.value))


def _adaptable_disposition(
    row: StagedBattleRow,
    *,
    state: IngestionState,
    issues: tuple[RecordIssue, ...],
    representative_archive_member: str | None,
    representative_row_number: int | None,
) -> DispositionRow:
    return DispositionRow(
        archive_member=row.archive_member,
        row_number=row.row_number,
        state=state,
        issues=issues,
        source_id=row.source_id,
        timestamp=row.timestamp,
        mode=row.mode,
        balance_era_id=row.balance_era_id,
        outcome=row.outcome,
        event_key=row.event_key,
        fingerprint=row.fingerprint,
        side_a_player_id=row.side_a_player_id,
        side_b_player_id=row.side_b_player_id,
        representative_archive_member=representative_archive_member,
        representative_row_number=representative_row_number,
    )


def disposition_from_unadaptable(row: UnadaptableRow) -> DispositionRow:
    """Copy an unadaptable staging row into the unified ledger without identity fields."""
    return DispositionRow(
        archive_member=row.archive_member,
        row_number=row.row_number,
        state=row.state,
        issues=row.issues,
    )


def finalize_event_group(rows: Sequence[StagedBattleRow]) -> tuple[DispositionRow, ...]:
    """Assign final states for one event_key group ordered by source location."""
    ordered = tuple(sorted(rows, key=lambda row: (row.archive_member, row.row_number)))
    if not ordered:
        return ()
    fingerprints = {row.fingerprint for row in ordered}
    if len(fingerprints) > 1:
        return tuple(
            _adaptable_disposition(
                row,
                state=RecordState.QUARANTINED,
                issues=_sorted_issues((RecordIssue.CONFLICTING_BATTLE,), row.observation_issues),
                representative_archive_member=None,
                representative_row_number=None,
            )
            for row in ordered
        )

    representative = ordered[0]
    representative_member = representative.archive_member
    representative_row_number = representative.row_number
    if representative.observation_issues:
        lead = _adaptable_disposition(
            representative,
            state=RecordState.UNSUPPORTED,
            issues=representative.observation_issues,
            representative_archive_member=representative_member,
            representative_row_number=representative_row_number,
        )
    else:
        lead = _adaptable_disposition(
            representative,
            state=RecordState.VALID,
            issues=(),
            representative_archive_member=representative_member,
            representative_row_number=representative_row_number,
        )
    duplicates = tuple(
        _adaptable_disposition(
            row,
            state=RecordState.QUARANTINED,
            issues=_sorted_issues((RecordIssue.DUPLICATE_BATTLE,), row.observation_issues),
            representative_archive_member=representative_member,
            representative_row_number=representative_row_number,
        )
        for row in ordered[1:]
    )
    return (lead, *duplicates)


@dataclass(frozen=True)
class GroupingResult:
    summary: DispositionSummary
    disposition_files: tuple[Path, ...]
    row_count: int


def group_staged_dataset(
    staging_workspace: Path,
    output_dir: Path,
    *,
    source_row_count: int,
    config: StagingConfig,
    temp_directory: Path,
) -> GroupingResult:
    """Group staged Parquet externally and write one disposition file per archive member."""
    if source_row_count < 0:
        raise KaggleV6GroupingError("source_row_count must be non-negative")
    dispositions = output_dir / "dispositions"
    if dispositions.exists():
        raise KaggleV6GroupingError("disposition output already exists")
    temp_directory.mkdir(parents=True, exist_ok=True)
    staged_files = list_parquet_files(staging_workspace / "staging")
    unadaptable_files = list_parquet_files(staging_workspace / "unadaptable")
    connection = connect_staging_duckdb(
        memory_limit=config.memory_limit,
        threads=config.threads,
        temp_directory=temp_directory,
    )
    try:
        create_ledger(connection, staged_files, unadaptable_files)
        row_count, summary = summary_from_ledger(connection, source_row_count=source_row_count)
        files = write_disposition_files(
            connection,
            output_dir,
            row_group_rows=config.parquet_row_group_rows,
        )
        return GroupingResult(summary=summary, disposition_files=files, row_count=row_count)
    except BaseException:
        shutil.rmtree(dispositions, ignore_errors=True)
        raise
    finally:
        connection.close()
