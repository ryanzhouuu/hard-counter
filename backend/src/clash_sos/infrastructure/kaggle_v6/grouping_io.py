"""DuckDB ledger construction and Parquet writes for disposition grouping."""

from pathlib import Path, PurePosixPath

import duckdb

from clash_sos.domain.canonical import RecordIssue, RecordState
from clash_sos.domain.processed_manifest import (
    INGESTION_ISSUES,
    RECORD_STATES,
    DispositionSummary,
    IssueCount,
    StateCount,
)

_LEDGER_COLUMNS = """
    archive_member, row_number, state, issues, source_id, timestamp, mode,
    balance_era_id, outcome, event_key, fingerprint, side_a_player_id,
    side_b_player_id, representative_archive_member, representative_row_number
"""

_STAGED_LEDGER_SQL = """
CREATE TEMP TABLE staged_ledger AS
SELECT
    archive_member, row_number,
    CASE
        WHEN fingerprint_count > 1 OR source_rank > 1 THEN 'quarantined'
        WHEN len(observation_issues) = 0 THEN 'valid'
        ELSE 'unsupported'
    END AS state,
    CASE
        WHEN fingerprint_count > 1 THEN list_sort(
            list_distinct(list_concat(observation_issues, ['conflicting_battle']))
        )
        WHEN source_rank > 1 THEN list_sort(
            list_distinct(list_concat(observation_issues, ['duplicate_battle']))
        )
        ELSE observation_issues
    END AS issues,
    source_id, timestamp, mode, balance_era_id, outcome, event_key, fingerprint,
    side_a_player_id, side_b_player_id,
    CASE WHEN fingerprint_count > 1 THEN NULL ELSE group_representative_member END
        AS representative_archive_member,
    CASE WHEN fingerprint_count > 1 THEN NULL ELSE group_representative_row END
        AS representative_row_number
FROM (
    SELECT *,
        COUNT(DISTINCT fingerprint) OVER g AS fingerprint_count,
        ROW_NUMBER() OVER o AS source_rank,
        first_value(archive_member) OVER o AS group_representative_member,
        first_value(row_number) OVER o AS group_representative_row
    FROM staged
    WINDOW
        g AS (PARTITION BY event_key),
        o AS (PARTITION BY event_key ORDER BY archive_member, row_number)
)
"""

_UNADAPTABLE_LEDGER_SQL = """
CREATE TEMP TABLE unadaptable_ledger AS
SELECT
    archive_member, row_number, state, issues,
    NULL::VARCHAR AS source_id, NULL::TIMESTAMPTZ AS timestamp,
    NULL::VARCHAR AS mode, NULL::VARCHAR AS balance_era_id,
    NULL::VARCHAR AS outcome, NULL::VARCHAR AS event_key,
    NULL::VARCHAR AS fingerprint, NULL::VARCHAR AS side_a_player_id,
    NULL::VARCHAR AS side_b_player_id,
    NULL::VARCHAR AS representative_archive_member,
    NULL::BIGINT AS representative_row_number
FROM read_parquet(?)
"""


class KaggleV6GroupingError(ValueError):
    pass


def list_parquet_files(directory: Path) -> tuple[Path, ...]:
    """Return sorted Parquet files under a directory, or empty if it is absent."""
    if not directory.exists():
        return ()
    return tuple(sorted(path for path in directory.rglob("*.parquet") if path.is_file()))


def disposition_file_path(output_dir: Path, archive_member: str) -> Path:
    """Build dispositions/{member_stem}.parquet from a source member name."""
    return output_dir / "dispositions" / f"{PurePosixPath(archive_member).stem}.parquet"


def create_ledger(
    connection: duckdb.DuckDBPyConnection,
    staged_files: tuple[Path, ...],
    unadaptable_files: tuple[Path, ...],
) -> None:
    """Build a temp `ledger` table from staged and unadaptable Parquet files."""
    execute = connection.execute
    if staged_files:
        execute(
            "CREATE TEMP TABLE staged AS SELECT * FROM read_parquet(?)",
            [str(path) for path in staged_files],
        )
        execute(_STAGED_LEDGER_SQL)
    if unadaptable_files:
        execute(_UNADAPTABLE_LEDGER_SQL, [[str(path) for path in unadaptable_files]])
    if staged_files and unadaptable_files:
        execute(
            f"CREATE TEMP TABLE ledger AS SELECT {_LEDGER_COLUMNS} FROM staged_ledger "
            f"UNION ALL SELECT {_LEDGER_COLUMNS} FROM unadaptable_ledger"
        )
    elif staged_files:
        execute(f"CREATE TEMP TABLE ledger AS SELECT {_LEDGER_COLUMNS} FROM staged_ledger")
    elif unadaptable_files:
        execute(f"CREATE TEMP TABLE ledger AS SELECT {_LEDGER_COLUMNS} FROM unadaptable_ledger")


def summary_from_ledger(
    connection: duckdb.DuckDBPyConnection, *, source_row_count: int
) -> tuple[int, DispositionSummary]:
    """Reconcile ledger counts with the audited source row count."""
    execute = connection.execute
    duplicates = execute(
        "SELECT archive_member, row_number FROM ledger GROUP BY 1, 2 HAVING COUNT(*) > 1"
    ).fetchall()
    if duplicates:
        raise KaggleV6GroupingError("source locations must be unique")
    counted = execute("SELECT COUNT(*) FROM ledger").fetchone()
    if counted is None:
        raise KaggleV6GroupingError("disposition row count must equal source_row_count")
    row_count = int(counted[0])
    if row_count != source_row_count:
        raise KaggleV6GroupingError("disposition row count must equal source_row_count")
    state_counts = {
        RecordState(str(state)): int(count)
        for state, count in execute("SELECT state, COUNT(*) FROM ledger GROUP BY 1").fetchall()
    }
    issue_counts = {
        RecordIssue(str(issue)): int(count)
        for issue, count in execute(
            "SELECT issue, COUNT(*) FROM ledger, UNNEST(issues) AS u(issue) GROUP BY 1"
        ).fetchall()
    }
    return row_count, DispositionSummary(
        source_row_count=source_row_count,
        states=tuple(
            StateCount(state=state, count=state_counts.get(state, 0)) for state in RECORD_STATES
        ),
        issues=tuple(
            IssueCount(issue=issue, count=issue_counts.get(issue, 0)) for issue in INGESTION_ISSUES
        ),
        duplicate_row_count=issue_counts.get(RecordIssue.DUPLICATE_BATTLE, 0),
        conflict_row_count=issue_counts.get(RecordIssue.CONFLICTING_BATTLE, 0),
    )


def write_disposition_files(
    connection: duckdb.DuckDBPyConnection,
    output_dir: Path,
    *,
    row_group_rows: int,
) -> tuple[Path, ...]:
    """Write one zstd disposition Parquet file per archive member."""
    execute = connection.execute
    members = [
        str(row[0])
        for row in execute("SELECT DISTINCT archive_member FROM ledger ORDER BY 1").fetchall()
    ]
    files: list[Path] = []
    for member in members:
        path = disposition_file_path(output_dir, member)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = str(path).replace("'", "''")
        execute(
            f"""
            COPY (
                SELECT {_LEDGER_COLUMNS} FROM ledger
                WHERE archive_member = ? ORDER BY row_number
            ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(row_group_rows)})
            """,
            [member],
        )
        files.append(path)
    return tuple(files)
