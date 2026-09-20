"""DuckDB Parquet aggregates and joins for disposition grouping.

Never windows the corpus into DuckDB temp. Identity keys are projected to
prefix-partitioned Parquet, each prefix is aggregated with GROUP BY event_key,
and dispositions are joined per archive member.
"""

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
SELECT
    s.archive_member, s.row_number,
    CASE
        WHEN g.fingerprint_count > 1 THEN 'quarantined'
        WHEN s.archive_member IS DISTINCT FROM g.representative_archive_member
            OR s.row_number IS DISTINCT FROM g.representative_row_number
            THEN 'quarantined'
        WHEN len(s.observation_issues) = 0 THEN 'valid'
        ELSE 'unsupported'
    END AS state,
    CASE
        WHEN g.fingerprint_count > 1 THEN list_sort(
            list_distinct(list_concat(s.observation_issues, ['conflicting_battle']))
        )
        WHEN s.archive_member IS DISTINCT FROM g.representative_archive_member
            OR s.row_number IS DISTINCT FROM g.representative_row_number
            THEN list_sort(
                list_distinct(list_concat(s.observation_issues, ['duplicate_battle']))
            )
        ELSE s.observation_issues
    END AS issues,
    s.source_id, s.timestamp, s.mode, s.balance_era_id, s.outcome, s.event_key, s.fingerprint,
    s.side_a_player_id, s.side_b_player_id,
    CASE WHEN g.fingerprint_count > 1 THEN NULL ELSE g.representative_archive_member END
        AS representative_archive_member,
    CASE WHEN g.fingerprint_count > 1 THEN NULL ELSE g.representative_row_number END
        AS representative_row_number
FROM read_parquet(?, hive_partitioning = false) AS s
INNER JOIN read_parquet(?, hive_partitioning = false) AS g
    ON s.event_key = g.event_key
WHERE s.archive_member = ?
"""

_UNADAPTABLE_LEDGER_SQL = """
SELECT
    archive_member, row_number, state, issues,
    NULL::VARCHAR AS source_id, NULL::TIMESTAMPTZ AS timestamp,
    NULL::VARCHAR AS mode, NULL::VARCHAR AS balance_era_id,
    NULL::VARCHAR AS outcome, NULL::VARCHAR AS event_key,
    NULL::VARCHAR AS fingerprint, NULL::VARCHAR AS side_a_player_id,
    NULL::VARCHAR AS side_b_player_id,
    NULL::VARCHAR AS representative_archive_member,
    NULL::BIGINT AS representative_row_number
FROM read_parquet(?, hive_partitioning = false)
WHERE archive_member = ?
"""


class KaggleV6GroupingError(ValueError):
    pass


def _sql_path(path: Path) -> str:
    """Quote a filesystem path for interpolation into a DuckDB COPY target."""
    return str(path).replace("'", "''")


def _sql_file_list(files: tuple[Path, ...]) -> str:
    """Render a DuckDB list literal of quoted Parquet paths."""
    quoted = ", ".join(f"'{_sql_path(path)}'" for path in files)
    return f"[{quoted}]"


def list_parquet_files(directory: Path) -> tuple[Path, ...]:
    """Return sorted Parquet files under a directory, or empty if it is absent."""
    if not directory.exists():
        return ()
    return tuple(sorted(path for path in directory.rglob("*.parquet") if path.is_file()))


def disposition_file_path(output_dir: Path, archive_member: str) -> Path:
    """Build dispositions/{member_stem}.parquet from a source member name."""
    return output_dir / "dispositions" / f"{PurePosixPath(archive_member).stem}.parquet"


def parquet_files_for_member(files: tuple[Path, ...], archive_member: str) -> tuple[Path, ...]:
    """Restrict part files to the staging directory named after the member stem."""
    stem = PurePosixPath(archive_member).stem
    return tuple(path for path in files if path.parent.name == stem)


def write_identity_projection(
    connection: duckdb.DuckDBPyConnection,
    staged_files: tuple[Path, ...],
    identity_dir: Path,
    *,
    row_group_rows: int,
) -> None:
    """Write event_key prefixes and identity columns; omit staged deck payloads."""
    if not staged_files:
        raise KaggleV6GroupingError("staged parquet is required for identity projection")
    identity_dir.mkdir(parents=True, exist_ok=True)
    target = _sql_path(identity_dir)
    connection.execute(
        f"""
        COPY (
            SELECT
                substring(event_key, 1, 2) AS prefix,
                event_key,
                fingerprint,
                archive_member,
                row_number
            FROM read_parquet(?, hive_partitioning = false)
        ) TO '{target}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            PARTITION_BY (prefix),
            ROW_GROUP_SIZE {int(row_group_rows)}
        )
        """,
        [[str(path) for path in staged_files]],
    )


def write_event_groups(
    connection: duckdb.DuckDBPyConnection,
    identity_dir: Path,
    groups_dir: Path,
    *,
    row_group_rows: int,
) -> tuple[Path, ...]:
    """GROUP BY event_key inside one prefix partition at a time and COPY to Parquet."""
    partitions = sorted(path for path in identity_dir.iterdir() if path.is_dir())
    if not partitions:
        raise KaggleV6GroupingError("identity projection produced no partitions")
    groups_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for index, partition in enumerate(partitions):
        parts = list_parquet_files(partition)
        if not parts:
            continue
        dest = groups_dir / f"part-{index:03d}.parquet"
        target = _sql_path(dest)
        connection.execute(
            f"""
            COPY (
                SELECT
                    event_key,
                    COUNT(DISTINCT fingerprint) AS fingerprint_count,
                    arg_min(archive_member, (archive_member, row_number))
                        AS representative_archive_member,
                    arg_min(row_number, (archive_member, row_number))
                        AS representative_row_number
                FROM read_parquet(?, hive_partitioning = false)
                GROUP BY event_key
            ) TO '{target}' (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE {int(row_group_rows)}
            )
            """,
            [[str(path) for path in parts]],
        )
        files.append(dest)
    if not files:
        raise KaggleV6GroupingError("event_key aggregates produced no parquet")
    return tuple(files)


def distinct_archive_members(
    connection: duckdb.DuckDBPyConnection, files: tuple[Path, ...]
) -> tuple[str, ...]:
    """Return sorted archive_member values stored inside Parquet parts."""
    if not files:
        return ()
    rows = connection.execute(
        """
        SELECT DISTINCT archive_member
        FROM read_parquet(?, hive_partitioning = false)
        ORDER BY 1
        """,
        [[str(path) for path in files]],
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def write_disposition_files(
    connection: duckdb.DuckDBPyConnection,
    output_dir: Path,
    *,
    staged_files: tuple[Path, ...],
    unadaptable_files: tuple[Path, ...],
    group_files: tuple[Path, ...],
    row_group_rows: int,
) -> tuple[Path, ...]:
    """Join one archive member at a time onto event_key aggregates and COPY zstd Parquet."""
    members = tuple(
        sorted(
            {
                *distinct_archive_members(connection, staged_files),
                *distinct_archive_members(connection, unadaptable_files),
            }
        )
    )
    files: list[Path] = []
    group_paths = [str(path) for path in group_files]
    for member in members:
        member_staged = parquet_files_for_member(staged_files, member)
        member_unadaptable = parquet_files_for_member(unadaptable_files, member)
        selects: list[str] = []
        params: list[object] = []
        if member_staged:
            if not group_files:
                raise KaggleV6GroupingError("event_key aggregates are required for staged rows")
            selects.append(_STAGED_LEDGER_SQL)
            params.extend(([str(path) for path in member_staged], group_paths, member))
        if member_unadaptable:
            selects.append(_UNADAPTABLE_LEDGER_SQL)
            params.extend(([str(path) for path in member_unadaptable], member))
        if not selects:
            continue
        body = " UNION ALL ".join(selects)
        path = disposition_file_path(output_dir, member)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = _sql_path(path)
        connection.execute(
            f"""
            COPY (
                SELECT {_LEDGER_COLUMNS} FROM (
                    {body}
                )
                ORDER BY row_number
            ) TO '{target}' (
                FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(row_group_rows)}
            )
            """,
            params,
        )
        files.append(path)
    return tuple(files)


def summary_from_ledger(
    connection: duckdb.DuckDBPyConnection,
    ledger_files: tuple[Path, ...],
    *,
    source_row_count: int,
) -> tuple[int, DispositionSummary]:
    """Reconcile disposition Parquet counts with the audited source row count."""
    if not ledger_files:
        raise KaggleV6GroupingError("disposition row count must equal source_row_count")
    execute = connection.execute
    execute(
        f"""
        CREATE OR REPLACE TEMP VIEW ledger AS
        SELECT {_LEDGER_COLUMNS}
        FROM read_parquet({_sql_file_list(ledger_files)}, hive_partitioning = false)
        """
    )
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
