"""Stream normalized JSONL into bounded Parquet parts and a frozen temporal split."""

from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

from clash_sos.domain.attention_dataset import official_row_type
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.canonical_dataset import CANONICAL_SCHEMA
from clash_sos.infrastructure.kaggle_v6.staging_io import polars_schema


def write_snapshot_parts(
    source: Path,
    directory: Path,
    *,
    schema: AttentionCardSchema,
    dataset_version: str,
    start: datetime,
    end: datetime,
    batch_rows: int,
) -> int:
    """Validate each row under the selected official mode contract before writing."""
    if batch_rows < 1:
        raise ValueError("snapshot batch size must be positive")
    row_type = official_row_type(schema.canonical_schema_version)
    directory.mkdir()
    columns = {
        **polars_schema(CANONICAL_SCHEMA),
        "side_a_tower": pl.String,
        "side_b_tower": pl.String,
        "side_a_tower_level": pl.UInt8,
        "side_b_tower_level": pl.UInt8,
    }
    records: list[dict[str, object]] = []
    count = 0
    part = 0

    def flush() -> None:
        nonlocal part
        pl.DataFrame(records, schema=columns).write_parquet(directory / f"{part:06d}.parquet")
        part += 1
        records.clear()

    with source.open() as input_file:
        for number, line in enumerate(input_file, 1):
            try:
                row = row_type.model_validate_json(line)
                if (
                    row.dataset_version != dataset_version
                    or row.balance_era_id != schema.balance_era_id
                    or not start <= row.timestamp < end
                ):
                    raise ValueError("row dataset, era, or timestamp disagrees with snapshot")
                for ids, forms, levels, tower, tower_level in (
                    (
                        row.side_a_card_ids,
                        row.side_a_card_forms,
                        row.side_a_card_levels,
                        row.side_a_tower,
                        row.side_a_tower_level,
                    ),
                    (
                        row.side_b_card_ids,
                        row.side_b_card_forms,
                        row.side_b_card_levels,
                        row.side_b_tower,
                        row.side_b_tower_level,
                    ),
                ):
                    schema.encode_side(
                        tuple(f"{card}:{form}" for card, form in zip(ids, forms, strict=True)),
                        levels=levels,
                        tower=tower,
                        tower_level=tower_level,
                    )
            except ValueError as error:
                raise ValueError(f"invalid normalized battle on line {number}: {error}") from error
            record = row.model_dump(mode="json")
            record["timestamp"] = row.timestamp
            records.append(record)
            count += 1
            if len(records) == batch_rows:
                flush()
    if records:
        flush()
    if not count:
        raise ValueError("official snapshot source is empty")
    return count


def export_snapshot(
    connection: duckdb.DuckDBPyConnection,
    workspace: Path,
    *,
    train_end: datetime,
    validation_end: datetime,
) -> dict[str, int]:
    parts = str(workspace / "parts/*.parquet")
    duplicate = connection.execute(
        "SELECT EXISTS(SELECT 1 FROM read_parquet(?) GROUP BY fingerprint HAVING COUNT(*) > 1) "
        "OR EXISTS(SELECT 1 FROM read_parquet(?) GROUP BY event_key HAVING COUNT(*) > 1)",
        [parts, parts],
    ).fetchone()
    if duplicate is None or duplicate[0]:
        raise ValueError("snapshot source contains duplicate or conflicting battle identities")
    connection.execute(
        "COPY (SELECT * FROM read_parquet($parts) ORDER BY timestamp, fingerprint, "
        "archive_member, row_number) TO $destination (FORMAT PARQUET, COMPRESSION ZSTD)",
        {"parts": parts, "destination": str(workspace / "canonical.parquet")},
    )
    connection.execute(
        "COPY (SELECT timestamp, fingerprint, archive_member, row_number, "
        "CASE WHEN timestamp < $train_end THEN 'train' "
        "WHEN timestamp < $validation_end THEN 'validation' "
        "ELSE 'test' END AS partition FROM read_parquet($parts) "
        "ORDER BY timestamp, fingerprint, archive_member, row_number) "
        "TO $destination (FORMAT PARQUET, COMPRESSION ZSTD)",
        {
            "train_end": train_end,
            "validation_end": validation_end,
            "parts": parts,
            "destination": str(workspace / "splits-temporal.parquet"),
        },
    )
    counts = connection.execute(
        "SELECT partition, COUNT(*) FROM read_parquet(?) GROUP BY partition",
        [str(workspace / "splits-temporal.parquet")],
    ).fetchall()
    return {str(name): int(count) for name, count in counts}
