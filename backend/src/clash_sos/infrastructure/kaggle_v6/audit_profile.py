"""DuckDB schema and aggregate profiling for Kaggle version 6 Parquet."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import duckdb

from clash_sos.domain.manifests import (
    DatasetObservationsManifest,
    ModeCountManifest,
)
from clash_sos.infrastructure.kaggle_v6.schema import CARD_COLUMNS, KaggleSchemaColumn


@dataclass(frozen=True)
class ParquetProfile:
    row_count: int
    timestamp_min: datetime | None
    timestamp_max: datetime | None
    mode_counts: Counter[str]
    card_id_min: int | None
    card_id_max: int | None


def read_parquet_schema(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> tuple[KaggleSchemaColumn, ...]:
    rows = cast(
        list[tuple[object, ...]],
        connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall(),
    )
    return tuple(
        KaggleSchemaColumn(name=str(row[0]), physical_type=str(row[1]), nullable=row[2] == "YES")
        for row in rows
    )


def profile_parquet(connection: duckdb.DuckDBPyConnection, path: Path) -> ParquetProfile:
    least_cards = f"least({', '.join(CARD_COLUMNS)})"
    greatest_cards = f"greatest({', '.join(CARD_COLUMNS)})"
    query = f"""
        SELECT game_mode, count(*), min(time)::VARCHAR, max(time)::VARCHAR,
               min({least_cards}), max({greatest_cards})
        FROM read_parquet(?) GROUP BY game_mode ORDER BY game_mode NULLS FIRST
    """
    rows = cast(list[tuple[object, ...]], connection.execute(query, [str(path)]).fetchall())
    modes: Counter[str] = Counter()
    row_count = 0
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None
    card_min: int | None = None
    card_max: int | None = None
    for mode, count, minimum_time, maximum_time, minimum_card, maximum_card in rows:
        modes[str(mode) if mode is not None else "<null>"] += int(cast(int, count))
        row_count += int(cast(int, count))
        timestamp_min = _minimum_time(timestamp_min, minimum_time)
        timestamp_max = _maximum_time(timestamp_max, maximum_time)
        card_min = _minimum_integer(card_min, minimum_card)
        card_max = _maximum_integer(card_max, maximum_card)
    return ParquetProfile(row_count, timestamp_min, timestamp_max, modes, card_min, card_max)


def combine_profiles(profiles: list[ParquetProfile]) -> DatasetObservationsManifest:
    modes: Counter[str] = Counter()
    for profile in profiles:
        modes.update(profile.mode_counts)
    ordered_modes = tuple(sorted(modes))
    return DatasetObservationsManifest(
        row_count=sum(profile.row_count for profile in profiles),
        timestamp_column="time",
        timestamp_min=min(
            (profile.timestamp_min for profile in profiles if profile.timestamp_min), default=None
        ),
        timestamp_max=max(
            (profile.timestamp_max for profile in profiles if profile.timestamp_max), default=None
        ),
        modes=ordered_modes,
        mode_counts=tuple(
            ModeCountManifest(mode=mode, row_count=modes[mode]) for mode in ordered_modes
        ),
        card_id_min=min(
            (profile.card_id_min for profile in profiles if profile.card_id_min is not None),
            default=None,
        ),
        card_id_max=max(
            (profile.card_id_max for profile in profiles if profile.card_id_max is not None),
            default=None,
        ),
    )


def _minimum_time(current: datetime | None, value: object) -> datetime | None:
    parsed = datetime.fromisoformat(str(value)) if value is not None else None
    present = tuple(item for item in (current, parsed) if item is not None)
    return min(present) if present else None


def _maximum_time(current: datetime | None, value: object) -> datetime | None:
    parsed = datetime.fromisoformat(str(value)) if value is not None else None
    present = tuple(item for item in (current, parsed) if item is not None)
    return max(present) if present else None


def _minimum_integer(current: int | None, value: object) -> int | None:
    parsed = int(cast(int, value)) if value is not None else None
    present = tuple(item for item in (current, parsed) if item is not None)
    return min(present) if present else None


def _maximum_integer(current: int | None, value: object) -> int | None:
    parsed = int(cast(int, value)) if value is not None else None
    present = tuple(item for item in (current, parsed) if item is not None)
    return max(present) if present else None
