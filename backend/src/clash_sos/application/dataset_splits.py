"""Application workflows that write leakage-checked dataset split files."""

from datetime import datetime
from pathlib import Path

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.processed_manifest import (
    DEFAULT_PLAYER_HASH_SEED,
    DEFAULT_PLAYER_TRAIN_MAX,
    DEFAULT_PLAYER_VALIDATION_MAX,
    PlayerDisjointSplitManifest,
    TemporalSplitManifest,
)
from clash_sos.infrastructure.kaggle_v6.split_io import (
    KaggleV6SplitError as KaggleV6SplitError,
)
from clash_sos.infrastructure.kaggle_v6.split_io import (
    write_player_disjoint_split_parquet,
    write_temporal_split_parquet,
)
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb


def write_temporal_split(
    canonical_path: Path,
    output_path: Path,
    *,
    train_end: datetime,
    validation_end: datetime,
    config: StagingConfig,
    temp_directory: Path,
) -> TemporalSplitManifest:
    """Write splits-temporal.parquet with half-open UTC cutovers and one row per battle."""
    if output_path.exists():
        raise KaggleV6SplitError("temporal split already exists")
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = connect_staging_duckdb(
        memory_limit=config.memory_limit,
        threads=1,
        temp_directory=temp_directory,
    )
    try:
        return write_temporal_split_parquet(
            connection,
            canonical_path=canonical_path,
            output_path=output_path,
            train_end=train_end,
            validation_end=validation_end,
            row_group_rows=config.parquet_row_group_rows,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()


def write_player_disjoint_split(
    canonical_path: Path,
    output_path: Path,
    *,
    config: StagingConfig,
    temp_directory: Path,
    player_seed: int = DEFAULT_PLAYER_HASH_SEED,
    player_train_max: float = DEFAULT_PLAYER_TRAIN_MAX,
    player_validation_max: float = DEFAULT_PLAYER_VALIDATION_MAX,
) -> PlayerDisjointSplitManifest:
    """Write retained same-partition battles; bridges are counted and omitted."""
    if output_path.exists():
        raise KaggleV6SplitError("player-disjoint split already exists")
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = connect_staging_duckdb(
        memory_limit=config.memory_limit,
        threads=1,
        temp_directory=temp_directory,
    )
    try:
        return write_player_disjoint_split_parquet(
            connection,
            canonical_path=canonical_path,
            output_path=output_path,
            seed=player_seed,
            train_max=player_train_max,
            validation_max=player_validation_max,
            row_group_rows=config.parquet_row_group_rows,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
