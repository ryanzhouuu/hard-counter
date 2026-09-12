"""Application workflows that write leakage-checked dataset split files."""

from datetime import datetime
from pathlib import Path

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.processed_manifest import TemporalSplitManifest
from clash_sos.infrastructure.kaggle_v6.split_io import (
    KaggleV6SplitError as KaggleV6SplitError,
)
from clash_sos.infrastructure.kaggle_v6.split_io import write_temporal_split_parquet
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
