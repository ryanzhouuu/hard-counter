"""Materialize accepted canonical Parquet from a grouped staging workspace.

Writes `canonical.parquet` beside existing disposition files. Does not publish a
processed dataset version, generate splits, or assemble a processed manifest.
"""

from dataclasses import dataclass
from pathlib import Path

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.canonical_dataset import stream_logical_canonical_content_hash
from clash_sos.domain.processed_manifest import DEFAULT_DATASET_VERSION
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.grouping_io import list_parquet_files
from clash_sos.infrastructure.kaggle_v6.materialize_io import (
    KaggleV6MaterializeError as KaggleV6MaterializeError,
)
from clash_sos.infrastructure.kaggle_v6.materialize_io import (
    iter_canonical_rows,
    write_canonical_parquet,
)
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb


@dataclass(frozen=True)
class MaterializeResult:
    canonical_path: Path
    row_count: int
    size_bytes: int
    sha256: str
    logical_sha256: str


def materialize_canonical_dataset(
    staging_workspace: Path,
    output_dir: Path,
    *,
    config: StagingConfig,
    temp_directory: Path,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    published_version: Path | None = None,
) -> MaterializeResult:
    """Write sorted canonical Parquet with physical and logical hashes.

    Uses a single-threaded DuckDB session so physical file bytes do not depend on
    `config.threads`. Raises if `published_version` already exists. On failure after
    the output file is created, deletes only `canonical.parquet`.
    """
    if published_version is not None and published_version.exists():
        raise KaggleV6MaterializeError("published dataset version already exists")
    output_path = output_dir / "canonical.parquet"
    if output_path.exists():
        raise KaggleV6MaterializeError("canonical parquet already exists")
    disposition_files = list_parquet_files(output_dir / "dispositions")
    if not disposition_files:
        raise KaggleV6MaterializeError("disposition parquet is required")
    staging_files = list_parquet_files(staging_workspace / "staging")
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = connect_staging_duckdb(
        memory_limit=config.memory_limit,
        threads=1,
        temp_directory=temp_directory,
    )
    try:
        row_count = write_canonical_parquet(
            connection,
            staging_files=staging_files,
            disposition_files=disposition_files,
            output_path=output_path,
            dataset_version=dataset_version,
            row_group_rows=config.parquet_row_group_rows,
        )
        size_bytes, sha256 = hash_file(output_path, config.chunk_size)
        logical_sha256 = stream_logical_canonical_content_hash(
            iter_canonical_rows(output_path, batch_rows=config.batch_rows)
        )
        return MaterializeResult(
            canonical_path=output_path,
            row_count=row_count,
            size_bytes=size_bytes,
            sha256=sha256,
            logical_sha256=logical_sha256,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
