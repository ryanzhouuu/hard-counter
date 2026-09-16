"""Read-only verification of a published Kaggle v6 processed dataset version.

Composes manifest validation, inventoried-file hashing, and the existing DuckDB
artifact checks. Never writes under the published version directory.
"""

from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.processed_manifest import ProcessedDatasetManifest, ProcessedOutputFile
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb
from clash_sos.infrastructure.kaggle_v6.verify_io import (
    KaggleV6VerifyError,
    verify_processed_artifacts,
)


class KaggleV6DatasetVerifyError(ValueError):
    pass


def _inventoried_path(version_dir: Path, file: ProcessedOutputFile) -> Path:
    path = version_dir / file.path
    if not path.is_file():
        raise KaggleV6DatasetVerifyError(f"missing inventoried file: {file.path}")
    return path


def _kind_path(
    manifest: ProcessedDatasetManifest,
    version_dir: Path,
    kind: Literal["canonical", "temporal_split", "player_disjoint_split"],
) -> Path:
    match = next(file for file in manifest.files if file.kind == kind)
    return version_dir / match.path


def verify_kaggle_v6_dataset(
    version_dir: Path,
    *,
    config: StagingConfig,
    temp_directory: Path,
) -> ProcessedDatasetManifest:
    """Validate a published version in place. Never writes under version_dir."""
    manifest_path = version_dir / "manifest.json"
    if not manifest_path.is_file():
        raise KaggleV6DatasetVerifyError("processed manifest is required")
    try:
        manifest = ProcessedDatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise KaggleV6DatasetVerifyError("processed manifest is invalid") from error
    for file in manifest.files:
        path = _inventoried_path(version_dir, file)
        size, digest = hash_file(path, config.chunk_size)
        if size != file.size_bytes or digest != file.sha256:
            raise KaggleV6DatasetVerifyError(f"inventoried file hash mismatch: {file.path}")
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = connect_staging_duckdb(
        memory_limit=config.memory_limit,
        threads=config.threads,
        temp_directory=temp_directory,
    )
    try:
        verify_processed_artifacts(
            connection,
            canonical_path=_kind_path(manifest, version_dir, "canonical"),
            disposition_files=tuple(
                version_dir / file.path for file in manifest.files if file.kind == "disposition"
            ),
            temporal_split_path=_kind_path(manifest, version_dir, "temporal_split"),
            player_split_path=_kind_path(manifest, version_dir, "player_disjoint_split"),
            train_end=manifest.temporal_split.train_end,
            validation_end=manifest.temporal_split.validation_end,
            excluded_bridge_rows=manifest.player_disjoint_split.excluded_bridge_rows,
        )
    except KaggleV6VerifyError as error:
        raise KaggleV6DatasetVerifyError(str(error)) from error
    finally:
        connection.close()
    return manifest
