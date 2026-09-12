"""Assemble processed manifests, verification reports, and publication cleanup.

Accepted summaries are DuckDB aggregates over `canonical.parquet`. Frozen checks
raise unless every artifact passes. Manifest assembly inventories hashed files
and writes `manifest.json` last. Publication is a same-filesystem rename.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.canonical_dataset import CANONICAL_SCHEMA_VERSION
from clash_sos.domain.dataset_splits import VERIFICATION_CHECK_IDS
from clash_sos.domain.processed_manifest import (
    DEFAULT_DATASET_VERSION,
    PREPARATION_POLICY_VERSION,
    AcceptedSummary,
    DispositionSummary,
    EraCount,
    PlayerDisjointSplitManifest,
    ProcessedDatasetManifest,
    ProcessedOutputFile,
    ProcessedValidation,
    ProcessingConfiguration,
    TemporalSplitManifest,
    VerificationCheckResult,
    VerificationReport,
    dump_processed_manifest,
    dump_verification_report,
)
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.balance_eras import KAGGLE_V6_ERA_REGISTRY_VERSION
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.grouping_io import list_parquet_files
from clash_sos.infrastructure.kaggle_v6.publish_io import (
    check_prepare_preconditions,
    cleanup_prepare_workspaces,
)
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_SOURCE_ID
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb, python_cell
from clash_sos.infrastructure.kaggle_v6.verify_io import (
    KaggleV6VerifyError,
    verify_processed_artifacts,
)

FileKind = Literal[
    "canonical",
    "disposition",
    "temporal_split",
    "player_disjoint_split",
    "verification_report",
]


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


def _inventory_file(
    path: Path,
    root: Path,
    kind: FileKind,
    *,
    chunk_size: int,
    logical_sha256: str | None = None,
) -> ProcessedOutputFile:
    size_bytes, digest = hash_file(path, chunk_size)
    return ProcessedOutputFile(
        path=path.relative_to(root).as_posix(),
        kind=kind,
        size_bytes=size_bytes,
        sha256=digest,
        logical_sha256=logical_sha256,
    )


def assemble_processed_manifest(
    output_dir: Path,
    *,
    config: StagingConfig,
    temporal_split: TemporalSplitManifest,
    player_disjoint_split: PlayerDisjointSplitManifest,
    dispositions: DispositionSummary,
    accepted: AcceptedSummary,
    canonical_logical_sha256: str,
    raw_manifest_path: Path,
    archive_sha256: str,
    source_schema_fingerprint: str,
    source_members: tuple[str, ...],
    dataset_version: str = DEFAULT_DATASET_VERSION,
) -> ProcessedDatasetManifest:
    """Inventory hashed processed artifacts; omit manifest.json from the file list."""
    files = [
        _inventory_file(
            output_dir / "canonical.parquet",
            output_dir,
            "canonical",
            chunk_size=config.chunk_size,
            logical_sha256=canonical_logical_sha256,
        )
    ]
    for path in list_parquet_files(output_dir / "dispositions"):
        files.append(_inventory_file(path, output_dir, "disposition", chunk_size=config.chunk_size))
    files.append(
        _inventory_file(
            output_dir / "splits-player-disjoint.parquet",
            output_dir,
            "player_disjoint_split",
            chunk_size=config.chunk_size,
        )
    )
    files.append(
        _inventory_file(
            output_dir / "splits-temporal.parquet",
            output_dir,
            "temporal_split",
            chunk_size=config.chunk_size,
        )
    )
    files.append(
        _inventory_file(
            output_dir / "verification-report.json",
            output_dir,
            "verification_report",
            chunk_size=config.chunk_size,
        )
    )
    _, raw_audit_manifest_sha256 = hash_file(raw_manifest_path, config.chunk_size)
    return ProcessedDatasetManifest(
        dataset_version=dataset_version,
        preparation_policy_version=PREPARATION_POLICY_VERSION,
        canonical_schema_version=CANONICAL_SCHEMA_VERSION,
        catalog_version=KAGGLE_V6_CARDS.version,
        era_registry_version=KAGGLE_V6_ERA_REGISTRY_VERSION,
        source_id=KAGGLE_V6_SOURCE_ID,
        raw_audit_manifest_sha256=raw_audit_manifest_sha256,
        archive_sha256=archive_sha256,
        source_schema_fingerprint=source_schema_fingerprint,
        source_members=source_members,
        dispositions=dispositions,
        accepted=accepted,
        files=tuple(sorted(files, key=lambda file: file.path)),
        processing=ProcessingConfiguration(
            memory_limit=config.memory_limit,
            threads=config.threads,
            chunk_size=config.chunk_size,
        ),
        temporal_split=temporal_split,
        player_disjoint_split=player_disjoint_split,
        validation=ProcessedValidation(),
    )


def write_processed_manifest(output_dir: Path, manifest: ProcessedDatasetManifest) -> Path:
    """Write byte-stable manifest.json last. Raises if the file already exists."""
    path = output_dir / "manifest.json"
    if path.exists():
        raise KaggleV6PrepareError("processed manifest already exists")
    path.write_bytes(dump_processed_manifest(manifest))
    return path


def execute_with_prepare_cleanup[T](
    *,
    destination: Path,
    output_workspace: Path,
    staging_workspace: Path,
    body: Callable[[], T],
) -> T:
    """Run prepare work and delete workspaces on failure. Never deletes destination."""
    check_prepare_preconditions(
        destination=destination,
        output_workspace=output_workspace,
        staging_workspace=staging_workspace,
    )
    try:
        return body()
    except BaseException:
        cleanup_prepare_workspaces(output_workspace, staging_workspace)
        raise
