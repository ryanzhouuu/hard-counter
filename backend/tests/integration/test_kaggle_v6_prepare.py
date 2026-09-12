import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import duckdb
import pytest

from clash_sos.application.dataset_prepare import KaggleV6PrepareError, prepare_kaggle_v6_dataset
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.manifests import (
    ArchiveManifest,
    DatasetFileManifest,
    DatasetManifest,
    DatasetObservationsManifest,
    DatasetSchemaManifest,
    DatasetValidationManifest,
    ModeCountManifest,
    SchemaColumnManifest,
)
from clash_sos.domain.processed_manifest import (
    ProcessedDatasetManifest,
    player_hash_fraction,
    player_partition,
)
from clash_sos.infrastructure.kaggle_v6.audit import write_dataset_manifest
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.publish_io import KaggleV6PublishError
from clash_sos.infrastructure.kaggle_v6.schema import KAGGLE_V6_SCHEMA, validate_kaggle_v6_schema
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_SOURCE_ID

CONFIG = StagingConfig(threads=1, memory_limit="256MB", batch_rows=8)
TRAIN_END = datetime(2026, 6, 10, tzinfo=UTC)
VALIDATION_END = datetime(2026, 6, 20, tzinfo=UTC)


def _grouping_helpers() -> Any:
    path = Path(__file__).with_name("test_kaggle_v6_grouping.py")
    spec = importlib.util.spec_from_file_location("kaggle_v6_grouping_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("grouping helpers are unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _player_tag(partition: str, *, used: set[str]) -> str:
    for index in range(1, 10_000):
        tag = f"#P{index:04d}"
        if tag in used:
            continue
        if player_partition(player_hash_fraction(tag)) == partition:
            used.add(tag)
            return tag
    raise AssertionError(f"no player tag hashed to {partition}")


def _create_prepare_archive(directory: Path) -> tuple[Path, Path, Path]:
    helpers = _grouping_helpers()
    used: set[str] = set()
    train_a = _player_tag("train", used=used)
    train_b = _player_tag("train", used=used)
    validation_a = _player_tag("validation", used=used)
    a_path = directory / "a.parquet"
    helpers.write_kaggle_parquet(
        a_path,
        [
            helpers.kaggle_row(
                time=datetime(2026, 6, 9, tzinfo=UTC),
                winner_id=train_a,
                loser_id=train_b,
            ),
            helpers.kaggle_row(
                time=TRAIN_END,
                winner_id=train_b,
                loser_id=train_a,
            ),
            helpers.kaggle_row(
                time=VALIDATION_END,
                winner_id=train_a,
                loser_id=validation_a,
            ),
        ],
    )
    mapping = directory / "cardToID.json"
    mapping.write_text(
        json.dumps({entry.source_name: entry.source_id for entry in KAGGLE_V6_CARDS.entries}),
        encoding="utf-8",
    )
    archive_path = directory / "source.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.write(a_path, "a.parquet")
        archive.write(mapping, "cardToID.json")
    return archive_path, a_path, mapping


def _write_raw_manifest(archive: Path, parquet: Path, mapping: Path, dest: Path) -> None:
    size, digest = hash_file(archive, CONFIG.chunk_size)
    parquet_size, parquet_hash = hash_file(parquet, CONFIG.chunk_size)
    mapping_size, mapping_hash = hash_file(mapping, CONFIG.chunk_size)
    write_dataset_manifest(
        DatasetManifest(
            source_id=KAGGLE_V6_SOURCE_ID,
            archive=ArchiveManifest(path=archive.name, size_bytes=size, sha256=digest),
            files=(
                DatasetFileManifest(
                    path=parquet.name,
                    kind="parquet",
                    size_bytes=parquet_size,
                    sha256=parquet_hash,
                    row_count=3,
                ),
                DatasetFileManifest(
                    path=mapping.name,
                    kind="card_mapping",
                    size_bytes=mapping_size,
                    sha256=mapping_hash,
                ),
            ),
            schema=DatasetSchemaManifest(
                format="parquet",
                columns=tuple(
                    SchemaColumnManifest(
                        name=column.name,
                        physical_type=column.physical_type,
                        nullable=column.nullable,
                    )
                    for column in KAGGLE_V6_SCHEMA
                ),
                fingerprint=validate_kaggle_v6_schema(KAGGLE_V6_SCHEMA),
            ),
            observations=DatasetObservationsManifest(
                row_count=3,
                timestamp_column="time",
                timestamp_min=datetime(2026, 6, 9, tzinfo=UTC),
                timestamp_max=VALIDATION_END,
                modes=("Ranked1v1_NewArena",),
                mode_counts=(ModeCountManifest(mode="Ranked1v1_NewArena", row_count=3),),
                card_id_min=0,
                card_id_max=15,
            ),
            validation=DatasetValidationManifest(status="not_run"),
        ),
        dest,
    )


def _prepare(tmp_path: Path, destination: Path, archive: Path, raw_manifest: Path) -> Path:
    suffix = destination.name
    return prepare_kaggle_v6_dataset(
        archive,
        destination,
        staging_workspace=tmp_path / f"staging-{suffix}",
        output_workspace=tmp_path / f"output-{suffix}",
        temp_directory=tmp_path / f"tmp-{suffix}",
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
        config=CONFIG,
        raw_manifest_path=raw_manifest,
    )


def test_prepare_kaggle_v6_dataset_repeats_to_two_destinations(tmp_path: Path) -> None:
    archive, parquet, mapping = _create_prepare_archive(tmp_path)
    raw_manifest = tmp_path / "raw-audit.json"
    _write_raw_manifest(archive, parquet, mapping, raw_manifest)
    dest_a = tmp_path / "dest-a"
    dest_b = tmp_path / "dest-b"
    published_a = _prepare(tmp_path, dest_a, archive, raw_manifest)
    published_b = _prepare(tmp_path, dest_b, archive, raw_manifest)
    assert published_a.joinpath("verification-report.json").read_bytes() == (
        published_b.joinpath("verification-report.json").read_bytes()
    )
    assert published_a.joinpath("manifest.json").read_bytes() == (
        published_b.joinpath("manifest.json").read_bytes()
    )
    assert not any(published_a.rglob("staging"))
    manifest = ProcessedDatasetManifest.model_validate_json(
        published_a.joinpath("manifest.json").read_text(encoding="utf-8")
    )
    connection = duckdb.connect()
    try:
        temporal_rows = connection.execute(
            "SELECT partition, COUNT(*) FROM read_parquet(?) GROUP BY 1",
            [str(published_a / "splits-temporal.parquet")],
        ).fetchall()
        retained_row = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(published_a / "splits-player-disjoint.parquet")],
        ).fetchone()
    finally:
        connection.close()
    temporal_counts = {str(partition): int(count) for partition, count in temporal_rows}
    assert {
        partition.partition: partition.row_count for partition in manifest.temporal_split.partitions
    } == temporal_counts
    retained = 0 if retained_row is None else int(retained_row[0])
    assert retained + manifest.player_disjoint_split.excluded_bridge_rows == (
        manifest.accepted.row_count
    )
    first_manifest = published_a.joinpath("manifest.json").read_bytes()
    with pytest.raises((KaggleV6PrepareError, KaggleV6PublishError), match="already exists"):
        _prepare(tmp_path, dest_a, archive, raw_manifest)
    assert published_a.joinpath("manifest.json").read_bytes() == first_manifest
