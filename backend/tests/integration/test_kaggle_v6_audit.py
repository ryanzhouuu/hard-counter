import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import duckdb
import pytest

from clash_sos.infrastructure.kaggle_v6.audit import (
    audit_kaggle_v6_archive,
    write_dataset_manifest,
)
from clash_sos.infrastructure.kaggle_v6.audit_io import KaggleV6AuditError
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.schema import CARD_COLUMNS, LEVEL_COLUMNS


def create_parquet(
    path: Path,
    mode: str = "Ranked1v1_NewArena",
    card_type: str = "UTINYINT",
) -> None:
    columns = [
        "winner_id VARCHAR",
        "loser_id VARCHAR",
        "time TIMESTAMPTZ",
        "game_mode VARCHAR",
        *(f"{name} {card_type}" for name in CARD_COLUMNS),
        *(f"{name} UTINYINT" for name in LEVEL_COLUMNS),
    ]
    values: list[object] = [
        "#AAA",
        "#BBB",
        "2026-06-21T12:00:00-04:00",
        mode,
        *range(16),
        *((16,) * 16),
    ]
    connection = duckdb.connect()
    try:
        connection.execute(f"CREATE TABLE battles ({', '.join(columns)})")
        placeholders = ", ".join("?" for _ in values)
        connection.execute(f"INSERT INTO battles VALUES ({placeholders})", values)
        connection.execute("COPY battles TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()


def create_archive(directory: Path, *, incompatible_schema: bool = False) -> Path:
    member_paths: list[Path] = []
    for index in range(23):
        member = directory / f"2026-06-21_{index}.parquet"
        create_parquet(
            member,
            mode="Ladder" if index == 0 else "Ranked1v1_NewArena",
            card_type="USMALLINT" if incompatible_schema and index == 0 else "UTINYINT",
        )
        member_paths.append(member)
    mapping = directory / "cardToID.json"
    mapping.write_text(
        json.dumps({entry.source_name: entry.source_id for entry in KAGGLE_V6_CARDS.entries}),
        encoding="utf-8",
    )
    archive_path = directory / "source.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        for path in (*member_paths, mapping):
            archive.write(path, path.name)
    return archive_path


def test_audit_profiles_all_members_and_reconciles_rows(tmp_path: Path) -> None:
    archive_path = create_archive(tmp_path)

    manifest = audit_kaggle_v6_archive(
        archive_path,
        temp_directory=tmp_path / "work",
        expected_archive_size=None,
        expected_archive_sha256=None,
    )

    assert len(manifest.files) == 24
    assert manifest.observations.row_count == 23
    assert manifest.observations.timestamp_min is not None
    assert manifest.observations.timestamp_min.isoformat() == "2026-06-21T16:00:00+00:00"
    assert manifest.observations.modes == ("Ladder", "Ranked1v1_NewArena")
    assert [count.row_count for count in manifest.observations.mode_counts] == [1, 22]
    assert manifest.observations.card_id_min == 0
    assert manifest.observations.card_id_max == 15
    assert manifest.validation.status == "passed"
    assert all(file.row_count == 1 for file in manifest.files if file.kind == "parquet")


def test_identical_audits_write_byte_identical_manifests(tmp_path: Path) -> None:
    archive_path = create_archive(tmp_path)
    first = audit_kaggle_v6_archive(
        archive_path,
        temp_directory=tmp_path / "work",
        expected_archive_size=None,
        expected_archive_sha256=None,
    )
    second = audit_kaggle_v6_archive(
        archive_path,
        temp_directory=tmp_path / "work",
        expected_archive_size=None,
        expected_archive_sha256=None,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_dataset_manifest(first, first_path)
    write_dataset_manifest(second, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert not list(first_path.parent.glob("tmp*"))


def test_audit_rejects_unpinned_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(b"not the pinned archive")

    with pytest.raises(KaggleV6AuditError, match="size"):
        audit_kaggle_v6_archive(archive_path, temp_directory=tmp_path / "work")


def test_audit_rejects_incompatible_member_schema(tmp_path: Path) -> None:
    archive_path = create_archive(tmp_path, incompatible_schema=True)

    with pytest.raises(KaggleV6AuditError, match="incompatible schema"):
        audit_kaggle_v6_archive(
            archive_path,
            temp_directory=tmp_path / "work",
            expected_archive_size=None,
            expected_archive_sha256=None,
        )
