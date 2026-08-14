import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pytest

from clash_sos.infrastructure.kaggle_v6.audit_io import (
    KaggleV6AuditError,
    hash_file,
    validate_archive_members,
    validate_card_mapping,
)
from clash_sos.infrastructure.kaggle_v6.audit_profile import (
    ParquetProfile,
    combine_profiles,
)
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def test_hash_file_streams_size_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"abcdef")

    assert hash_file(path, chunk_size=2) == (
        6,
        "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721",
    )


def test_archive_members_are_safe_complete_and_sorted(tmp_path: Path) -> None:
    path = tmp_path / "source.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("part.parquet", b"parquet")
        archive.writestr("cardToID.json", b"{}")

    with ZipFile(path) as archive:
        assert [member.filename for member in validate_archive_members(archive)] == [
            "cardToID.json",
            "part.parquet",
        ]


def test_archive_rejects_nested_member_paths(tmp_path: Path) -> None:
    path = tmp_path / "source.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("nested/part.parquet", b"parquet")
        archive.writestr("cardToID.json", b"{}")

    with ZipFile(path) as archive, pytest.raises(KaggleV6AuditError, match="unsafe"):
        validate_archive_members(archive)


def test_card_mapping_must_match_versioned_catalog(tmp_path: Path) -> None:
    path = tmp_path / "cardToID.json"
    path.write_text(
        json.dumps({entry.source_name: entry.source_id for entry in KAGGLE_V6_CARDS.entries}),
        encoding="utf-8",
    )
    validate_card_mapping(path)

    path.write_text("{}", encoding="utf-8")
    with pytest.raises(KaggleV6AuditError, match="versioned card catalog"):
        validate_card_mapping(path)


def test_profiles_combine_in_deterministic_order() -> None:
    profiles = [
        ParquetProfile(
            row_count=2,
            timestamp_min=datetime(2026, 6, 22, tzinfo=UTC),
            timestamp_max=datetime(2026, 6, 23, tzinfo=UTC),
            mode_counts=Counter({"Ranked": 2}),
            card_id_min=1,
            card_id_max=10,
        ),
        ParquetProfile(
            row_count=1,
            timestamp_min=datetime(2026, 6, 21, tzinfo=UTC),
            timestamp_max=datetime(2026, 6, 21, tzinfo=UTC),
            mode_counts=Counter({"Ladder": 1}),
            card_id_min=0,
            card_id_max=15,
        ),
    ]

    observations = combine_profiles(profiles)

    assert observations.row_count == 3
    assert observations.modes == ("Ladder", "Ranked")
    assert observations.timestamp_min == datetime(2026, 6, 21, tzinfo=UTC)
    assert observations.card_id_min == 0
    assert observations.card_id_max == 15
