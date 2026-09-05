import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile

import duckdb
import pytest
from pydantic import ValidationError

from clash_sos.application.dataset_staging import (
    StagingConfig,
    adapted_to_staged_row,
    adapted_to_unadaptable_row,
    ensure_raw_audit_manifest,
)
from clash_sos.domain.canonical import BattleOutcome, RecordIssue, RecordState
from clash_sos.domain.canonical_dataset import deck_content_hash
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
from clash_sos.domain.staged_dataset import StagedBattleRow, UnadaptableRow
from clash_sos.infrastructure.kaggle_v6.adapter import SourceRowLocation, adapt_row
from clash_sos.infrastructure.kaggle_v6.audit import write_dataset_manifest
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.balance_eras import KAGGLE_V6_ERA_REGISTRY
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_SOURCE_ID
from clash_sos.infrastructure.kaggle_v6.staging_io import (
    KaggleV6StagingError,
    connect_staging_duckdb,
    list_parquet_members,
    part_path,
    python_cell,
    write_staged_part,
    write_unadaptable_part,
)


class Int64(int):
    pass


def valid_row() -> dict[str, object]:
    row: dict[str, object] = {
        "winner_id": "#winner",
        "loser_id": "#loser",
        "time": datetime(2026, 6, 21, 8, tzinfo=timezone(timedelta(hours=-4))),
        "game_mode": "Ranked1v1_NewArena",
    }
    for side, offset in (("winner", 0), ("loser", 8)):
        for index in range(8):
            row[f"{side}_card_{index}"] = offset + index
            row[f"{side}_card_{index}_level"] = 16
    return row


def location() -> SourceRowLocation:
    return SourceRowLocation(archive_member="part.parquet", row_number=12)


def staged_row(**overrides: object) -> StagedBattleRow:
    card_ids = tuple(f"card-{index}" for index in range(8))
    card_forms = ("base",) * 8
    card_levels = (16,) * 8
    payload: dict[str, object] = {
        "source_id": KAGGLE_V6_SOURCE_ID,
        "timestamp": datetime(2026, 6, 21, 12, tzinfo=UTC),
        "mode": "Ranked1v1_NewArena",
        "balance_era_id": "2026-06",
        "outcome": BattleOutcome.SIDE_A_WIN,
        "event_key": "a" * 64,
        "fingerprint": "a" * 64,
        "side_a_player_id": "#AAA",
        "side_b_player_id": "#BBB",
        "side_a_card_ids": card_ids,
        "side_a_card_forms": card_forms,
        "side_a_card_levels": card_levels,
        "side_a_deck_hash": deck_content_hash(card_ids, card_forms, card_levels),
        "side_b_card_ids": card_ids,
        "side_b_card_forms": card_forms,
        "side_b_card_levels": card_levels,
        "side_b_deck_hash": deck_content_hash(card_ids, card_forms, card_levels),
        "observation_issues": (),
        "archive_member": "part.parquet",
        "row_number": 0,
    }
    payload.update(overrides)
    return StagedBattleRow.model_validate(payload)


def test_staging_config_rejects_non_positive_values() -> None:
    with pytest.raises(ValidationError):
        StagingConfig(threads=0)
    with pytest.raises(ValidationError):
        StagingConfig(batch_rows=0)
    with pytest.raises(ValidationError):
        StagingConfig(chunk_size=0)


def test_part_path_uses_member_stem_and_zero_padded_index() -> None:
    assert part_path(Path("/ws"), "staging", "2026-06-21_0.parquet", 0) == Path(
        "/ws/staging/2026-06-21_0/part-00000.parquet"
    )


def test_python_cell_coerces_naive_datetime_and_integral_subclasses() -> None:
    naive = datetime(2026, 6, 21, 12)
    coerced = python_cell(naive)
    assert isinstance(coerced, datetime)
    assert coerced.tzinfo is UTC
    assert python_cell(True) is True
    integral = python_cell(Int64(7))
    assert type(integral) is int
    assert integral == 7


def test_list_parquet_members_returns_sorted_names(tmp_path: Path) -> None:
    mapping = tmp_path / "cardToID.json"
    mapping.write_text(
        json.dumps({entry.source_name: entry.source_id for entry in KAGGLE_V6_CARDS.entries}),
        encoding="utf-8",
    )
    first = tmp_path / "z.parquet"
    second = tmp_path / "a.parquet"
    first.write_bytes(b"parquet")
    second.write_bytes(b"parquet")
    archive_path = tmp_path / "source.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.write(second, "a.parquet")
        archive.write(first, "z.parquet")
        archive.write(mapping, "cardToID.json")
    with ZipFile(archive_path) as archive:
        assert list_parquet_members(archive) == ("a.parquet", "z.parquet")


def test_connect_staging_duckdb_uses_utc(tmp_path: Path) -> None:
    connection = connect_staging_duckdb(memory_limit="256MB", threads=1, temp_directory=tmp_path)
    try:
        timezone_name = connection.execute("SELECT current_setting('TimeZone')").fetchone()
        assert timezone_name == ("UTC",)
    finally:
        connection.close()


def test_write_staged_and_unadaptable_parts_round_trip(tmp_path: Path) -> None:
    from clash_sos.domain.canonical import BattleOutcome
    from clash_sos.domain.canonical_dataset import deck_content_hash

    card_ids = tuple(f"card-{index}" for index in range(8))
    card_forms = ("base",) * 8
    card_levels = (16,) * 8
    deck_hash = deck_content_hash(card_ids, card_forms, card_levels)
    sha = "a" * 64
    rows = (
        StagedBattleRow.model_validate(
            {
                "source_id": KAGGLE_V6_SOURCE_ID,
                "timestamp": datetime(2026, 6, 21, 12, tzinfo=UTC),
                "mode": "Ranked1v1_NewArena",
                "balance_era_id": "2026-06",
                "outcome": BattleOutcome.SIDE_A_WIN,
                "event_key": sha,
                "fingerprint": sha,
                "side_a_player_id": "#AAA",
                "side_b_player_id": "#BBB",
                "side_a_card_ids": card_ids,
                "side_a_card_forms": card_forms,
                "side_a_card_levels": card_levels,
                "side_a_deck_hash": deck_hash,
                "side_b_card_ids": card_ids,
                "side_b_card_forms": card_forms,
                "side_b_card_levels": card_levels,
                "side_b_deck_hash": deck_hash,
                "observation_issues": (),
                "archive_member": "part.parquet",
                "row_number": 0,
            }
        ),
        StagedBattleRow.model_validate(
            {
                "source_id": KAGGLE_V6_SOURCE_ID,
                "timestamp": datetime(2026, 6, 22, 12, tzinfo=UTC),
                "mode": "Ladder",
                "balance_era_id": None,
                "outcome": BattleOutcome.SIDE_A_WIN,
                "event_key": sha,
                "fingerprint": sha,
                "side_a_player_id": "#AAA",
                "side_b_player_id": "#CCC",
                "side_a_card_ids": card_ids,
                "side_a_card_forms": card_forms,
                "side_a_card_levels": card_levels,
                "side_a_deck_hash": deck_hash,
                "side_b_card_ids": card_ids,
                "side_b_card_forms": card_forms,
                "side_b_card_levels": card_levels,
                "side_b_deck_hash": deck_hash,
                "observation_issues": (RecordIssue.UNSUPPORTED_MODE,),
                "archive_member": "part.parquet",
                "row_number": 1,
            }
        ),
    )
    staged_path = tmp_path / "staging.parquet"
    write_staged_part(staged_path, rows, row_group_rows=131072)
    connection = duckdb.connect()
    try:
        result = connection.execute(
            "SELECT row_number, mode FROM read_parquet(?) ORDER BY row_number",
            [str(staged_path)],
        ).fetchall()
        assert result == [(0, "Ranked1v1_NewArena"), (1, "Ladder")]
    finally:
        connection.close()

    unadaptable_rows = (
        UnadaptableRow(
            archive_member="part.parquet",
            row_number=2,
            state=RecordState.INVALID,
            issues=(RecordIssue.MALFORMED_TIMESTAMP,),
        ),
    )
    unadaptable_path = tmp_path / "unadaptable.parquet"
    write_unadaptable_part(unadaptable_path, unadaptable_rows, row_group_rows=131072)
    connection = duckdb.connect()
    try:
        columns = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(unadaptable_path)]
        ).fetchall()
        assert [str(row[0]) for row in columns] == [
            "archive_member",
            "row_number",
            "state",
            "issues",
        ]
    finally:
        connection.close()

    with pytest.raises(KaggleV6StagingError, match="empty"):
        write_staged_part(staged_path, [], row_group_rows=131072)
    with pytest.raises(KaggleV6StagingError, match="empty"):
        write_unadaptable_part(unadaptable_path, [], row_group_rows=131072)


def test_write_staged_part_keeps_era_after_leading_nulls(tmp_path: Path) -> None:
    rows = [
        staged_row(
            balance_era_id=None,
            observation_issues=(),
            side_b_player_id=f"#B{index}",
            row_number=index,
        )
        for index in range(100)
    ]
    rows.append(
        staged_row(
            mode="Ladder",
            balance_era_id="2026-06",
            observation_issues=(RecordIssue.UNSUPPORTED_MODE,),
            side_b_player_id="#ERA",
            row_number=100,
        )
    )
    path = tmp_path / "staged.parquet"
    write_staged_part(path, rows, row_group_rows=131072)
    connection = duckdb.connect()
    try:
        era_row = connection.execute(
            "SELECT balance_era_id, observation_issues FROM read_parquet(?) WHERE row_number = 100",
            [str(path)],
        ).fetchone()
        null_count = connection.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE balance_era_id IS NULL",
            [str(path)],
        ).fetchone()
    finally:
        connection.close()
    assert era_row is not None
    assert era_row[0] == "2026-06"
    assert list(era_row[1]) == ["unsupported_mode"]
    assert null_count == (100,)


def test_adapted_to_staged_row_maps_population_observations() -> None:
    ranked = adapt_row(valid_row(), location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)
    staged = adapted_to_staged_row(ranked)
    assert staged.balance_era_id == "2026-06"
    assert staged.observation_issues == ()

    ladder_row = valid_row()
    ladder_row["game_mode"] = "Ladder"
    ladder = adapt_row(ladder_row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)
    ladder_staged = adapted_to_staged_row(ladder)
    assert ladder_staged.observation_issues == (RecordIssue.UNSUPPORTED_MODE,)


def test_adapted_to_unadaptable_row_omits_detail() -> None:
    row = valid_row()
    row["time"] = None
    record = adapt_row(row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)
    unadaptable = adapted_to_unadaptable_row(record)
    assert unadaptable.state is RecordState.INVALID
    assert unadaptable.issues == (RecordIssue.MALFORMED_TIMESTAMP,)
    assert "detail" not in unadaptable.model_dump(mode="python")


def test_ensure_raw_audit_manifest_reuses_matching_file(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(b"tiny-archive")
    size, digest = hash_file(archive_path, 1024)
    manifest_path = tmp_path / "manifest.json"
    manifest = DatasetManifest(
        source_id=KAGGLE_V6_SOURCE_ID,
        archive=ArchiveManifest(path="source.zip", size_bytes=size, sha256=digest),
        files=(
            DatasetFileManifest(
                path="battles.parquet",
                kind="parquet",
                size_bytes=5,
                sha256="a" * 64,
            ),
        ),
        schema=DatasetSchemaManifest(
            format="parquet",
            columns=(
                SchemaColumnManifest(
                    name="timestamp",
                    physical_type="timestamp[us]",
                    nullable=False,
                ),
            ),
            fingerprint="a" * 64,
        ),
        observations=DatasetObservationsManifest(
            row_count=1,
            timestamp_column="timestamp",
            timestamp_min=datetime(2026, 6, 21, tzinfo=UTC),
            timestamp_max=datetime(2026, 6, 21, tzinfo=UTC),
            modes=("Ranked1v1_NewArena",),
            mode_counts=(ModeCountManifest(mode="Ranked1v1_NewArena", row_count=1),),
            card_id_min=0,
            card_id_max=175,
        ),
        validation=DatasetValidationManifest(status="not_run"),
    )
    write_dataset_manifest(manifest, manifest_path)
    config = StagingConfig()
    with patch("clash_sos.application.dataset_staging.audit_kaggle_v6_archive") as audit:
        restored = ensure_raw_audit_manifest(
            archive_path,
            manifest_path,
            temp_directory=tmp_path / "work",
            config=config,
            expected_archive_size=None,
            expected_archive_sha256=None,
        )
    audit.assert_not_called()
    assert restored.archive.sha256 == digest


def test_ensure_raw_audit_manifest_rejects_mismatched_hash(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(b"tiny-archive")
    size, _ = hash_file(archive_path, 1024)
    manifest_path = tmp_path / "manifest.json"
    manifest = DatasetManifest(
        source_id=KAGGLE_V6_SOURCE_ID,
        archive=ArchiveManifest(path="source.zip", size_bytes=size, sha256="b" * 64),
        files=(
            DatasetFileManifest(
                path="battles.parquet",
                kind="parquet",
                size_bytes=5,
                sha256="a" * 64,
            ),
        ),
        schema=DatasetSchemaManifest(
            format="parquet",
            columns=(
                SchemaColumnManifest(
                    name="timestamp",
                    physical_type="timestamp[us]",
                    nullable=False,
                ),
            ),
            fingerprint="a" * 64,
        ),
        observations=DatasetObservationsManifest(
            row_count=1,
            timestamp_column="timestamp",
            timestamp_min=datetime(2026, 6, 21, tzinfo=UTC),
            timestamp_max=datetime(2026, 6, 21, tzinfo=UTC),
            modes=("Ranked1v1_NewArena",),
            mode_counts=(ModeCountManifest(mode="Ranked1v1_NewArena", row_count=1),),
            card_id_min=0,
            card_id_max=175,
        ),
        validation=DatasetValidationManifest(status="not_run"),
    )
    write_dataset_manifest(manifest, manifest_path)
    with pytest.raises(KaggleV6StagingError, match="does not match archive"):
        ensure_raw_audit_manifest(
            archive_path,
            manifest_path,
            temp_directory=tmp_path / "work",
            config=StagingConfig(),
            expected_archive_size=None,
            expected_archive_sha256=None,
        )
