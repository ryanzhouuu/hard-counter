from pathlib import Path

import pytest
from experiments.common.artifacts import file_record
from experiments.common.data_access import ReportingContract
from experiments.common.prospective_io import load_reporting_jsonl
from experiments.common.source_rows import encode_official_row, population_identity
from test_tower_attention import tower_schema
from tower_dataset_fixture import stamp, tower_rows

from clash_sos.domain.attention_dataset import TowerBattleRowV2
from clash_sos.domain.attention_schema import AttentionCardSchema


def reporting_source(tmp_path: Path) -> tuple[Path, AttentionCardSchema, ReportingContract]:
    schema = AttentionCardSchema.model_validate(
        tower_schema()
        .model_copy(update={"canonical_schema_version": "official-ranked16-schema:v2"})
        .model_dump()
    )
    source_rows = tuple(
        TowerBattleRowV2.model_validate(
            {**row.model_dump(mode="python"), "mode": "Ranked1v1_NewArena2"}
        )
        for row in tower_rows()
    )
    future_row = source_rows[-1].model_copy(update={"timestamp": stamp(28)})
    path = tmp_path / "future.jsonl"
    path.write_text(future_row.model_dump_json() + "\n")
    future = population_identity(
        (encode_official_row(future_row, schema, 0),),
        (file_record(path, tmp_path, row_count=1),),
        schema,
        0,
        start=stamp(28),
        end=stamp(29),
    )
    fit_rows = tuple(encode_official_row(row, schema, 0) for row in source_rows[:-1])
    fit = population_identity(
        fit_rows,
        (
            future.snapshot_files[0].model_copy(
                update={
                    "path": "fit.jsonl",
                    "sha256": "a" * 64,
                }
            ),
        ),
        schema,
        0,
        start=stamp(1),
        end=stamp(20),
    )
    contract = ReportingContract(future, fit, "b" * 64, stamp(27), stamp(27))
    return path, schema, contract


def test_future_jsonl_uses_frozen_orientation_and_never_constructs_fit_rows(tmp_path: Path) -> None:
    path, schema, contract = reporting_source(tmp_path)
    access = load_reporting_jsonl(path, schema, contract, row_cap=1)
    rows = access.read("reporting", "report")
    assert len(rows) == 1
    assert rows[0].key[0] == stamp(28)
    assert len(rows[0].tokens[0]) == 9
    assert access.contract == contract
    assert not (tmp_path / "train").exists()


def test_prospective_rejects_changed_bytes_encoding_and_caps(tmp_path: Path) -> None:
    path, schema, contract = reporting_source(tmp_path)
    with pytest.raises(ValueError, match="positive row cap"):
        load_reporting_jsonl(path, schema, contract, row_cap=0)
    wrong_catalog = schema.model_copy(update={"catalog_sha256": "c" * 64})
    with pytest.raises(ValueError, match="inventory or frozen schema"):
        load_reporting_jsonl(path, wrong_catalog, contract, row_cap=1)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source bytes"):
        load_reporting_jsonl(path, schema, contract, row_cap=1)


def test_prospective_rejects_earlier_battles_and_duplicate_events(tmp_path: Path) -> None:
    path, schema, original = reporting_source(tmp_path)
    future = TowerBattleRowV2.model_validate_json(path.read_bytes())
    for rows in ((future, future), (future.model_copy(update={"timestamp": stamp(26)}),)):
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
        files = (file_record(path, tmp_path, row_count=len(rows)),)
        population = original.population.model_copy(update={"snapshot_files": files})
        contract = ReportingContract(
            population, original.fit_population, "b" * 64, stamp(27), stamp(27)
        )
        with pytest.raises(ValueError):
            load_reporting_jsonl(path, schema, contract, row_cap=2)


def test_prospective_source_count_and_cap_are_explicit(tmp_path: Path) -> None:
    path, schema, original = reporting_source(tmp_path)
    future = TowerBattleRowV2.model_validate_json(path.read_bytes())
    second = future.model_copy(
        update={"row_number": 99, "event_key": "d" * 64, "fingerprint": "e" * 64}
    )
    path.write_text(future.model_dump_json() + "\n" + second.model_dump_json() + "\n")
    population = original.population.model_copy(
        update={
            "snapshot_files": (file_record(path, tmp_path, row_count=2),),
        }
    )
    contract = ReportingContract(
        population, original.fit_population, "b" * 64, stamp(27), stamp(27)
    )
    with pytest.raises(ValueError, match="exceeds row cap"):
        load_reporting_jsonl(path, schema, contract, row_cap=1)
