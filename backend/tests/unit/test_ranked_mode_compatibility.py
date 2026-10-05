"""Preserve both ranked API names through export, caches, and research loading."""

from collections import Counter
from json import loads
from pathlib import Path

import polars as pl
import pytest
from experiments.common.artifacts import file_record
from experiments.common.data_access import ReportingContract
from experiments.common.prospective_io import load_reporting_jsonl
from experiments.common.protocol import search_protocol
from experiments.common.source_io import load_snapshot
from experiments.common.source_rows import encode_official_row, population_identity
from test_collector_normalize import battle
from test_tower_attention import tower_schema
from tower_dataset_fixture import VERSION, stamp, tower_rows, write_tower_source

from clash_sos.application.attention_dataset_prepare import prepare_official_attention_dataset
from clash_sos.application.collector_export import export_collected_matches
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.attention_dataset import TowerBattleRow, official_row_type
from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.infrastructure.clash_royale.collector_normalize import normalize_battle
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore


def mixed_schema() -> AttentionCardSchema:
    return AttentionCardSchema.model_validate(
        {**tower_schema().model_dump(), "canonical_schema_version": "official-ranked16-schema:v3"}
    )


def test_export_preserves_both_ranked_names_without_changing_collection(tmp_path: Path) -> None:
    store = CollectorStore(tmp_path / "collector.sqlite")
    try:
        for day, mode in ((2, "Ranked1v1_NewArena2"), (4, "Ranked1v1_NewArena")):
            raw = battle()
            raw["battleTime"] = f"202609{day:02d}T120000.000Z"
            raw["gameMode"]["name"] = mode
            store.record_poll(
                "#ABC",
                [normalize_battle(raw, "#ABC")],
                Counter(),
                observed_at=stamp(27),
                next_due=stamp(28),
                log_length=1,
            )
        before = store.summary()
        source = tmp_path / "mixed.jsonl"
        result = export_collected_matches(
            store,
            source,
            dataset_version=VERSION,
            balance_era_id="2026-09",
            start=stamp(1),
            end=stamp(27),
        )
        rows = [loads(line) for line in source.read_text().splitlines()]
        assert (result.written, result.skipped_mode) == (2, 0)
        assert [row["mode"] for row in rows] == ["Ranked1v1_NewArena2", "Ranked1v1_NewArena"]
        assert [row["row_number"] for row in rows] == [0, 1]
        assert store.summary() == before
    finally:
        store.close()


@pytest.mark.parametrize("future_mode", ["Ranked1v1_NewArena", "Ranked1v1_NewArena2"])
def test_mixed_snapshot_preserves_modes_through_cache_and_research(
    tmp_path: Path,
    future_mode: str,
) -> None:
    source = tmp_path / "mixed.jsonl"
    originals = tuple(
        row.model_copy(update={"mode": "Ranked1v1_NewArena2"}) if index % 2 else row
        for index, row in enumerate(tower_rows())
    )
    write_tower_source(source, originals)
    dataset = prepare_official_attention_dataset(
        source,
        tmp_path / "dataset",
        schema=mixed_schema(),
        dataset_version=VERSION,
        start=stamp(1),
        train_end=stamp(10),
        validation_end=stamp(20),
        end=stamp(27),
        watch_fraction=0.3,
        config=StagingConfig(batch_rows=2, memory_limit="256MB"),
    )
    canonical = pl.read_parquet(dataset / "canonical.parquet")
    assert canonical.get_column("mode").to_list() == [row.mode for row in originals]
    schema = AttentionCardSchema.model_validate_json((dataset / "feature-schema.json").read_bytes())
    original = AttentionProtocol.model_validate_json((dataset / "protocol.json").read_bytes())
    assert schema.canonical_schema_version == "official-ranked16-schema:v3"
    resolved = search_protocol(original, dataset, calibration_end=stamp(11))
    protocol = tmp_path / "search.json"
    protocol.write_text(resolved.model_dump_json())
    access, population, _ = load_snapshot(
        dataset,
        protocol,
        dataset / "feature-schema.json",
        tmp_path / "cache",
        row_cap=5,
    )
    assert population.mode == "pathOfLegend"
    assert access.read("refit", "fit") == tuple(
        encode_official_row(row, schema, original.mirror_seed) for row in originals[:3]
    )
    future = originals[-1].model_copy(update={"timestamp": stamp(28), "mode": future_mode})
    reporting = tmp_path / "future.jsonl"
    reporting.write_text(future.model_dump_json() + "\n")
    future_population = population_identity(
        (encode_official_row(future, schema, 0),),
        (file_record(reporting, tmp_path, row_count=1),),
        schema,
        0,
        start=stamp(28),
        end=stamp(29),
    )
    contract = ReportingContract(future_population, population, "b" * 64, stamp(27), stamp(27))
    assert (
        len(
            load_reporting_jsonl(reporting, schema, contract, row_cap=1).read("reporting", "report")
        )
        == 1
    )


@pytest.mark.parametrize("version", ["official-ranked16-schema:v1", "official-ranked16-schema:v2"])
def test_legacy_research_schemas_still_reject_the_other_ranked_name(version: str) -> None:
    schema = AttentionCardSchema.model_validate(
        {**tower_schema().model_dump(), "canonical_schema_version": version}
    )
    other = "Ranked1v1_NewArena2" if version.endswith("v1") else "Ranked1v1_NewArena"
    row = tower_rows()[0].model_copy(update={"mode": other})
    with pytest.raises(ValueError, match="mode or era"):
        encode_official_row(row, schema, 0)


def test_mixed_research_schema_still_checks_mode_and_era() -> None:
    schema = mixed_schema()
    for changes in ({"mode": "Ladder"}, {"balance_era_id": "other"}):
        row: TowerBattleRow = tower_rows()[0].model_copy(update=changes)
        with pytest.raises(ValueError, match="mode or era"):
            encode_official_row(row, schema, 0)


@pytest.mark.parametrize(
    "changes",
    [{"mode": "Ladder"}, {"source_id": "kaggle-v6"}, {"side_a_tower_level": 15}],
)
def test_mixed_snapshot_contract_rejects_non_ranked_sources_and_wrong_levels(
    changes: dict[str, object],
) -> None:
    payload = {**tower_rows()[0].model_dump(), **changes}
    with pytest.raises(ValueError):
        official_row_type("official-ranked16-schema:v3").model_validate(payload)
