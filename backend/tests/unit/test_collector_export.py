"""Export retained matches through the official v2 snapshot contract."""

from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_collector_normalize import battle

from clash_sos.application.attention_dataset_prepare import prepare_official_attention_dataset
from clash_sos.application.collector_export import export_collected_matches
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.attention_dataset import AttentionDatasetManifest, TowerBattleRowV2
from clash_sos.domain.attention_schema import AttentionModelConfig, build_attention_schema
from clash_sos.infrastructure.clash_royale.catalog import (
    CURRENT_CARD_ATTRIBUTES,
    CURRENT_CARD_CATALOG,
    CURRENT_TOWER_CATALOG,
)
from clash_sos.infrastructure.clash_royale.collector_normalize import normalize_battle
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore

START = datetime(2026, 9, 1, tzinfo=UTC)
END = datetime(2026, 10, 1, tzinfo=UTC)


def test_export_publishes_current_mode_and_leaves_sqlite_intact(tmp_path: Path) -> None:
    store = CollectorStore(tmp_path / "collector.sqlite")
    days = (2, 4, 9, 10, 19, 25)
    try:
        for day in reversed(days):
            raw = battle()
            raw["battleTime"] = f"202609{day:02d}T120000.000Z"
            store.record_poll(
                "#ABC",
                [normalize_battle(raw, "#ABC")],
                Counter(),
                observed_at=END,
                next_due=END + timedelta(hours=1),
                log_length=1,
            )
        before = store.summary()
        source = tmp_path / "current.jsonl"
        result = export_collected_matches(
            store,
            source,
            dataset_version="current-tiny-v1",
            balance_era_id="2026-09",
            start=START,
            end=END,
        )
        rows = [
            TowerBattleRowV2.model_validate_json(line) for line in source.read_text().splitlines()
        ]
        assert (result.written, result.skipped_mode) == (6, 0)
        assert [row.timestamp.day for row in rows] == list(days)
        assert all(row.mode == "Ranked1v1_NewArena2" for row in rows)
        assert all(row.side_a_player_id.value == "#ABC" for row in rows)
        assert [row.row_number for row in rows] == list(range(6))
        schema = build_attention_schema(
            CURRENT_CARD_CATALOG.serialize(),
            attributes=CURRENT_CARD_ATTRIBUTES,
            tower_catalog=CURRENT_TOWER_CATALOG,
            network=AttentionModelConfig(),
            balance_era_id="2026-09",
            official_schema_version="official-ranked16-schema:v2",
        )
        published = prepare_official_attention_dataset(
            source,
            tmp_path / "published",
            schema=schema,
            dataset_version="current-tiny-v1",
            start=START,
            train_end=datetime(2026, 9, 10, tzinfo=UTC),
            validation_end=datetime(2026, 9, 20, tzinfo=UTC),
            end=END,
            watch_fraction=0.3,
            config=StagingConfig(batch_rows=2, memory_limit="256MB"),
        )
        manifest = AttentionDatasetManifest.model_validate_json(
            (published / "manifest.json").read_bytes()
        )
        assert manifest.partition_counts == {"train": 3, "validation": 2, "test": 1}
        assert manifest.canonical_schema_version == "official-ranked16-schema:v2"
        assert store.summary() == before
        with pytest.raises(ValueError, match="already exists"):
            export_collected_matches(
                store,
                source,
                dataset_version="current-tiny-v1",
                balance_era_id="2026-09",
                start=START,
                end=END,
            )
    finally:
        store.close()


def test_export_excludes_collisions_and_ineligible_matches(tmp_path: Path) -> None:
    store = CollectorStore(tmp_path / "collector.sqlite")
    original = battle()
    changed = deepcopy(original)
    changed["opponent"][0]["crowns"] = 4
    old_mode = deepcopy(original)
    old_mode["battleTime"] = "20260928T120000.000Z"
    old_mode["gameMode"]["name"] = "Ranked1v1_NewArena"
    try:
        store.record_poll(
            "#ABC",
            [normalize_battle(original, "#ABC"), normalize_battle(old_mode, "#ABC")],
            Counter(),
            observed_at=END,
            next_due=END,
            log_length=2,
        )
        store.record_poll(
            "#DEF",
            [normalize_battle(changed, "#DEF")],
            Counter(),
            observed_at=END,
            next_due=END,
            log_length=1,
        )
        destination = tmp_path / "empty.jsonl"
        with pytest.raises(ValueError, match="no eligible current-ranked"):
            export_collected_matches(
                store,
                destination,
                dataset_version="tiny",
                balance_era_id="2026-09",
                start=START,
                end=END,
            )
        assert not destination.exists()
        assert store.summary()["matches_conflicted"] == 1
        assert store.summary()["variants"] == 3
    finally:
        store.close()
