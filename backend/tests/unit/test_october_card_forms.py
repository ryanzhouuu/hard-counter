"""October forms remain distinct through collection and dataset export."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from test_collector_normalize import battle
from test_collector_run import FakeClock, config

from clash_sos.application.collector_export import export_collected_matches
from clash_sos.application.collector_run import collect_for_duration
from clash_sos.domain.attention_dataset import TowerBattleRowV3
from clash_sos.infrastructure.clash_royale.adapter import adapt_battle
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_CARD_CATALOG
from clash_sos.infrastructure.clash_royale.client import RoyaleClient
from clash_sos.infrastructure.clash_royale.collector_normalize import (
    BattleRejected,
    normalize_battle,
)
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def test_current_catalog_preserves_source_ids_and_appends_october_forms() -> None:
    assert CURRENT_CARD_CATALOG.version == "official-cards:2026-10-12"
    assert CURRENT_CARD_CATALOG.entries[:176] == KAGGLE_V6_CARDS.entries
    assert [
        (entry.source_id, entry.card.identity_key) for entry in CURRENT_CARD_CATALOG.entries[176:]
    ] == [
        (176, "ronin:base"),
        (177, "minion-giant:base"),
        (178, "elite-barbarians:evolution"),
        (179, "valkyrie:hero"),
        (180, "berserker:hero"),
        (181, "ice-wizard:hero"),
        (182, "electro-giant:evolution"),
        (183, "electro-wizard:hero"),
    ]


@pytest.mark.parametrize(
    ("name", "api_id", "max_level", "markers", "identity"),
    [
        ("Electro Giant", 26000085, 11, {}, "electro-giant:base"),
        ("Electro Giant", 26000085, 11, {"evolutionLevel": 1}, "electro-giant:evolution"),
        ("Evo Electro Giant", 26000085, 11, {"evolutionLevel": 1}, "electro-giant:evolution"),
        ("Electro Wizard", 26000042, 8, {}, "electro-wizard:base"),
        ("Electro Wizard", 26000042, 8, {"evolutionLevel": 2}, "electro-wizard:hero"),
        ("Electro Wizard", 26000042, 8, {"heroLevel": 1}, "electro-wizard:hero"),
        ("Hero Electro Wizard", 26000042, 8, {"heroLevel": 1}, "electro-wizard:hero"),
    ],
)
def test_october_forms_normalize_without_base_substitution(
    name: str, api_id: int, max_level: int, markers: dict[str, int], identity: str
) -> None:
    raw = battle()
    raw["team"][0]["cards"][0] = {
        "name": name,
        "id": api_id,
        "level": max_level,
        "maxLevel": max_level,
        **markers,
    }
    adapted = adapt_battle(raw, "#ABC")
    assert adapted.skip_reason is None
    assert adapted.player_deck[0] == identity
    normalized = normalize_battle(raw, "#ABC")
    assert normalized.exclusion_reason is None
    assert (identity, 16) in normalized.sides[0].cards


@pytest.mark.parametrize(
    ("name", "form"),
    [("Electro Giant", 2), ("Electro Giant", 3), ("Electro Wizard", 1), ("Electro Wizard", 3)],
)
def test_uncataloged_forms_still_reject_without_base_substitution(name: str, form: int) -> None:
    raw = battle()
    raw["team"][0]["cards"][0].update(name=name, evolutionLevel=form)
    with pytest.raises(BattleRejected, match="unknown_card"):
        normalize_battle(raw, "#ABC")


def test_collector_persists_and_exports_both_october_forms(tmp_path: Path) -> None:
    raw = battle()
    raw["team"][0]["cards"][0] = {
        "name": "Electro Giant",
        "id": 26000085,
        "level": 11,
        "maxLevel": 11,
        "evolutionLevel": 1,
    }
    raw["opponent"][0]["cards"][0] = {
        "name": "Electro Wizard",
        "id": 26000042,
        "level": 8,
        "maxLevel": 8,
        "evolutionLevel": 2,
    }
    clock = FakeClock()

    async def campaign(store: CollectorStore) -> dict[str, int]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[raw])),
            base_url="https://api.clashroyale.com/v1/",
        ) as http:
            return await collect_for_duration(
                RoyaleClient(http, "test-token"),
                store,
                ("#ABC",),
                config(1),
                now=clock.now,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

    database = tmp_path / "collector.sqlite"
    store = CollectorStore(database)
    try:
        summary = asyncio.run(campaign(store))
        assert summary["matches_eligible"] == 1
        assert summary.get("rejected_unknown_card", 0) == 0
    finally:
        store.close()
    store = CollectorStore(database)
    try:
        destination = tmp_path / "export.jsonl"
        result = export_collected_matches(
            store,
            destination,
            dataset_version="october-forms",
            balance_era_id="test",
            start=datetime(2026, 9, 1, tzinfo=UTC),
            end=datetime(2026, 10, 1, tzinfo=UTC),
        )
        assert result.written == 1
        row = TowerBattleRowV3.model_validate_json(destination.read_text())
        assert ("electro-giant", "evolution", 16) in zip(
            row.side_a_card_ids, row.side_a_card_forms, row.side_a_card_levels, strict=True
        )
        assert ("electro-wizard", "hero", 16) in zip(
            row.side_b_card_ids, row.side_b_card_forms, row.side_b_card_levels, strict=True
        )
    finally:
        store.close()
