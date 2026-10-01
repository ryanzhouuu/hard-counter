"""Reviewed capabilities stay sourced, partial, and separate from frozen inputs."""

from hashlib import sha256
from pathlib import Path

import pytest
from experiments.matchup_features.responses import answers, fixed_cost
from experiments.mechanics.load import load, load_partial_catalog

from clash_sos.infrastructure.clash_royale.catalog import (
    CURRENT_CARD_CATALOG,
    CURRENT_TOWER_CATALOG,
)


@pytest.mark.parametrize(
    "identity",
    [
        "furnace:base",
        "goblin-hut:base",
        "zappies:base",
        "mother-witch:base",
        "royal-delivery:base",
        "goblin-curse:base",
    ],
)
def test_reviewed_air_channels_have_verified_evidence(identity: str) -> None:
    catalog = load_partial_catalog()
    entry = next(e for e in catalog.entries.values() if e.identity == identity)
    assert entry.field("targets_air").status == "verified"
    assert entry.flag("targets_air")
    assert answers(entry, "airborne")
    assert entry.field("spell").status == "verified"
    assert entry.field("damage_per_hit").status == "unknown"


def test_review_preserves_unknowns_forms_and_conditional_costs() -> None:
    catalog = load_partial_catalog()
    assert catalog.version == "mechanics-partial:2026-10-01-r4"
    entries = {e.identity: e for e in catalog.entries.values()}
    identities = {entry.card.identity_key for entry in CURRENT_CARD_CATALOG.entries}
    identities.update(entry.identity for entry in CURRENT_TOWER_CATALOG.entries)
    assert set(entries) == identities
    assert not entries["furnace:base"].flag("defensive_building")
    assert entries["furnace:base"].field("defensive_building").known
    assert entries["goblin-hut:base"].flag("defensive_building")
    assert answers(entries["royal-delivery:base"], "multi_unit")
    assert not answers(entries["mother-witch:base"], "multi_unit")
    assert entries["furnace:base"].field("area_damage").status == "unknown"
    assert not answers(entries["furnace:base"], "building_targeting")
    assert entries["furnace:evolution"].flag("targets_air")
    assert entries["furnace:evolution"].flag("spawns_units")
    assert entries["spirit-empress:base"].field("airborne").status == "unknown"
    assert fixed_cost(entries["spirit-empress:base"]) is None
    assert entries["minion-giant:base"].flag("airborne")
    assert entries["minion-giant:base"].flag("building_targeting")
    assert entries["ice-wizard:hero"].flag("conditional")
    assert entries["ice-wizard:hero"].flag("targets_air")
    assert entries["rage:base"].flag("targets_air")


def test_september_snapshot_bytes_and_unknowns_remain_frozen() -> None:
    path = Path("experiments/mechanics/inputs/2026-09-30-partial.json")
    assert sha256(path.read_bytes()).hexdigest() == (
        "f87c16176271fe36e75b39ee73197ea37cc3a1e1a56e322433cc52500ca04c85"
    )
    previous = load(path)
    assert previous.version == "official-static-partial:2026-09-30"
    entry = next(e for e in previous.entries.values() if e.identity == "furnace:base")
    assert entry.field("defensive_building").status == "unknown"
    assert previous.digest != load_partial_catalog().digest
