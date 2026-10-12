from hashlib import sha256
from pathlib import Path

import pytest
from experiments.matchup_features.features import extract
from experiments.matchup_features.responses import answers, summarize
from experiments.mechanics.load import load, load_partial_catalog

PREVIOUS = Path("experiments/mechanics/inputs/2026-10-05-partial-r9.json")
CURRENT = Path("experiments/mechanics/inputs/2026-10-12-partial-r10.json")


def test_swarm_splash_revision_preserves_frozen_evidence_and_era() -> None:
    assert sha256(PREVIOUS.read_bytes()).hexdigest() == (
        "2abbda8dba5852f9c9cffaf29a0b0fe0764410ad64db0aab5be4525916eba0b3"
    )
    previous = load(PREVIOUS)
    current = load_partial_catalog()
    assert current.version == "mechanics-partial:2026-10-12-r10"
    assert current.digest == load(CURRENT).digest != previous.digest
    assert (current.era_start, current.era_end) == (previous.era_start, previous.era_end)
    assert current.entries.keys() == previous.entries.keys()
    additions = 0
    for token, old in previous.entries.items():
        new = current.entries[token]
        assert (new.identity, new.kind, new.base_identity) == (
            old.identity,
            old.kind,
            old.base_identity,
        )
        assert all(new.fields[name] == field for name, field in old.fields.items())
        for name in new.fields.keys() - old.fields.keys():
            assert name in {"multi_unit", "area_damage"}
            field = new.field(name)
            assert field.source_url and field.source_effective_date == "2026-10-12"
            assert field.status in {"verified", "not_applicable"}
            additions += 1
    assert additions == 365


def test_swarm_splash_coverage_preserves_conditional_unknowns() -> None:
    catalog = load_partial_catalog()
    cards = [entry for entry in catalog.entries.values() if entry.kind == "card"]
    assert len(cards) == 182
    for field, positive, missing in (
        ("multi_unit", 33, {"mirror:base", "clone:base"}),
        ("area_damage", 71, {"mirror:base", "minion-giant:base"}),
    ):
        assert sum(entry.field(field).known for entry in cards) == 180
        assert sum(entry.flag(field) for entry in cards) == positive
        assert {entry.identity for entry in cards if not entry.field(field).known} == missing
    towers = [entry for entry in catalog.entries.values() if entry.kind == "tower"]
    for tower in towers:
        assert tower.field("multi_unit").status == "not_applicable"
        assert tower.field("area_damage").value is False
        assert tower.field("area_damage").known
        assert not answers(tower, "multi_unit")


@pytest.mark.parametrize(
    ("identity", "swarm", "splash"),
    [
        ("archers:base", True, False),
        ("skeleton-dragons:base", True, True),
        ("wall-breakers:base", True, True),
        ("goblin-barrel:base", True, False),
        ("goblinstein:champion", True, False),
        ("goblins:hero", True, False),
        ("witch:base", False, True),
        ("night-witch:base", False, False),
        ("graveyard:base", False, False),
        ("barbarian-hut:base", False, False),
        ("tombstone:base", False, False),
        ("suspicious-bush:base", False, False),
        ("skeleton-barrel:base", False, False),
        ("balloon:base", False, False),
        ("golem:base", False, False),
        ("giant-skeleton:base", False, False),
        ("goblin-giant:base", False, False),
        ("ram-rider:base", False, False),
        ("electro-wizard:base", False, True),
        ("goblin-drill:base", False, True),
        ("royal-delivery:base", False, True),
        ("hunter:base", False, False),
        ("electro-dragon:base", False, False),
        ("electro-spirit:base", False, False),
        ("electro-giant:base", False, False),
        ("lightning:base", False, False),
        ("vines:base", False, False),
        ("mighty-miner:champion", False, False),
        ("little-prince:champion", False, False),
        ("monk:champion", False, False),
        ("ice-golem:hero", False, False),
        ("giant:hero", False, False),
        ("balloon:hero", False, False),
        ("skeletons:evolution", True, False),
        ("skeleton-barrel:evolution", False, False),
        ("goblin-giant:evolution", False, False),
        ("furnace:evolution", False, False),
        ("hunter:evolution", False, False),
        ("electro-dragon:evolution", False, False),
        ("elite-barbarians:evolution", True, False),
        ("cannon:base", False, False),
        ("royal-giant:base", False, False),
        ("battle-ram:base", False, False),
        ("cannon:evolution", False, True),
        ("royal-giant:evolution", False, True),
        ("battle-ram:evolution", False, True),
        ("dart-goblin:evolution", False, True),
        ("tesla:evolution", False, True),
        ("royal-hogs:evolution", True, True),
    ],
)
def test_ordinary_deployment_and_direct_area_semantics(
    identity: str, swarm: bool, splash: bool
) -> None:
    catalog = load_partial_catalog()
    entry = next(entry for entry in catalog.entries.values() if entry.identity == identity)
    assert entry.field("multi_unit").value is swarm
    assert entry.field("area_damage").value is splash
    assert entry.field("multi_unit").known and entry.field("area_damage").known


def test_real_swarm_response_counts_costs_and_swap_invariance() -> None:
    catalog = load_partial_catalog()
    tokens = {entry.identity: token for token, entry in catalog.entries.items()}
    identities = (
        (
            "goblins:base",
            "archers:base",
            "minions:base",
            "skeletons:base",
            "giant-skeleton:base",
            "lumberjack:base",
            "giant:base",
            "electro-dragon:base",
            "tower-princess:tower",
        ),
        (
            "witch:base",
            "valkyrie:base",
            "electro-wizard:base",
            "fireball:base",
            "lightning:base",
            "furnace:base",
            "hunter:base",
            "knight:base",
            "cannoneer:tower",
        ),
    )
    pair = [[tokens[name] for name in side] for side in identities]
    sides = [[catalog.for_token(token) for token in side] for side in pair]
    summary = summarize(sides[0], sides[1], "multi_unit")
    assert summary.threats == 4
    assert len(summary.deployable) == 3
    assert len(summary.spells) == 1
    assert not summary.towers
    assert summary.minimum_cost == 4
    result = extract(pair, catalog, cost=False)
    values = dict(zip(result.names, result.values, strict=True))
    assert values["response.multi_unit.threat_count"] == 4
    assert values["response.multi_unit.deployable_answers"] == 2
    assert values["response.multi_unit.spell_answers"] == 1
    assert values["response.multi_unit.minimum_response_cost"] == 4
    assert values["response.multi_unit.response_context"] == 4
    assert result.values == tuple(-v for v in extract(pair[::-1], catalog, cost=False).values)
    permuted = [side[:8][::-1] + side[8:] for side in pair]
    assert extract(permuted, catalog, cost=False) == result
    assert not any(extract([pair[0], pair[0]], catalog, cost=False).values)
    old = extract(pair, load(PREVIOUS), cost=False)
    assert old.names == result.names
    assert old.values != result.values
