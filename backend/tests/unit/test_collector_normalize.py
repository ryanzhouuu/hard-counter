"""Official match identity and strict training eligibility at the collection edge."""

from copy import deepcopy
from typing import Any

import pytest

from clash_sos.infrastructure.clash_royale.card_api_catalog import CURRENT_CARD_API_CATALOG
from clash_sos.infrastructure.clash_royale.collector_normalize import (
    BattleRejected,
    normalize_battle,
)

LEFT = ("knight", "archers", "goblins", "giant", "pekka", "minions", "balloon", "witch")
RIGHT = (
    "barbarians",
    "golem",
    "skeletons",
    "valkyrie",
    "skeleton-army",
    "bomber",
    "musketeer",
    "baby-dragon",
)
NAMES = {
    "pekka": "P.E.K.K.A",
    "skeleton-army": "Skeleton Army",
    "baby-dragon": "Baby Dragon",
}


def battle() -> dict[str, Any]:
    """Supply complete official-style levels and IDs on both sides."""

    def cards(roots: tuple[str, ...]) -> list[dict[str, object]]:
        return [
            {
                "name": NAMES.get(root, root.replace("-", " ").title()),
                "id": CURRENT_CARD_API_CATALOG[root].api_id,
                "level": CURRENT_CARD_API_CATALOG[root].max_api_level,
                "maxLevel": CURRENT_CARD_API_CATALOG[root].max_api_level,
            }
            for root in roots
        ]

    tower = [{"id": 159000000, "name": "Tower Princess", "level": 16, "maxLevel": 16}]
    return {
        "battleTime": "20260927T120000.000Z",
        "type": "pathOfLegend",
        "gameMode": {"name": "Ranked1v1_NewArena2"},
        "team": [{"tag": "#ABC", "crowns": 3, "cards": cards(LEFT), "supportCards": tower}],
        "opponent": [{"tag": "#DEF", "crowns": 1, "cards": cards(RIGHT), "supportCards": tower}],
    }


def test_opposite_player_and_card_order_share_one_variant() -> None:
    raw = battle()
    first = normalize_battle(raw, "#ABC")
    swapped = deepcopy(raw)
    swapped["team"], swapped["opponent"] = swapped["opponent"], swapped["team"]
    swapped["team"][0]["cards"].reverse()
    second = normalize_battle(swapped, "#DEF")

    assert first.event_key == second.event_key
    assert first.variant_hash == second.variant_hash
    assert first.exclusion_reason is None
    assert first.winner_tag == "#ABC"
    assert [side.tag for side in first.sides] == ["#ABC", "#DEF"]
    assert all(level == 16 for side in first.sides for _, level in side.cards)


def test_tower_and_outcome_disagreements_retain_match_identity() -> None:
    raw = battle()
    first = normalize_battle(raw, "#ABC")
    tower_change = deepcopy(raw)
    tower_change["opponent"][0]["supportCards"] = [
        {"id": 159000001, "name": "Cannoneer", "level": 11, "maxLevel": 11}
    ]
    outcome_change = deepcopy(raw)
    outcome_change["team"][0]["crowns"] = 0
    assert normalize_battle(tower_change, "#ABC").event_key == first.event_key
    assert normalize_battle(outcome_change, "#ABC").event_key == first.event_key
    assert normalize_battle(tower_change, "#ABC").variant_hash != first.variant_hash
    assert normalize_battle(outcome_change, "#ABC").variant_hash != first.variant_hash


def test_nonmax_levels_and_draws_are_normalized_but_ineligible() -> None:
    raw = battle()
    raw["team"][0]["cards"][3]["level"] -= 1
    assert normalize_battle(raw, "#ABC").exclusion_reason == "non_max_level"
    raw["team"][0]["cards"][3]["level"] += 1
    raw["team"][0]["crowns"] = 1
    assert normalize_battle(raw, "#ABC").exclusion_reason == "draw"
    raw["team"][0]["crowns"] = 3
    raw["type"] = "PvP"
    assert normalize_battle(raw, "#ABC").exclusion_reason == "unsupported_mode"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("card_id", "invalid_card_id"),
        ("card_level", "invalid_card_level"),
        ("card_name", "unknown_card"),
        ("tower", "missing_tower"),
        ("opponent", "team_battle"),
    ],
)
def test_bad_observations_have_countable_reasons(change: str, reason: str) -> None:
    raw = battle()
    if change == "card_id":
        raw["team"][0]["cards"][0]["id"] = 999
    elif change == "card_level":
        raw["team"][0]["cards"][0]["maxLevel"] = 1
    elif change == "card_name":
        raw["team"][0]["cards"][0]["name"] = "Future"
    elif change == "tower":
        del raw["team"][0]["supportCards"]
    else:
        raw["opponent"].append({})
    with pytest.raises(BattleRejected, match=reason):
        normalize_battle(raw, "#ABC")
