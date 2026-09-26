"""Actual supportCards/form representations retain independent tower inputs."""

import pytest
from test_royale_adapter import battle

from clash_sos.infrastructure.clash_royale.adapter import adapt_battle
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_TOWER_CATALOG
from clash_sos.infrastructure.clash_royale.tower_adapter import adapt_tower


def test_every_tower_normalizes_without_consuming_a_deck_slot() -> None:
    for entry in CURRENT_TOWER_CATALOG.entries:
        raw = battle()
        raw["team"][0]["supportCards"] = [
            {
                "id": entry.api_id,
                "name": entry.name,
                "level": entry.max_api_level,
                "maxLevel": entry.max_api_level,
            }
        ]
        result = adapt_battle(raw, "#ABC")
        assert result.player_tower is not None
        assert result.player_tower.identity == entry.identity and result.player_tower.level == 16
        assert len(result.player_deck) == 8 and result.skip_reason is None
        swapped = adapt_battle(raw, "#DEF")
        assert swapped.opponent_tower == result.player_tower


@pytest.mark.parametrize(
    ("values", "issue"),
    [
        (None, "missing_tower"),
        (list[object](), "missing_tower"),
        ([dict[str, object]()], "unknown_tower"),
        ([{"id": 999, "name": "Future"}], "unknown_tower"),
        ([{"id": 159000000, "name": "Cannoneer"}], "invalid_tower"),
        ([{"id": 159000000, "level": True}], "invalid_tower"),
        ([{"id": 159000000, "level": 17}], "invalid_tower"),
        ([dict[str, object](), dict[str, object]()], "invalid_tower"),
    ],
)
def test_missing_and_conflicting_towers_remain_visible(values: object, issue: str) -> None:
    tower = adapt_tower({"supportCards": values})
    assert tower.identity is None and tower.issue == issue


@pytest.mark.parametrize(
    ("name", "form", "identity"),
    [
        ("Ronin", None, "ronin:base"),
        ("Minion Giant", None, "minion-giant:base"),
        ("Elite Barbarians", 1, "elite-barbarians:evolution"),
        ("Valkyrie", 2, "valkyrie:hero"),
        ("Berserker", 2, "berserker:hero"),
        ("Ice Wizard", 2, "ice-wizard:hero"),
    ],
)
def test_released_cards_map_from_observed_api_markers(
    name: str, form: int | None, identity: str
) -> None:
    raw = battle()
    raw["team"][0]["cards"][0] = {"name": name, "evolutionLevel": form}
    assert adapt_battle(raw, "#ABC").player_deck[0] == identity
