"""Recent battle conversion keeps deck-only eligibility and clear skip reasons."""

from typing import Any

from clash_sos.infrastructure.clash_royale.adapter import adapt_battle


def battle(*, mode: str = "Ranked", level: int = 16) -> dict[str, Any]:
    names = ["Knight", "Archers", "Goblins", "Giant", "P.E.K.K.A", "Minions", "Balloon", "Witch"]
    other = [
        "Barbarians",
        "Golem",
        "Skeletons",
        "Valkyrie",
        "Skeleton Army",
        "Bomber",
        "Musketeer",
        "Baby Dragon",
    ]
    return {
        "battleTime": "20260925T120000.000Z",
        "type": "PvP",
        "gameMode": {"name": mode},
        "team": [
            {
                "tag": "#ABC",
                "name": "Player",
                "crowns": 3,
                "cards": [{"name": name, "level": level} for name in names],
            }
        ],
        "opponent": [
            {
                "tag": "#DEF",
                "name": "Opponent",
                "crowns": 1,
                "cards": [{"name": name, "level": level} for name in other],
            }
        ],
    }


def test_adapts_decisive_decks_without_mode_or_level_filter() -> None:
    result = adapt_battle(battle(mode="Friendly", level=1), "#ABC")

    assert result.skip_reason is None
    assert result.outcome == "win"
    assert result.mode == "Friendly"
    assert result.player_deck[0] == "knight:base"
    assert len(result.opponent_deck) == 8


def test_maps_known_evolution_and_hero_forms() -> None:
    raw = battle()
    raw["team"][0]["cards"][0]["evolutionLevel"] = 1
    raw["opponent"][0]["cards"][6] = {"name": "Hero Musketeer", "level": 16}

    result = adapt_battle(raw, "#ABC")

    assert result.skip_reason is None
    assert result.player_deck[0] == "knight:evolution"
    assert result.opponent_deck[6] == "musketeer:hero"


def test_form_marker_does_not_duplicate_name_prefix() -> None:
    raw = battle()
    raw["opponent"][0]["cards"][6] = {
        "name": "Hero Musketeer",
        "heroLevel": 1,
    }

    result = adapt_battle(raw, "#ABC")

    assert result.skip_reason is None
    assert result.opponent_deck[6] == "musketeer:hero"


def test_skips_new_cards_and_new_forms_without_guessing() -> None:
    raw = battle()
    raw["team"][0]["cards"][0]["name"] = "Future Card"
    assert adapt_battle(raw, "#ABC").skip_reason == "unknown_card"

    raw = battle()
    raw["team"][0]["cards"][0]["heroLevel"] = 1
    assert adapt_battle(raw, "#ABC").skip_reason is None
    raw["team"][0]["cards"][1]["heroLevel"] = 1
    assert adapt_battle(raw, "#ABC").skip_reason == "unknown_card"


def test_shows_team_battles_and_draws_without_scoring() -> None:
    raw = battle()
    raw["team"].append({"tag": "#ALLY"})
    assert adapt_battle(raw, "#ABC").skip_reason == "team_battle"

    raw = battle()
    raw["opponent"][0]["crowns"] = 3
    result = adapt_battle(raw, "#ABC")
    assert result.outcome == "draw"
    assert result.skip_reason == "draw"
