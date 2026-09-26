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


def test_reads_hero_form_from_evolution_level_two() -> None:
    raw = battle()
    raw["team"][0]["cards"][0]["evolutionLevel"] = 2
    raw["team"][0]["cards"][3]["evolutionLevel"] = 2

    result = adapt_battle(raw, "#ABC")

    assert result.skip_reason is None
    assert result.player_deck[0] == "knight:hero"
    assert result.player_deck[3] == "giant:hero"


def test_skips_unrecognized_evolution_level() -> None:
    raw = battle()
    raw["team"][0]["cards"][0]["evolutionLevel"] = 3

    assert adapt_battle(raw, "#ABC").skip_reason == "unknown_card"


def icons(slug: str) -> dict[str, str]:
    return {
        key: f"https://api-assets.clashroyale.com/{kind}/300/{slug}.png"
        for key, kind in (
            ("medium", "cards"),
            ("evolutionMedium", "cardevolutions"),
            ("heroMedium", "cardheroes"),
        )
    }


def test_selects_icon_for_played_form() -> None:
    raw = battle()
    cards = raw["team"][0]["cards"]
    for card in cards[:3]:
        card["iconUrls"] = icons(card["name"])
    cards[1]["evolutionLevel"] = 1
    cards[2]["evolutionLevel"] = 2

    result = adapt_battle(raw, "#ABC")

    assert [card.icon_url for card in result.player_cards[:3]] == [
        icons("Knight")["medium"],
        icons("Archers")["evolutionMedium"],
        icons("Goblins")["heroMedium"],
    ]


def test_rejects_icons_outside_official_host() -> None:
    raw = battle()
    raw["team"][0]["cards"][0]["iconUrls"] = {"medium": "https://example.com/knight.png"}
    raw["team"][0]["cards"][1]["iconUrls"] = "not an object"

    result = adapt_battle(raw, "#ABC")

    assert [card.icon_url for card in result.player_cards[:3]] == [None, None, None]


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
