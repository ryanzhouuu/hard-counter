"""Official card lists map onto catalog identities without trusting foreign hosts."""

from clash_sos.infrastructure.clash_royale.card_api_catalog import CURRENT_CARD_API_CATALOG
from clash_sos.infrastructure.clash_royale.card_media import EMPTY_MEDIA, parse_card_media

ASSETS = "https://api-assets.clashroyale.com"
IDS = {root: entry.api_id for root, entry in CURRENT_CARD_API_CATALOG.items()}
PAYLOAD: dict[str, object] = {
    "items": [
        {
            "id": IDS["knight"],
            "elixirCost": 3,
            "iconUrls": {
                "medium": f"{ASSETS}/cards/300/knight.png",
                "evolutionMedium": f"{ASSETS}/cardevolutions/300/knight.png",
                "heroMedium": f"{ASSETS}/cardheroes/300/knight.png",
            },
        },
        {"id": IDS["archer-queen"], "elixirCost": 5, "iconUrls": {"medium": f"{ASSETS}/aq.png"}},
        {"id": IDS["mirror"], "iconUrls": {"medium": "https://evil.example/mirror.png"}},
        {"id": 99_999_999, "elixirCost": 4, "iconUrls": {"medium": f"{ASSETS}/future.png"}},
        {"id": "26000001", "elixirCost": 3},
        "not a row",
    ],
    "supportItems": [{"id": 159_000_001, "iconUrls": {"medium": f"{ASSETS}/cannoneer.png"}}],
}


def test_icons_follow_card_forms() -> None:
    media = parse_card_media(PAYLOAD)
    assert media.icon("knight:base") == f"{ASSETS}/cards/300/knight.png"
    assert media.icon("knight:evolution") == f"{ASSETS}/cardevolutions/300/knight.png"
    assert media.icon("knight:hero") == f"{ASSETS}/cardheroes/300/knight.png"
    assert media.icon("archer-queen:champion") == f"{ASSETS}/aq.png"
    assert media.icon("archer-queen:evolution") == f"{ASSETS}/aq.png"
    assert media.icon("cannoneer:tower") == f"{ASSETS}/cannoneer.png"
    assert media.icon("tower-princess:tower") is None


def test_foreign_hosts_unknown_ids_and_missing_costs_are_dropped() -> None:
    media = parse_card_media(PAYLOAD)
    assert media.icon("mirror:base") is None
    assert dict(media.elixir) == {"knight": 3, "archer-queen": 5}


def test_unexpected_shapes_yield_empty_media() -> None:
    payloads: list[object] = [[], {"items": "x", "supportItems": None}, None]
    for payload in payloads:
        assert parse_card_media(payload) == EMPTY_MEDIA
