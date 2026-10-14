"""Official card lists map onto catalog identities without trusting foreign hosts."""

import asyncio

import pytest

from clash_sos.infrastructure.clash_royale.card_api_catalog import CURRENT_CARD_API_CATALOG
from clash_sos.infrastructure.clash_royale.card_media import (
    EMPTY_MEDIA,
    CardMedia,
    CardMediaCache,
    parse_card_media,
)
from clash_sos.infrastructure.clash_royale.client import RoyaleAPIError

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


def test_cache_serves_for_a_day_and_retries_failures_after_five_minutes() -> None:
    now = [0.0]
    replies: list[object] = [
        RoyaleAPIError("upstream_unavailable"),
        PAYLOAD,
        RoyaleAPIError("rate_limited"),
    ]
    calls: list[int] = []

    async def fetch() -> object:
        calls.append(1)
        reply = replies.pop(0)
        if isinstance(reply, RoyaleAPIError):
            raise reply
        return reply

    cache = CardMediaCache(fetch, clock=lambda: now[0])

    async def run() -> list[str | None]:
        seen: list[str | None] = []
        for moment in (0, 299, 300, 300 + 86_400):
            now[0] = moment
            seen.append((await cache.get()).icon("knight:base"))
        return seen

    knight = f"{ASSETS}/cards/300/knight.png"
    assert asyncio.run(run()) == [None, None, knight, knight]
    assert len(calls) == 3


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("empty", [{"items": []}, {"items": "invalid"}, None])
def test_empty_refresh_preserves_media_and_retries_after_five_minutes(
    warm: bool, empty: object
) -> None:
    now = [0.0]
    replies = ([PAYLOAD] if warm else []) + [empty, PAYLOAD]
    calls: list[int] = []

    async def fetch() -> object:
        calls.append(1)
        return replies.pop(0)

    cache = CardMediaCache(fetch, clock=lambda: now[0])

    async def run() -> None:
        expected = parse_card_media(PAYLOAD) if warm else EMPTY_MEDIA
        if warm:
            assert await cache.get() == expected
            now[0] = 86_400
        assert await cache.get() == expected
        now[0] += 299
        assert await cache.get() == expected
        assert len(calls) == (2 if warm else 1)
        now[0] += 1
        assert await cache.get() == parse_card_media(PAYLOAD)
        assert len(calls) == (3 if warm else 2)

    asyncio.run(run())


def test_concurrent_refreshes_share_one_fetch() -> None:
    calls: list[int] = []

    async def run() -> list[str | None]:
        release = asyncio.Event()

        async def fetch() -> object:
            calls.append(1)
            await release.wait()
            return PAYLOAD

        cache = CardMediaCache(fetch, clock=lambda: 0.0)
        waiting = [asyncio.create_task(cache.get()) for _ in range(3)]
        await asyncio.sleep(0)
        release.set()
        return [media.icon("knight:base") for media in await asyncio.gather(*waiting)]

    assert asyncio.run(run()) == [f"{ASSETS}/cards/300/knight.png"] * 3
    assert len(calls) == 1


def test_slow_refresh_serves_current_media_and_finishes_in_background() -> None:
    async def run() -> tuple[CardMedia, CardMedia]:
        release = asyncio.Event()

        async def fetch() -> object:
            await release.wait()
            return PAYLOAD

        cache = CardMediaCache(fetch, clock=lambda: 0.0, wait=0.01)
        first = await cache.get()
        release.set()
        await asyncio.sleep(0.01)
        return first, await cache.get()

    first, later = asyncio.run(run())
    assert first == EMPTY_MEDIA
    assert later.icon("knight:base") == f"{ASSETS}/cards/300/knight.png"
