"""Map the official card list onto catalog identities for images and elixir costs."""

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from clash_sos.infrastructure.clash_royale.adapter import ICON_HOST
from clash_sos.infrastructure.clash_royale.card_api_catalog import CURRENT_CARD_API_CATALOG
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_TOWER_CATALOG
from clash_sos.infrastructure.clash_royale.client import RoyaleAPIError

FORM_ICON_KEYS = {"evolution": "evolutionMedium", "hero": "heroMedium"}


@dataclass(frozen=True)
class CardMedia:
    icons: Mapping[str, Mapping[str, str]] = field(default_factory=dict[str, Mapping[str, str]])
    elixir: Mapping[str, int] = field(default_factory=dict[str, int])
    tower_icons: Mapping[str, str] = field(default_factory=dict[str, str])

    def icon(self, identity: str) -> str | None:
        """Prefer form art; base art stands in when a form has none."""
        root, form = identity.rsplit(":", 1)
        if form == "tower":
            return self.tower_icons.get(identity)
        urls = self.icons.get(root, {})
        return urls.get(FORM_ICON_KEYS.get(form, "medium")) or urls.get("medium")


EMPTY_MEDIA = CardMedia()


def _rows(payload: object, key: str) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    rows = cast(dict[str, object], payload).get(key)
    if not isinstance(rows, list):
        return []
    return [
        cast(dict[str, object], row) for row in cast(list[object], rows) if isinstance(row, dict)
    ]


def _official_icons(row: dict[str, object]) -> dict[str, str]:
    raw = row.get("iconUrls")
    if not isinstance(raw, dict):
        return {}
    return {
        key: url
        for key, url in cast(dict[str, object], raw).items()
        if isinstance(url, str) and url.startswith(ICON_HOST)
    }


def parse_card_media(payload: object) -> CardMedia:
    roots = {entry.api_id: root for root, entry in CURRENT_CARD_API_CATALOG.items()}
    icons: dict[str, dict[str, str]] = {}
    elixir: dict[str, int] = {}
    for row in _rows(payload, "items"):
        api_id = row.get("id")
        root = roots.get(api_id) if type(api_id) is int else None
        if root is None:
            continue
        if found := _official_icons(row):
            icons[root] = found
        cost = row.get("elixirCost")
        if type(cost) is int:
            elixir[root] = cost
    towers: dict[str, str] = {}
    for row in _rows(payload, "supportItems"):
        api_id = row.get("id")
        entry = CURRENT_TOWER_CATALOG.by_api_id.get(api_id) if type(api_id) is int else None
        url = _official_icons(row).get("medium")
        if entry is not None and url is not None:
            towers[entry.identity] = url
    if not icons and not elixir and not towers:
        return EMPTY_MEDIA
    return CardMedia(icons=icons, elixir=elixir, tower_icons=towers)


class CardMediaCache:
    """Serve the last good card list for a day; retry failed fetches after five minutes."""

    def __init__(
        self,
        fetch: Callable[[], Awaitable[object]],
        *,
        clock: Callable[[], float] = time.monotonic,
        ttl: float = 86_400,
        retry_after: float = 300,
    ) -> None:
        self._fetch = fetch
        self._clock = clock
        self._ttl = ttl
        self._retry_after = retry_after
        self._media = EMPTY_MEDIA
        self._expires = float("-inf")

    async def get(self) -> CardMedia:
        now = self._clock()
        if now < self._expires:
            return self._media
        try:
            media = parse_card_media(await self._fetch())
        except RoyaleAPIError:
            self._expires = now + self._retry_after
            return self._media
        self._media, self._expires = media, now + self._ttl
        return media
