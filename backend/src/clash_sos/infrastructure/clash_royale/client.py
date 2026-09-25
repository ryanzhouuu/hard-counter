"""Server-side access to the official player and recent-battle endpoints."""

from typing import cast
from urllib.parse import quote

import httpx


class RoyaleAPIError(Exception):
    """A stable failure category for API responses and transport errors."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RoyaleClient:
    """Fetch player data with a key that never leaves the backend."""

    def __init__(self, client: httpx.AsyncClient, token: str) -> None:
        if not token.strip():
            raise RoyaleAPIError("missing_credentials")
        self.client = client
        self.token = token

    async def player(self, tag: str) -> dict[str, object]:
        """Return one player object or a categorized upstream failure."""
        payload = await self._get(f"players/{quote(tag, safe='')}")
        if not isinstance(payload, dict):
            raise RoyaleAPIError("invalid_response")
        return cast(dict[str, object], payload)

    async def battles(self, tag: str) -> list[dict[str, object]]:
        """Return recent battle objects; an object wrapper is invalid here."""
        payload = await self._get(f"players/{quote(tag, safe='')}/battlelog")
        if not isinstance(payload, list):
            raise RoyaleAPIError("invalid_response")
        rows = cast(list[object], payload)
        if any(not isinstance(row, dict) for row in rows):
            raise RoyaleAPIError("invalid_response")
        return cast(list[dict[str, object]], rows)

    async def _get(self, path: str) -> object:
        try:
            response = await self.client.get(
                path,
                headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
            )
        except httpx.RequestError as error:
            raise RoyaleAPIError("upstream_unavailable") from error
        if response.status_code == 404:
            raise RoyaleAPIError("player_not_found")
        if response.status_code in {401, 403}:
            raise RoyaleAPIError("credentials_rejected")
        if response.status_code == 429:
            raise RoyaleAPIError("rate_limited")
        if response.is_error:
            raise RoyaleAPIError("upstream_unavailable")
        try:
            return response.json()
        except ValueError as error:
            raise RoyaleAPIError("invalid_response") from error
