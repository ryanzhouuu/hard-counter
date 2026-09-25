"""Official API client contracts without live network access."""

import asyncio

import httpx
import pytest

from clash_sos.infrastructure.clash_royale.client import RoyaleAPIError, RoyaleClient


def test_client_encodes_tag_and_keeps_token_in_backend_header() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/battlelog"):
            return httpx.Response(200, json=[{"battleTime": "20260925T120000.000Z"}])
        return httpx.Response(200, json={"tag": "#ABC", "name": "Player"})

    async def fetch() -> tuple[dict[str, object], list[dict[str, object]]]:
        async with httpx.AsyncClient(
            base_url="https://api.clashroyale.com/v1/", transport=httpx.MockTransport(respond)
        ) as http:
            client = RoyaleClient(http, "secret-token")
            return await client.player("#ABC"), await client.battles("#ABC")

    player, battles = asyncio.run(fetch())

    assert player["name"] == "Player"
    assert len(battles) == 1
    assert [request.url.path for request in seen] == [
        "/v1/players/#ABC",
        "/v1/players/#ABC/battlelog",
    ]
    assert all(request.url.raw_path.startswith(b"/v1/players/%23ABC") for request in seen)
    assert all(request.headers["authorization"] == "Bearer secret-token" for request in seen)


@pytest.mark.parametrize(
    ("status", "code"),
    [(404, "player_not_found"), (403, "credentials_rejected"), (429, "rate_limited")],
)
def test_client_maps_expected_failures(status: int, code: str) -> None:
    async def fetch() -> None:
        async with httpx.AsyncClient(
            base_url="https://api.clashroyale.com/v1/",
            transport=httpx.MockTransport(lambda request: httpx.Response(status)),
        ) as http:
            await RoyaleClient(http, "secret-token").player("#ABC")

    with pytest.raises(RoyaleAPIError, match=code):
        asyncio.run(fetch())


def test_client_rejects_bad_json_shape_and_missing_token() -> None:
    async def fetch() -> None:
        async with httpx.AsyncClient(
            base_url="https://api.clashroyale.com/v1/",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        ) as http:
            await RoyaleClient(http, "secret-token").battles("#ABC")

    with pytest.raises(RoyaleAPIError, match="invalid_response"):
        asyncio.run(fetch())
    with pytest.raises(RoyaleAPIError, match="missing_credentials"):
        RoyaleClient(httpx.AsyncClient(), " ")
