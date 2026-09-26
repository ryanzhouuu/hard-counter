"""The HTTP route keeps the key server-side and returns live analysis."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient, Response

from clash_sos.application import live_analysis
from clash_sos.application.live_model import DeckPair, LiveModelInfo
from clash_sos.domain.analytics import MatchupPrediction, PredictionState
from clash_sos.infrastructure.clash_royale.client import RoyaleAPIError, RoyaleClient
from clash_sos.infrastructure.settings import Settings
from clash_sos.interfaces.api.main import create_app

KNIGHT_ICON = "https://api-assets.clashroyale.com/cards/300/knight.png"


def _settings() -> Settings:
    return Settings.model_validate({"CLASH_ROYALE_API_TOKEN": "test-secret"})


async def _request(path: str, settings: Settings) -> Response:
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings)), base_url="http://test"
    ) as client:
        return await client.get(path)


def test_lookup_returns_scored_recent_battle(monkeypatch: pytest.MonkeyPatch) -> None:
    async def player(_self: RoyaleClient, tag: str) -> dict[str, object]:
        assert tag == "#ABC"
        return {"tag": tag, "name": "Player", "trophies": 1234}

    async def battles(_self: RoyaleClient, _tag: str) -> list[dict[str, object]]:
        names = [
            "Knight",
            "Archers",
            "Goblins",
            "Giant",
            "P.E.K.K.A",
            "Minions",
            "Balloon",
            "Witch",
        ]
        opponents = [
            "Barbarians",
            "Golem",
            "Skeletons",
            "Valkyrie",
            "Skeleton Army",
            "Bomber",
            "Musketeer",
            "Baby Dragon",
        ]
        cards: list[dict[str, object]] = [{"name": name} for name in names]
        cards[0]["iconUrls"] = {"medium": KNIGHT_ICON}
        return [
            {
                "battleTime": "20260925T120000Z",
                "gameMode": {"name": "Friendly"},
                "team": [{"tag": "#ABC", "crowns": 2, "cards": cards}],
                "opponent": [
                    {
                        "tag": "#DEF",
                        "name": "Rival",
                        "crowns": 1,
                        "cards": [{"name": name} for name in opponents],
                    }
                ],
            }
        ]

    info = LiveModelInfo("attention", "data", "catalog", "2026-06")

    def score(
        _artifact: Path, _pairs: Sequence[DeckPair]
    ) -> tuple[LiveModelInfo, tuple[MatchupPrediction, ...]]:
        return info, (
            MatchupPrediction(
                state=PredictionState.AVAILABLE,
                side_a_win_probability=0.7,
                provenance=info.provenance,
            ),
        )

    monkeypatch.setattr(RoyaleClient, "player", player)
    monkeypatch.setattr(RoyaleClient, "battles", battles)
    monkeypatch.setattr(live_analysis, "score_live_decks", score)
    response = asyncio.run(_request("/api/player-analysis?tag=abc&window=1", _settings()))

    assert response.status_code == 200
    body = response.json()
    assert body["player"] == {"tag": "#ABC", "name": "Player", "trophies": 1234}
    assert body["model"]["training_era_id"] == "2026-06"
    assert body["schedule"]["strength_of_schedule"] == pytest.approx(0.3)
    assert body["battles"][0]["win_probability"] == 0.7
    assert body["battles"][0]["player_cards"][:2] == [
        {"name": "Knight", "icon_url": KNIGHT_ICON},
        {"name": "Archers", "icon_url": None},
    ]
    assert "test-secret" not in response.text


def test_lookup_rejects_bad_tag() -> None:
    response = asyncio.run(_request("/api/player-analysis?tag=oops!", _settings()))
    assert response.status_code == 400


def test_lookup_reports_key_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    async def player(_self: RoyaleClient, _tag: str) -> dict[str, object]:
        raise RoyaleAPIError("credentials_rejected")

    monkeypatch.setattr(RoyaleClient, "player", player)
    response = asyncio.run(_request("/api/player-analysis?tag=ABC", _settings()))
    assert response.status_code == 503
    assert "allowed IP" in response.json()["detail"]
