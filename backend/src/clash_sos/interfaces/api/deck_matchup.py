"""Serve manual matchup predictions; card images and costs come from a cached official list."""

import httpx
from fastapi import APIRouter, HTTPException

from clash_sos.application.deck_matchup import (
    MatchupInputError,
    MatchupOptions,
    MatchupRequest,
    MatchupResult,
    matchup_options,
    predict_deck_matchup,
)
from clash_sos.application.live_model import LiveModelError
from clash_sos.infrastructure.clash_royale.card_media import CardMediaCache
from clash_sos.infrastructure.clash_royale.client import RoyaleAPIError, RoyaleClient
from clash_sos.infrastructure.settings import Settings
from clash_sos.interfaces.api.player_analysis import ERRORS


async def _fetch_cards(settings: Settings) -> object:
    token = settings.royale_api_token
    if token is None:
        raise RoyaleAPIError("missing_credentials")
    async with httpx.AsyncClient(base_url=settings.royale_api_base_url, timeout=12) as client:
        return await RoyaleClient(client, token.get_secret_value()).cards()


def matchup_router(settings: Settings) -> APIRouter:
    router = APIRouter()
    media = CardMediaCache(lambda: _fetch_cards(settings))

    async def options() -> MatchupOptions:
        try:
            return matchup_options(settings.active_model_path, await media.get())
        except LiveModelError as error:
            status, detail = ERRORS.get(error.args[0], ERRORS["model_unavailable"])
            raise HTTPException(status_code=status, detail=detail) from error

    def predict(request: MatchupRequest) -> MatchupResult:
        try:
            return predict_deck_matchup(settings.active_model_path, request)
        except MatchupInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except LiveModelError as error:
            status, detail = ERRORS.get(error.args[0], ERRORS["model_unavailable"])
            raise HTTPException(status_code=status, detail=detail) from error

    router.add_api_route("/api/matchup-options", options, methods=["GET"])
    router.add_api_route("/api/matchup-prediction", predict, methods=["POST"])
    return router
