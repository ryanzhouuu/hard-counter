"""HTTP boundary for private, on-demand Clash Royale player lookup."""

import re

import httpx
from fastapi import APIRouter, HTTPException, Query

from clash_sos.application.live_analysis import LiveAnalysisResponse, analyze_live_player
from clash_sos.application.live_model import LiveModelError
from clash_sos.infrastructure.card_inputs import CardInputError
from clash_sos.infrastructure.clash_royale.client import RoyaleAPIError, RoyaleClient
from clash_sos.infrastructure.settings import Settings

BASE_URL = "https://api.clashroyale.com/v1/"

ERRORS: dict[str, tuple[int, str]] = {
    "missing_credentials": (503, "Add CLASH_ROYALE_API_TOKEN to the backend .env file."),
    "credentials_rejected": (503, "The API key was rejected. Check the key and its allowed IP."),
    "player_not_found": (404, "Player tag not found."),
    "rate_limited": (429, "Clash Royale is rate limiting this key. Try again shortly."),
    "upstream_unavailable": (502, "Clash Royale is unavailable. Try again shortly."),
    "invalid_response": (502, "Clash Royale returned an unexpected response."),
    "model_unavailable": (503, "The selected model artifact is unavailable."),
    "missing_ml_runtime": (503, "The selected model requires the ML runtime."),
    "catalog_unavailable": (503, "The selected live card catalog is unavailable or invalid."),
}


def normalize_tag(raw: str) -> str:
    """Accept a typed tag with or without its leading hash."""
    tag = raw.strip().upper()
    if not tag.startswith("#"):
        tag = f"#{tag}"
    if not re.fullmatch(r"#[A-Z0-9]{3,15}", tag):
        raise HTTPException(status_code=400, detail="Enter a valid player tag.")
    return tag


def player_router(settings: Settings) -> APIRouter:
    """Bind the selected model and secret token to one route."""
    router = APIRouter()

    async def player_analysis(
        tag: str = Query(min_length=1), window: int = Query(default=5, ge=1, le=30)
    ) -> LiveAnalysisResponse:
        normalized = normalize_tag(tag)
        token = settings.royale_api_token
        if token is None:
            raise HTTPException(status_code=503, detail=ERRORS["missing_credentials"][1])
        try:
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=12) as client:
                royale = RoyaleClient(client, token.get_secret_value())
                profile = await royale.player(normalized)
                battles = await royale.battles(normalized)
            return analyze_live_player(
                profile,
                battles,
                tag=normalized,
                artifact=settings.active_model_path,
                window_size=window,
                catalog_path=settings.live_catalog_path,
            )
        except CardInputError as error:
            raise HTTPException(status_code=503, detail=ERRORS["catalog_unavailable"][1]) from error
        except (RoyaleAPIError, LiveModelError) as error:
            status, detail = ERRORS.get(error.args[0], ERRORS["upstream_unavailable"])
            raise HTTPException(status_code=status, detail=detail) from error
        except ValueError as error:
            raise HTTPException(
                status_code=502, detail="The player analysis could not be completed."
            ) from error

    router.add_api_route(
        "/api/player-analysis",
        player_analysis,
        methods=["GET"],
        response_model=LiveAnalysisResponse,
    )
    return router
