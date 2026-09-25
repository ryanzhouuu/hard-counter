from typing import Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from clash_sos import __version__
from clash_sos.infrastructure.settings import Settings, get_settings
from clash_sos.interfaces.api.player_analysis import player_router


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    application = FastAPI(title=resolved_settings.app_name, version=__version__)
    application.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
    application.include_router(player_router(resolved_settings))
    return application


app = create_app()


def run() -> None:
    uvicorn.run(
        "clash_sos.interfaces.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
