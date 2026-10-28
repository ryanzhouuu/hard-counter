"""Expose API routes and a discoverable frontend mount for Vercel's CDN collector."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from clash_sos.interfaces.api.main import create_app


def create_vercel_app(public_directory: Path = Path(__file__).parent / "public") -> FastAPI:
    """Keep API routes ahead of the frontend, allowing CDN-only asset directories."""
    application = create_app()
    application.mount(
        "/", StaticFiles(directory=public_directory, html=True, check_dir=False), name="frontend"
    )
    return application


app = create_vercel_app()
