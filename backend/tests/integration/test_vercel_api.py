"""The deployment entrypoint exposes built assets without intercepting API routes."""

import asyncio
from pathlib import Path

import pytest
from app import create_vercel_app
from httpx import ASGITransport, AsyncClient, Response


@pytest.fixture
def public(tmp_path: Path) -> Path:
    (tmp_path / "index.html").write_text('<html><div id="root"></div></html>')
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app-hash.js").write_text("window.app = true;")
    return tmp_path


def request(public: Path, path: str, *, method: str = "GET") -> Response:
    async def run() -> Response:
        async with AsyncClient(
            transport=ASGITransport(app=create_vercel_app(public)), base_url="http://test"
        ) as client:
            if method == "POST":
                return await client.post(path, json={})
            return await client.get(path)

    return asyncio.run(run())


@pytest.mark.parametrize("path", ["/", "/?view=matchup", "/?tag=ABC&window=5"])
def test_frontend_entrypoint_handles_query_parameters(public: Path, path: str) -> None:
    response = request(public, path)

    assert response.status_code == 200
    assert response.text == (public / "index.html").read_text()


def test_frontend_assets_are_served_at_their_built_paths(public: Path) -> None:
    response = request(public, "/assets/app-hash.js")

    assert response.status_code == 200
    assert response.text == "window.app = true;"


def test_api_routes_take_precedence_over_the_frontend_mount(public: Path) -> None:
    assert request(public, "/health").json()["status"] == "ok"
    assert request(public, "/api/matchup-prediction", method="POST").status_code == 422


@pytest.mark.parametrize("path", ["/assets/missing.js", "/api/not-a-route"])
def test_missing_paths_remain_not_found(public: Path, path: str) -> None:
    assert request(public, path).status_code == 404


def test_api_startup_does_not_require_locally_present_cdn_assets(tmp_path: Path) -> None:
    assert request(tmp_path / "missing-public", "/health").status_code == 200
