import asyncio

from httpx import ASGITransport, AsyncClient, Response

from clash_sos.infrastructure.settings import Settings
from clash_sos.interfaces.api.main import create_app


async def request_health() -> Response:
    transport = ASGITransport(app=create_app(Settings()))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/health")


def test_health_endpoint_reports_version() -> None:
    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_application_uses_configured_name() -> None:
    settings = Settings(app_name="Test API")

    application = create_app(settings)

    assert application.title == "Test API"
