from pathlib import Path

import pytest

from clash_sos.infrastructure.settings import Settings


def test_settings_use_project_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLASH_SOS_APP_NAME", raising=False)
    monkeypatch.delenv("CLASH_SOS_LOG_LEVEL", raising=False)

    settings = Settings()

    assert settings.app_name == "Clash SoS API"
    assert settings.log_level == "INFO"


def test_settings_accept_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLASH_SOS_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.log_level == "DEBUG"


def test_settings_accept_live_catalog_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLASH_SOS_LIVE_CATALOG_PATH", "data/config/cards.json")
    assert Settings().live_catalog_path == Path("data/config/cards.json")
