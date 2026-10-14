"""Manual predictions use frozen model inputs without a Royale API token."""

import asyncio
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient, Response
from test_tower_attention import tower_schema

from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.model_artifact import ModelArtifactManifest, ModelOutputFile
from clash_sos.infrastructure.clash_royale import client as client_module
from clash_sos.infrastructure.clash_royale.card_api_catalog import CURRENT_CARD_API_CATALOG
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.ml import attention_artifact_io
from clash_sos.infrastructure.settings import Settings
from clash_sos.interfaces.api.main import create_app

CARDS = [entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries[:16]]


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    members = {
        "card_catalog": ("catalog.json", KAGGLE_V6_CARDS.serialize()),
        "evaluation": ("evaluation.json", b"{}"),
        "feature_schema": ("feature-schema.json", b'{"feature_schema_version":"card-log-odds:v1"}'),
        "predictor": ("predictor.json", b'{"effects":{"knight:base":0.7}}'),
    }
    for path, body in members.values():
        (tmp_path / path).write_bytes(body)
    manifest = ModelArtifactManifest(
        model_version="manual-test",
        dataset_version="test-data",
        catalog_version=KAGGLE_V6_CARDS.version,
        balance_era_id="2026-06",
        files=tuple(
            sorted(
                (
                    ModelOutputFile.model_validate(
                        {
                            "path": path,
                            "kind": kind,
                            "size_bytes": len(body),
                            "sha256": sha256(body).hexdigest(),
                        }
                    )
                    for kind, (path, body) in members.items()
                ),
                key=lambda file: file.path,
            )
        ),
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    return tmp_path


def request(
    artifact: Path, body: dict[str, object] | None = None, *, token: str | None = None
) -> Response:
    async def run() -> Response:
        settings = Settings.model_validate(
            {"active_model_path": artifact, "CLASH_ROYALE_API_TOKEN": token}
        )
        async with AsyncClient(
            transport=ASGITransport(app=create_app(settings)), base_url="http://test"
        ) as client:
            if body is None:
                return await client.get("/api/matchup-options")
            return await client.post("/api/matchup-prediction", json=body)

    return asyncio.run(run())


def decks() -> dict[str, object]:
    return {"side_a": {"cards": CARDS[:8]}, "side_b": {"cards": CARDS[8:]}}


def test_options_offer_frozen_forms_without_towers_for_legacy_models(artifact: Path) -> None:
    response = request(artifact)
    assert response.status_code == 200
    body = response.json()
    assert body["model"]["model_version"] == "manual-test"
    assert body["model"]["input_scope"] == "deck_only"
    assert body["towers"] == []
    assert {card["identity"] for card in body["cards"]} == {
        entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries
    }
    assert {card["form"] for card in body["cards"]} >= {"base", "evolution", "hero", "champion"}


def test_options_omit_media_without_a_token(artifact: Path) -> None:
    cards = request(artifact).json()["cards"]
    assert all(card["icon_url"] is None and card["elixir"] is None for card in cards)


def test_options_include_official_media(artifact: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def cards(_self: object) -> dict[str, object]:
        return {
            "items": [
                {
                    "id": CURRENT_CARD_API_CATALOG["knight"].api_id,
                    "elixirCost": 3,
                    "iconUrls": {"medium": "https://api-assets.clashroyale.com/cards/300/k.png"},
                }
            ]
        }

    monkeypatch.setattr(client_module.RoyaleClient, "cards", cards, raising=False)
    response = request(artifact, token="test-token")
    knight = next(card for card in response.json()["cards"] if card["identity"] == "knight:base")
    assert knight == {
        "identity": "knight:base",
        "name": "Knight",
        "form": "base",
        "icon_url": "https://api-assets.clashroyale.com/cards/300/k.png",
        "elixir": 3,
    }


def test_prediction_preserves_orientation_and_model_provenance(artifact: Path) -> None:
    response = request(artifact, decks())
    assert response.status_code == 200
    body = response.json()
    assert body["side_a_win_probability"] > 0.5
    assert body["side_a_win_probability"] + body["side_b_win_probability"] == pytest.approx(1)
    assert body["model"]["training_era_id"] == "2026-06"
    assert body["model"]["dataset_version"] == "test-data"
    reverse = request(artifact, {"side_a": {"cards": CARDS[8:]}, "side_b": {"cards": CARDS[:8]}})
    assert reverse.json()["side_a_win_probability"] == pytest.approx(body["side_b_win_probability"])
    identical = request(artifact, {"side_a": {"cards": CARDS[:8]}, "side_b": {"cards": CARDS[:8]}})
    assert identical.json()["side_a_win_probability"] == 0.5


@pytest.mark.parametrize(
    "cards",
    [
        CARDS[:7],
        CARDS[:9],
        [CARDS[0]] * 8,
        ["knight:evolution", *CARDS[:7]],
        ["tower-princess:tower", *CARDS[1:8]],
        [1, *CARDS[1:8]],
    ],
)
def test_invalid_decks_are_rejected(artifact: Path, cards: list[object]) -> None:
    response = request(artifact, {"side_a": {"cards": cards}, "side_b": {"cards": CARDS[8:]}})
    assert response.status_code == 422


def test_unknown_forms_are_rejected_without_base_substitution(artifact: Path) -> None:
    response = request(
        artifact,
        {"side_a": {"cards": ["future:base", *CARDS[1:8]]}, "side_b": {"cards": CARDS[8:]}},
    )
    assert response.status_code == 422
    assert "does not cover" in response.json()["detail"]


def test_legacy_model_rejects_ignored_tower_input(artifact: Path) -> None:
    response = request(
        artifact,
        {
            "side_a": {"cards": CARDS[:8], "tower": "cannoneer:tower"},
            "side_b": {"cards": CARDS[8:]},
        },
    )
    assert response.status_code == 422
    assert "omit tower troops" in response.json()["detail"]


@pytest.mark.parametrize("body", [None, {}])
def test_missing_artifact_is_reported(tmp_path: Path, body: dict[str, object] | None) -> None:
    response = request(tmp_path, decks() if body == {} else body)
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_corrupt_frozen_catalog_is_not_offered(artifact: Path) -> None:
    (artifact / "catalog.json").write_text("{}")
    assert request(artifact).status_code == 503


def test_tower_model_offers_and_requires_explicit_tower_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = tower_schema(neural=False)
    model = AttentionMatchupModel(schema)
    model.eval()
    manifest = SimpleNamespace(
        model_version="tower-test",
        dataset_version="current",
        catalog_version=schema.catalog_version,
        balance_era_id=schema.balance_era_id,
    )
    (tmp_path / "manifest.json").write_text('{"manifest_type":"matchup_attention"}')
    (tmp_path / "feature-schema.json").write_text(schema.model_dump_json())

    def load(_path: Path) -> tuple[SimpleNamespace, AttentionCardSchema, AttentionMatchupModel]:
        return manifest, schema, model

    monkeypatch.setattr(attention_artifact_io, "load_attention_artifact", load)
    options = request(tmp_path).json()
    assert options["model"]["input_scope"] == "deck_and_tower"
    assert {tower["identity"] for tower in options["towers"]} == {
        "tower-princess:tower",
        "cannoneer:tower",
        "dagger-duchess:tower",
        "royal-chef:tower",
    }
    missing = request(tmp_path, decks())
    assert missing.status_code == 422
    assert "both decks" in missing.json()["detail"]
    for tower, status in [("cannoneer:tower", 200), ("future:tower", 422)]:
        response = request(
            tmp_path,
            {
                "side_a": {"cards": CARDS[:8], "tower": tower},
                "side_b": {"cards": CARDS[8:], "tower": "tower-princess:tower"},
            },
        )
        assert response.status_code == status
