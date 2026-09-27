"""V2 inference carries both towers and excludes unknown tower records visibly."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_royale_adapter import battle
from test_tower_attention import tower_schema

from clash_sos.application.live_analysis import analyze_live_player
from clash_sos.application.live_model import score_live_decks
from clash_sos.application.model_predict import predict_matchup
from clash_sos.domain.analytics import PredictionState
from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_CARD_CATALOG
from clash_sos.infrastructure.ml import attention_artifact_io


def test_live_and_direct_inference_require_both_towers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = tower_schema()
    model = AttentionMatchupModel(schema)
    model.eval()
    manifest = SimpleNamespace(
        model_version="v2",
        dataset_version="current",
        catalog_version=schema.catalog_version,
        balance_era_id=schema.balance_era_id,
    )
    (tmp_path / "manifest.json").write_text('{"manifest_type":"matchup_attention"}')
    (tmp_path / "feature-schema.json").write_text("{}")

    def load(_path: Path) -> tuple[SimpleNamespace, AttentionCardSchema, AttentionMatchupModel]:
        return manifest, schema, model

    monkeypatch.setattr(attention_artifact_io, "load_attention_artifact", load)
    cards = tuple(entry.card.identity_key for entry in CURRENT_CARD_CATALOG.entries[:8])
    info, predictions = score_live_decks(
        tmp_path,
        [(cards, cards)] * 3,
        tower_pairs=[
            ("cannoneer:tower", "dagger-duchess:tower"),
            (None, "tower-princess:tower"),
            (cards[0], "tower-princess:tower"),
        ],
    )
    assert info.input_scope == "deck_and_tower"
    assert predictions[0].state is PredictionState.AVAILABLE
    assert all(item.state is PredictionState.UNAVAILABLE for item in predictions[1:])
    direct = predict_matchup(
        tmp_path,
        cards,
        cards,
        balance_era_id=schema.balance_era_id,
        side_a_tower="cannoneer:tower",
        side_b_tower="dagger-duchess:tower",
    )
    assert direct.side_a_win_probability == predictions[0].side_a_win_probability
    with pytest.raises(ValueError, match="known tower"):
        predict_matchup(tmp_path, cards, cards, balance_era_id=schema.balance_era_id)
    raw = battle()
    raw["team"][0]["supportCards"] = [{"id": 159000001, "level": 11}]
    raw["opponent"][0]["supportCards"] = [{"id": 159000002, "level": 8}]
    missing = battle()
    missing["battleTime"] = "20260925T110000Z"
    report = analyze_live_player(
        {"tag": "#ABC", "name": "Player"},
        [raw, missing],
        tag="#ABC",
        artifact=tmp_path,
        window_size=1,
    )
    assert report.model.input_scope == "deck_and_tower"
    assert report.battles[0].win_probability is not None
    assert report.battles[0].player_tower is not None
    assert report.battles[0].player_tower.level == 16
    assert report.battles[1].skip_reason == "missing_tower"
    assert report.schedule.eligible_count == 1
