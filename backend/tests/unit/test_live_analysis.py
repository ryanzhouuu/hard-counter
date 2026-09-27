"""Live reports keep unscored battles visible and summarize eligible ones."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from catalog_fixture import expanded_catalog

from clash_sos.application import live_analysis
from clash_sos.application.live_model import DeckPair, LiveModelInfo, TowerPair
from clash_sos.domain.analytics import MatchupPrediction, PredictionState
from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_schema import (
    AttentionCardSchema,
    AttentionModelConfig,
    build_attention_schema,
)
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.ml import attention_artifact_io


def _battle(time: str, *, unknown: bool = False, crowns: int = 2) -> dict[str, object]:
    """Build a complete official-style one-on-one battle."""
    cards = ["Knight", "Archers", "Goblins", "Giant", "P.E.K.K.A", "Minions", "Balloon", "Witch"]
    others = [
        "Barbarians",
        "Golem",
        "Skeletons",
        "Valkyrie",
        "Skeleton Army",
        "Bomber",
        "Musketeer",
        "Baby Dragon",
    ]
    if unknown:
        cards[0] = "Future Card"
    return {
        "battleTime": time,
        "gameMode": {"name": "Friendly"},
        "team": [{"tag": "#ABC", "crowns": crowns, "cards": [{"name": card} for card in cards]}],
        "opponent": [
            {
                "tag": "#DEF",
                "name": "Rival",
                "crowns": 1,
                "cards": [{"name": card} for card in others],
            }
        ],
    }


def test_live_report_scores_only_eligible_battles(monkeypatch: pytest.MonkeyPatch) -> None:
    info = LiveModelInfo("attention", "data", "catalog", "2026-06")
    seen: list[tuple[Path, Sequence[DeckPair]]] = []

    def score(
        artifact: Path, pairs: Sequence[DeckPair], *, tower_pairs: Sequence[TowerPair] | None = None
    ) -> tuple[LiveModelInfo, tuple[MatchupPrediction, ...]]:
        seen.append((artifact, pairs))
        return info, tuple(
            MatchupPrediction(
                state=PredictionState.AVAILABLE,
                side_a_win_probability=probability,
                provenance=info.provenance,
            )
            for probability in (0.25, 0.75)
        )

    monkeypatch.setattr(live_analysis, "score_live_decks", score)
    result = live_analysis.analyze_live_player(
        {"tag": "#ABC", "name": "Player", "trophies": 9000},
        [
            _battle("20260925T120000Z"),
            _battle("20260925T110000Z", unknown=True),
            _battle("20260925T100000Z", crowns=0),
        ],
        tag="#ABC",
        artifact=Path("model"),
        window_size=2,
    )
    assert len(seen) == 1
    assert len(seen[0][1]) == 2
    assert result.model.training_era_id == "2026-06"
    assert result.schedule.eligible_count == 2
    assert result.schedule.excluded_count == 1
    assert result.schedule.strength_of_schedule == pytest.approx(0.5)
    assert result.schedule.expected_wins == 1
    assert result.schedule.actual_wins == 1
    assert result.battles[0].win_probability == 0.25
    assert result.battles[1].skip_reason == "unknown_card"
    assert result.battles[1].win_probability is None
    assert result.battles[2].win_probability == 0.75


def test_live_report_needs_full_window(monkeypatch: pytest.MonkeyPatch) -> None:
    info = LiveModelInfo("attention", "data", "catalog", "2026-06")

    def score(
        _artifact: Path,
        _pairs: Sequence[DeckPair],
        *,
        tower_pairs: Sequence[TowerPair] | None = None,
    ) -> tuple[LiveModelInfo, tuple[MatchupPrediction, ...]]:
        return (
            info,
            (
                MatchupPrediction(
                    state=PredictionState.AVAILABLE,
                    side_a_win_probability=0.6,
                    provenance=info.provenance,
                ),
            ),
        )

    monkeypatch.setattr(live_analysis, "score_live_decks", score)
    result = live_analysis.analyze_live_player(
        {"tag": "#ABC", "name": "Player"},
        [_battle("20260925T120000Z")],
        tag="#ABC",
        artifact=Path("model"),
        window_size=5,
    )
    assert result.schedule.status == "insufficient_data"
    assert result.schedule.eligible_count == 1
    assert result.schedule.strength_of_schedule is None
    assert not any(battle.in_window for battle in result.battles)


def test_live_report_marks_newest_window_battles(monkeypatch: pytest.MonkeyPatch) -> None:
    info = LiveModelInfo("attention", "data", "catalog", "2026-06")

    def score(
        _artifact: Path,
        pairs: Sequence[DeckPair],
        *,
        tower_pairs: Sequence[TowerPair] | None = None,
    ) -> tuple[LiveModelInfo, tuple[MatchupPrediction, ...]]:
        prediction = MatchupPrediction(
            state=PredictionState.AVAILABLE,
            side_a_win_probability=0.5,
            provenance=info.provenance,
        )
        return info, tuple(prediction for _ in pairs)

    monkeypatch.setattr(live_analysis, "score_live_decks", score)
    result = live_analysis.analyze_live_player(
        {"tag": "#ABC", "name": "Player"},
        [
            _battle("20260925T100000Z"),
            _battle("20260925T120000Z"),
            _battle("20260925T110000Z", unknown=True),
        ],
        tag="#ABC",
        artifact=Path("model"),
        window_size=1,
    )
    assert [battle.in_window for battle in result.battles] == [True, False, False]


def test_live_report_rejects_mismatched_profile() -> None:
    with pytest.raises(ValueError, match="profile_tag_mismatch"):
        live_analysis.analyze_live_player(
            {"tag": "#OTHER"}, [], tag="#ABC", artifact=Path("model"), window_size=5
        )


@pytest.mark.parametrize("all_unsupported", [False, True])
def test_live_report_excludes_model_coverage_without_losing_other_battles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_unsupported: bool
) -> None:
    catalog, _ = expanded_catalog(1)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_bytes(catalog.serialize())
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text('{"manifest_type":"matchup_attention"}')
    schema = build_attention_schema(
        KAGGLE_V6_CARDS.serialize(),
        attributes=CARD_ATTRIBUTES,
        network=AttentionModelConfig(neural_component=False),
    )
    model = AttentionMatchupModel(schema)
    manifest = SimpleNamespace(
        model_version="test",
        dataset_version="test",
        catalog_version=schema.catalog_version,
        balance_era_id=schema.balance_era_id,
    )

    def load(_path: Path) -> tuple[SimpleNamespace, AttentionCardSchema, AttentionMatchupModel]:
        return manifest, schema, model

    monkeypatch.setattr(attention_artifact_io, "load_attention_artifact", load)
    future = _battle("20260925T120000Z")
    future["team"] = [
        {
            "tag": "#ABC",
            "crowns": 2,
            "cards": [
                {"name": name}
                for name in [
                    "Future 0",
                    "Archers",
                    "Goblins",
                    "Giant",
                    "P.E.K.K.A",
                    "Minions",
                    "Balloon",
                    "Witch",
                ]
            ],
        }
    ]
    raw = [future] if all_unsupported else [future, _battle("20260925T110000Z")]
    result = live_analysis.analyze_live_player(
        {"tag": "#ABC", "name": "Player"},
        raw,
        tag="#ABC",
        artifact=artifact,
        window_size=1,
        catalog_path=catalog_path,
    )
    assert result.battles[0].skip_reason == "model_coverage"
    assert result.battles[0].win_probability is None
    assert not result.battles[0].in_window
    assert result.schedule.excluded_count == 1
    assert result.schedule.eligible_count == (0 if all_unsupported else 1)
    if all_unsupported:
        assert result.schedule.status == "insufficient_data"
    else:
        assert result.battles[1].win_probability == 0.5
        assert result.battles[1].in_window
        assert result.schedule.expected_wins == 0.5
