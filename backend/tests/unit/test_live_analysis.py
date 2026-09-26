"""Live reports keep unscored battles visible and summarize eligible ones."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from clash_sos.application import live_analysis
from clash_sos.application.live_model import DeckPair, LiveModelInfo
from clash_sos.domain.analytics import MatchupPrediction, PredictionState


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
        artifact: Path, pairs: Sequence[DeckPair]
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
        _artifact: Path, _pairs: Sequence[DeckPair]
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
        _artifact: Path, pairs: Sequence[DeckPair]
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
