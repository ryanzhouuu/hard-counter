"""Build an on-demand player report from recent official battles."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from clash_sos.application.live_model import score_live_decks
from clash_sos.application.rolling_sos import summarize_observations
from clash_sos.domain.analytics import RollingSoSObservation
from clash_sos.infrastructure.clash_royale.adapter import LiveBattle, LiveCard, adapt_battle


class LivePlayer(BaseModel):
    tag: str
    name: str
    trophies: int | None


class LiveCardReport(BaseModel):
    name: str
    icon_url: str | None


class LiveBattleReport(BaseModel):
    timestamp: datetime | None
    mode: str
    opponent_name: str
    outcome: str
    player_cards: tuple[LiveCardReport, ...]
    opponent_cards: tuple[LiveCardReport, ...]
    skip_reason: str | None
    win_probability: float | None = Field(ge=0, le=1)


class LiveSchedule(BaseModel):
    status: str
    requested_window: int
    eligible_count: int
    excluded_count: int
    strength_of_schedule: float | None
    expected_wins: float | None
    actual_wins: int | None
    performance_above_expectation: float | None


class LiveModelReport(BaseModel):
    model_version: str
    dataset_version: str
    catalog_version: str
    training_era_id: str


class LiveAnalysisResponse(BaseModel):
    player: LivePlayer
    model: LiveModelReport
    schedule: LiveSchedule
    battles: tuple[LiveBattleReport, ...]


def _player(profile: dict[str, object], tag: str) -> LivePlayer:
    """Use only the requested profile and public display fields."""
    if profile.get("tag") != tag:
        raise ValueError("profile_tag_mismatch")
    name = profile.get("name")
    trophies = profile.get("trophies")
    return LivePlayer(
        tag=tag,
        name=name if isinstance(name, str) and name else tag,
        trophies=trophies if type(trophies) is int else None,
    )


def _card_reports(cards: tuple[LiveCard, ...]) -> tuple[LiveCardReport, ...]:
    return tuple(LiveCardReport(name=card.name, icon_url=card.icon_url) for card in cards)


def _report(battle: LiveBattle, probability: float | None) -> LiveBattleReport:
    return LiveBattleReport(
        timestamp=battle.timestamp,
        mode=battle.mode,
        opponent_name=battle.opponent_name,
        outcome=battle.outcome,
        player_cards=_card_reports(battle.player_cards),
        opponent_cards=_card_reports(battle.opponent_cards),
        skip_reason=battle.skip_reason,
        win_probability=probability,
    )


def analyze_live_player(
    profile: dict[str, object],
    raw_battles: list[dict[str, object]],
    *,
    tag: str,
    artifact: Path,
    window_size: int,
) -> LiveAnalysisResponse:
    """Score known decisive 1v1 battles and summarize the newest eligible window."""
    player = _player(profile, tag)
    battles = [adapt_battle(raw, tag) for raw in raw_battles]
    eligible = [battle for battle in battles if battle.skip_reason is None]
    info, predictions = score_live_decks(
        artifact,
        [(battle.player_deck, battle.opponent_deck) for battle in eligible],
    )
    observations: list[RollingSoSObservation] = []
    probabilities: dict[str, float] = {}
    for battle, prediction in zip(eligible, predictions, strict=True):
        if battle.timestamp is None or prediction.side_a_win_probability is None:
            raise ValueError("eligible_battle_unscored")
        probability = prediction.side_a_win_probability
        probabilities[battle.fingerprint] = probability
        observations.append(
            RollingSoSObservation(
                timestamp=battle.timestamp,
                battle_fingerprint=battle.fingerprint,
                player_win_probability=probability,
                actual_win=int(battle.outcome == "win"),
                difficulty=1 - probability,
                provenance=info.provenance,
            )
        )
    summary = summarize_observations(
        tag,
        observations,
        window_size=window_size,
        excluded_count=len(battles) - len(eligible),
    )
    ordered = sorted(
        battles,
        key=lambda battle: battle.timestamp.timestamp() if battle.timestamp else 0,
        reverse=True,
    )
    return LiveAnalysisResponse(
        player=player,
        model=LiveModelReport(**vars(info)),
        schedule=LiveSchedule(
            status=summary.status.value,
            requested_window=summary.requested_window,
            eligible_count=summary.eligible_count,
            excluded_count=summary.excluded_count,
            strength_of_schedule=summary.strength_of_schedule,
            expected_wins=summary.expected_wins,
            actual_wins=summary.actual_wins,
            performance_above_expectation=summary.performance_above_expectation,
        ),
        battles=tuple(_report(battle, probabilities.get(battle.fingerprint)) for battle in ordered),
    )
