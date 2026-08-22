"""Canonical row adapter for the winner-first Kaggle version 6 schema."""

from collections.abc import Mapping
from datetime import UTC, datetime

from pydantic import Field, ValidationError

from clash_sos.domain.canonical import (
    BalanceEraRegistry,
    Battle,
    BattleOutcome,
    BattleSide,
    CardRef,
    Deck,
    DomainModel,
    PlayerId,
    RecordDisposition,
    RecordIssue,
    RecordState,
)
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS, KaggleCardCatalog
from clash_sos.infrastructure.kaggle_v6.schema import ROW_COLUMNS
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_SOURCE_ID


class SourceRowLocation(DomainModel):
    archive_member: str = Field(min_length=1)
    row_number: int = Field(ge=0)

    @property
    def identity_key(self) -> str:
        return f"{self.archive_member}:{self.row_number}"


class AdaptedKaggleRecord(DomainModel):
    location: SourceRowLocation
    battle: Battle | None
    balance_era_id: str | None = None
    disposition: RecordDisposition


class KaggleRowError(ValueError):
    def __init__(self, issue: RecordIssue, detail: str) -> None:
        super().__init__(detail)
        self.issue = issue
        self.detail = detail


def adapt_valid_row(
    row: Mapping[str, object],
    *,
    catalog: KaggleCardCatalog = KAGGLE_V6_CARDS,
) -> Battle:
    missing = sorted(set(ROW_COLUMNS).difference(row))
    if missing:
        raise KaggleRowError(RecordIssue.MALFORMED_ROW, f"missing columns: {', '.join(missing)}")
    try:
        timestamp = _timestamp(row["time"])
        winner = _side(row, "winner", catalog)
        loser = _side(row, "loser", catalog)
        if winner.player_id == loser.player_id:
            raise KaggleRowError(
                RecordIssue.IDENTICAL_PLAYERS,
                "winner and loser must be distinct players",
            )
        mode = _required_string(row["game_mode"], "game_mode")
        return Battle(
            side_a=winner,
            side_b=loser,
            outcome=BattleOutcome.SIDE_A_WIN,
            timestamp=timestamp,
            mode=mode,
            source_id=KAGGLE_V6_SOURCE_ID,
        )
    except KaggleRowError:
        raise
    except ValidationError as error:
        raise KaggleRowError(RecordIssue.MALFORMED_ROW, str(error)) from error


def adapt_row(
    row: Mapping[str, object],
    *,
    location: SourceRowLocation,
    era_registry: BalanceEraRegistry,
    catalog: KaggleCardCatalog = KAGGLE_V6_CARDS,
) -> AdaptedKaggleRecord:
    try:
        battle = adapt_valid_row(row, catalog=catalog)
    except KaggleRowError as error:
        invalid_issues = {RecordIssue.MALFORMED_ROW, RecordIssue.MALFORMED_TIMESTAMP}
        state = RecordState.INVALID if error.issue in invalid_issues else RecordState.QUARANTINED
        return AdaptedKaggleRecord(
            location=location,
            battle=None,
            disposition=RecordDisposition(
                state=state,
                issues=(error.issue,),
                detail=error.detail,
            ),
        )

    era = era_registry.lookup(battle.timestamp)
    issues: list[RecordIssue] = []
    if battle.mode != "Ranked1v1_NewArena":
        issues.append(RecordIssue.UNSUPPORTED_MODE)
    if any(level != 16 for side in (battle.side_a, battle.side_b) for level in side.card_levels):
        issues.append(RecordIssue.NON_MAX_CARD_LEVEL)
    if era is None:
        issues.append(RecordIssue.STALE_BALANCE_ERA)
    disposition = (
        RecordDisposition(state=RecordState.UNSUPPORTED, issues=tuple(issues))
        if issues
        else RecordDisposition(state=RecordState.VALID)
    )
    return AdaptedKaggleRecord(
        location=location,
        battle=battle,
        balance_era_id=None if era is None else era.era_id,
        disposition=disposition,
    )


def _side(row: Mapping[str, object], side: str, catalog: KaggleCardCatalog) -> BattleSide:
    player = PlayerId(_required_string(row[f"{side}_id"], f"{side}_id"))
    cards: list[CardRef] = []
    levels: list[int] = []
    identities: set[str] = set()
    for index in range(8):
        card_value = row[f"{side}_card_{index}"]
        if card_value is None:
            raise KaggleRowError(RecordIssue.INCOMPLETE_DECK, f"{side} deck is incomplete")
        source_id = _integer(card_value, f"{side}_card_{index}")
        entry = catalog.find(source_id)
        if entry is None:
            raise KaggleRowError(RecordIssue.UNKNOWN_CARD, f"unknown source card ID {source_id}")
        if entry.card.identity_key in identities:
            raise KaggleRowError(RecordIssue.REPEATED_CARD, f"{side} deck repeats a card")
        identities.add(entry.card.identity_key)
        cards.append(entry.card)

        level_value = row[f"{side}_card_{index}_level"]
        if level_value is None:
            raise KaggleRowError(RecordIssue.INCOMPLETE_DECK, f"{side} levels are incomplete")
        levels.append(_integer(level_value, f"{side}_card_{index}_level"))
    return BattleSide(player_id=player, deck=Deck(cards=tuple(cards)), card_levels=tuple(levels))


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KaggleRowError(RecordIssue.MALFORMED_ROW, f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise KaggleRowError(RecordIssue.MALFORMED_ROW, f"{field} must be an integer")
    return value


def _timestamp(value: object) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise KaggleRowError(RecordIssue.MALFORMED_TIMESTAMP, "invalid time") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise KaggleRowError(RecordIssue.MALFORMED_TIMESTAMP, "time must include an offset")
    return value.astimezone(UTC)
