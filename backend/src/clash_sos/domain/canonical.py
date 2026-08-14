"""Canonical domain values and entities shared by preparation and application services."""

import re
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from json import dumps
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    field_validator,
    model_validator,
)


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CardId(RootModel[str]):
    """Stable catalog key; source IDs and display names are not stored here."""

    root: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

    @property
    def value(self) -> str:
        return self.root


class PlayerId(RootModel[str]):
    """Normalized player tag or source-scoped identifier."""

    root: str = Field(min_length=1, max_length=160)

    @field_validator("root", mode="before")
    @classmethod
    def normalize(cls, value: str) -> str:
        normalized = value.strip()
        if normalized.startswith("#"):
            normalized = normalized.upper()
        if not re.fullmatch(r"#[A-Z0-9]+|[A-Za-z0-9][A-Za-z0-9_.:-]*", normalized):
            raise ValueError("player ID must be a tag or source-scoped identifier")
        return normalized

    @property
    def value(self) -> str:
        return self.root


class CardForm(StrEnum):
    BASE = "base"
    EVOLUTION = "evolution"
    HERO = "hero"
    CHAMPION = "champion"


class CardRef(DomainModel):
    card_id: CardId
    form: CardForm

    @property
    def identity_key(self) -> str:
        return f"{self.card_id.value}:{self.form.value}"


class Deck(DomainModel):
    cards: tuple[CardRef, ...]

    @model_validator(mode="after")
    def validate_cards(self) -> Self:
        if len(self.cards) != 8:
            raise ValueError("deck must contain exactly eight cards")
        identities = [card.identity_key for card in self.cards]
        if len(set(identities)) != len(identities):
            raise ValueError("deck must contain unique card-form pairs")
        return self

    @computed_field
    @property
    def canonical_hash(self) -> str:
        payload = "|".join(sorted(card.identity_key for card in self.cards))
        return sha256(payload.encode("utf-8")).hexdigest()


class BattleOutcome(StrEnum):
    SIDE_A_WIN = "side_a_win"
    SIDE_B_WIN = "side_b_win"
    DRAW = "draw"

    def swapped(self) -> "BattleOutcome":
        return {
            self.SIDE_A_WIN: self.SIDE_B_WIN,
            self.SIDE_B_WIN: self.SIDE_A_WIN,
            self.DRAW: self.DRAW,
        }[self]


class RecordState(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    QUARANTINED = "quarantined"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_DATA = "insufficient_data"


class RecordIssue(StrEnum):
    MALFORMED_TIMESTAMP = "malformed_timestamp"
    UNKNOWN_CARD = "unknown_card"
    REPEATED_CARD = "repeated_card"
    INCOMPLETE_DECK = "incomplete_deck"
    DUPLICATE_BATTLE = "duplicate_battle"
    UNSUPPORTED_MODE = "unsupported_mode"
    NON_MAX_CARD_LEVEL = "non_max_card_level"
    DRAW_OUTCOME = "draw_outcome"
    STALE_BALANCE_ERA = "stale_balance_era"
    UNAVAILABLE_MODEL_COVERAGE = "unavailable_model_coverage"
    TARGET_PLAYER_NOT_FOUND = "target_player_not_found"
    INSUFFICIENT_HISTORY = "insufficient_history"


class RecordDisposition(DomainModel):
    state: RecordState
    issues: tuple[RecordIssue, ...] = ()
    detail: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.state is RecordState.VALID and self.issues:
            raise ValueError("valid records cannot carry issues")
        if self.state is not RecordState.VALID and not self.issues:
            raise ValueError("non-valid records must identify at least one issue")
        return self


class BalanceChange(DomainModel):
    card_id: CardId
    change_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class BalanceEra(DomainModel):
    era_id: str = Field(min_length=1)
    valid_from: datetime
    valid_to: datetime | None = None
    card_catalog_version: str = Field(min_length=1)
    changed_cards: tuple[BalanceChange, ...] = ()

    @field_validator("valid_from", "valid_to")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("balance-era timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("balance-era valid_to must be later than valid_from")
        return self

    def contains(self, timestamp: datetime) -> bool:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return self.valid_from <= timestamp and (self.valid_to is None or timestamp < self.valid_to)


class BattleSide(DomainModel):
    player_id: PlayerId
    deck: Deck
    card_levels: tuple[int, ...]

    @model_validator(mode="after")
    def validate_levels(self) -> Self:
        if len(self.card_levels) != len(self.deck.cards):
            raise ValueError("card levels must align with the deck order")
        if any(level < 1 or level > 16 for level in self.card_levels):
            raise ValueError("card levels must be between 1 and 16")
        return self


class Battle(DomainModel):
    side_a: BattleSide
    side_b: BattleSide
    outcome: BattleOutcome
    timestamp: datetime
    mode: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    balance_era: BalanceEra

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("battle timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_sides_and_era(self) -> Self:
        if self.side_a.player_id == self.side_b.player_id:
            raise ValueError("battle sides must have distinct players")
        if not self.balance_era.contains(self.timestamp):
            raise ValueError("battle timestamp must fall within its balance era")
        return self

    @computed_field
    @property
    def event_key(self) -> str:
        """Identify an event independently of side orientation and outcome."""

        return self._identity_hash(include_outcome=False)

    @computed_field
    @property
    def fingerprint(self) -> str:
        """Identify an event and its normalized outcome."""

        return self._identity_hash(include_outcome=True)

    def _identity_hash(self, *, include_outcome: bool) -> str:
        side_a = self._side_identity(self.side_a)
        side_b = self._side_identity(self.side_b)
        sides = sorted((side_a, side_b), key=lambda side: dumps(side, separators=(",", ":")))
        payload: dict[str, object] = {
            "source_id": self.source_id,
            "mode": self.mode,
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "sides": sides,
        }
        if include_outcome:
            payload["winner"] = self._winner_identity(side_a, side_b)
        encoded = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return sha256(encoded).hexdigest()

    @staticmethod
    def _side_identity(side: BattleSide) -> dict[str, object]:
        cards = sorted(
            (card.identity_key, level)
            for card, level in zip(side.deck.cards, side.card_levels, strict=True)
        )
        return {"player_id": side.player_id.value, "cards": cards}

    def _winner_identity(
        self, side_a: dict[str, object], side_b: dict[str, object]
    ) -> dict[str, object] | str:
        if self.outcome is BattleOutcome.DRAW:
            return "draw"
        return side_a if self.outcome is BattleOutcome.SIDE_A_WIN else side_b
