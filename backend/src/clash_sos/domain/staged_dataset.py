"""Staging-row contracts for adaptable Kaggle rows that are not yet accepted canonical battles."""

from collections.abc import Sequence
from datetime import datetime
from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from clash_sos.domain.canonical import (
    BattleOutcome,
    DomainModel,
    PlayerId,
    RecordIssue,
    RecordState,
)
from clash_sos.domain.canonical_dataset import DECK_SIZE, canonical_json_bytes, deck_content_hash
from clash_sos.domain.manifests import SchemaColumnManifest, Sha256

POPULATION_OBSERVATION_ISSUES: tuple[RecordIssue, ...] = (
    RecordIssue.NON_MAX_CARD_LEVEL,
    RecordIssue.STALE_BALANCE_ERA,
    RecordIssue.UNSUPPORTED_MODE,
)

STAGED_SCHEMA_VERSION = "kaggle-v6-staged-schema:v1"

_STAGED_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("source_id", "VARCHAR", False),
    ("timestamp", "TIMESTAMP WITH TIME ZONE", False),
    ("mode", "VARCHAR", False),
    ("balance_era_id", "VARCHAR", True),
    ("outcome", "VARCHAR", False),
    ("event_key", "VARCHAR", False),
    ("fingerprint", "VARCHAR", False),
    ("side_a_player_id", "VARCHAR", False),
    ("side_b_player_id", "VARCHAR", False),
    ("side_a_card_ids", "VARCHAR[]", False),
    ("side_a_card_forms", "VARCHAR[]", False),
    ("side_a_card_levels", "UTINYINT[]", False),
    ("side_a_deck_hash", "VARCHAR", False),
    ("side_b_card_ids", "VARCHAR[]", False),
    ("side_b_card_forms", "VARCHAR[]", False),
    ("side_b_card_levels", "UTINYINT[]", False),
    ("side_b_deck_hash", "VARCHAR", False),
    ("observation_issues", "VARCHAR[]", False),
    ("archive_member", "VARCHAR", False),
    ("row_number", "BIGINT", False),
)

_UNADAPTABLE_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("archive_member", "VARCHAR", False),
    ("row_number", "BIGINT", False),
    ("state", "VARCHAR", False),
    ("issues", "VARCHAR[]", False),
)

STAGED_SCHEMA: tuple[SchemaColumnManifest, ...] = tuple(
    SchemaColumnManifest(name=name, physical_type=physical_type, nullable=nullable)
    for name, physical_type, nullable in _STAGED_COLUMNS
)

UNADAPTABLE_SCHEMA: tuple[SchemaColumnManifest, ...] = tuple(
    SchemaColumnManifest(name=name, physical_type=physical_type, nullable=nullable)
    for name, physical_type, nullable in _UNADAPTABLE_COLUMNS
)


def logical_row_stream_hash(rows: Sequence[DomainModel]) -> str:
    """SHA-256 of newline-delimited canonical JSON in write order."""
    digest = sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row.model_dump(mode="python")) + b"\n")
    return digest.hexdigest()


class StagedBattleRow(DomainModel):
    source_id: str = Field(min_length=1)
    timestamp: datetime
    mode: str = Field(min_length=1)
    balance_era_id: str | None = None
    outcome: Literal[BattleOutcome.SIDE_A_WIN]
    event_key: Sha256
    fingerprint: Sha256
    side_a_player_id: PlayerId
    side_b_player_id: PlayerId
    side_a_card_ids: tuple[str, ...]
    side_a_card_forms: tuple[str, ...]
    side_a_card_levels: tuple[int, ...]
    side_a_deck_hash: Sha256
    side_b_card_ids: tuple[str, ...]
    side_b_card_forms: tuple[str, ...]
    side_b_card_levels: tuple[int, ...]
    side_b_deck_hash: Sha256
    observation_issues: tuple[RecordIssue, ...] = ()
    archive_member: str = Field(min_length=1)
    row_number: int = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("battle timestamp must be timezone-aware")
        return value

    @field_validator("balance_era_id")
    @classmethod
    def validate_balance_era_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("balance_era_id must be non-empty when present")
        return value

    @field_validator("observation_issues")
    @classmethod
    def validate_observation_issues(cls, value: tuple[RecordIssue, ...]) -> tuple[RecordIssue, ...]:
        if not value:
            return value
        allowed = set(POPULATION_OBSERVATION_ISSUES)
        if any(issue not in allowed for issue in value):
            raise ValueError("observation issues must be population predicates only")
        if len(value) != len(set(value)):
            raise ValueError("observation issues must be unique")
        sorted_values = tuple(sorted(value, key=lambda issue: issue.value))
        if value != sorted_values:
            raise ValueError("observation issues must be sorted by issue code")
        return value

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        self._validate_deck(
            self.side_a_card_ids,
            self.side_a_card_forms,
            self.side_a_card_levels,
            self.side_a_deck_hash,
        )
        self._validate_deck(
            self.side_b_card_ids,
            self.side_b_card_forms,
            self.side_b_card_levels,
            self.side_b_deck_hash,
        )
        if self.side_a_player_id == self.side_b_player_id:
            raise ValueError("battle sides must have distinct players")
        return self

    @staticmethod
    def _validate_deck(
        card_ids: tuple[str, ...],
        card_forms: tuple[str, ...],
        card_levels: tuple[int, ...],
        deck_hash: str,
    ) -> None:
        if not (len(card_ids) == len(card_forms) == len(card_levels) == DECK_SIZE):
            raise ValueError("deck arrays must contain exactly eight aligned entries")
        if any(level < 1 or level > 16 for level in card_levels):
            raise ValueError("card levels must be between 1 and 16")
        expected = deck_content_hash(card_ids, card_forms, card_levels)
        if deck_hash != expected:
            raise ValueError("deck hash must match the aligned card identity and levels")


class UnadaptableRow(DomainModel):
    archive_member: str = Field(min_length=1)
    row_number: int = Field(ge=0)
    state: Literal[RecordState.INVALID, RecordState.QUARANTINED]
    issues: tuple[RecordIssue, ...]

    @field_validator("issues")
    @classmethod
    def validate_issues(cls, value: tuple[RecordIssue, ...]) -> tuple[RecordIssue, ...]:
        if not value:
            raise ValueError("unadaptable rows must identify at least one issue")
        if len(value) != len(set(value)):
            raise ValueError("issues must be unique")
        sorted_values = tuple(sorted(value, key=lambda issue: issue.value))
        if value != sorted_values:
            raise ValueError("issues must be sorted by issue code")
        return value
