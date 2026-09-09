"""Accepted canonical row contracts for versioned processed datasets."""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from json import dumps
from typing import Any, Literal, Self, cast

from pydantic import Field, field_validator, model_validator

from clash_sos.domain.canonical import BattleOutcome, DomainModel, PlayerId
from clash_sos.domain.manifests import SchemaColumnManifest, Sha256

CANONICAL_SCHEMA_VERSION = "kaggle-v6-ranked16-schema:v1"
ACCEPTED_MODE = "Ranked1v1_NewArena"
ACCEPTED_CARD_LEVEL = 16
DECK_SIZE = 8

_CANONICAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("dataset_version", "VARCHAR"),
    ("source_id", "VARCHAR"),
    ("timestamp", "TIMESTAMP WITH TIME ZONE"),
    ("mode", "VARCHAR"),
    ("balance_era_id", "VARCHAR"),
    ("outcome", "VARCHAR"),
    ("event_key", "VARCHAR"),
    ("fingerprint", "VARCHAR"),
    ("side_a_player_id", "VARCHAR"),
    ("side_b_player_id", "VARCHAR"),
    ("side_a_card_ids", "VARCHAR[]"),
    ("side_a_card_forms", "VARCHAR[]"),
    ("side_a_card_levels", "UTINYINT[]"),
    ("side_a_deck_hash", "VARCHAR"),
    ("side_b_card_ids", "VARCHAR[]"),
    ("side_b_card_forms", "VARCHAR[]"),
    ("side_b_card_levels", "UTINYINT[]"),
    ("side_b_deck_hash", "VARCHAR"),
    ("archive_member", "VARCHAR"),
    ("row_number", "BIGINT"),
)

CANONICAL_SCHEMA: tuple[SchemaColumnManifest, ...] = tuple(
    SchemaColumnManifest(name=name, physical_type=physical_type, nullable=False)
    for name, physical_type in _CANONICAL_COLUMNS
)


def utc_z_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    utc = value.astimezone(UTC)
    base = utc.strftime("%Y-%m-%dT%H:%M:%S")
    if utc.microsecond:
        fraction = f"{utc.microsecond:06d}".rstrip("0")
        return f"{base}.{fraction}Z"
    return f"{base}Z"


def _canonicalize(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc_z_timestamp(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PlayerId):
        return value.value
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in cast(tuple[Any, ...], value)]
    if isinstance(value, list):
        return [_canonicalize(item) for item in cast(list[Any], value)]
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in cast(dict[Any, Any], value).items()}
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return dumps(
        _canonicalize(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def deck_content_hash(
    card_ids: Sequence[str],
    card_forms: Sequence[str],
    card_levels: Sequence[int],
) -> str:
    if not (len(card_ids) == len(card_forms) == len(card_levels) == DECK_SIZE):
        raise ValueError("deck must have eight aligned cards")
    entries = sorted(
        f"{card_id}:{form}:{level}"
        for card_id, form, level in zip(card_ids, card_forms, card_levels, strict=True)
    )
    return sha256("|".join(entries).encode("utf-8")).hexdigest()


def canonical_schema_fingerprint() -> str:
    payload = [
        {"name": column.name, "physical_type": column.physical_type, "nullable": column.nullable}
        for column in CANONICAL_SCHEMA
    ]
    return sha256(dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def canonical_row_sort_key(row: "CanonicalBattleRow") -> tuple[datetime, str, str, int]:
    return (row.timestamp, row.fingerprint, row.archive_member, row.row_number)


def stream_logical_canonical_content_hash(rows: Iterable["CanonicalBattleRow"]) -> str:
    """SHA-256 of a JSON array of rows in the given order, matching dumps of the full list."""
    digest = sha256()
    digest.update(b"[")
    first = True
    for row in rows:
        if not first:
            digest.update(b",")
        digest.update(canonical_json_bytes(_canonicalize(row.model_dump(mode="python"))))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def logical_canonical_content_hash(rows: Sequence["CanonicalBattleRow"]) -> str:
    """SHA-256 of sorted canonical rows using the streaming JSON-array encoding."""
    return stream_logical_canonical_content_hash(sorted(rows, key=canonical_row_sort_key))


class CanonicalBattleRow(DomainModel):
    dataset_version: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    timestamp: datetime
    mode: Literal["Ranked1v1_NewArena"]
    balance_era_id: str = Field(min_length=1)
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
    archive_member: str = Field(min_length=1)
    row_number: int = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("battle timestamp must be timezone-aware")
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
        if any(level != ACCEPTED_CARD_LEVEL for level in card_levels):
            raise ValueError("accepted canonical rows require card level 16 on every card")
        expected = deck_content_hash(card_ids, card_forms, card_levels)
        if deck_hash != expected:
            raise ValueError("deck hash must match the aligned card identity and levels")
