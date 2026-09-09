"""Unified per-row disposition ledger distinct from provisional adapter states.

Staging observations are not final. This contract is the published-shape row
written after identity grouping. Unadaptable rows keep location and issues only;
adaptable rows also carry provenance and a compact representative pointer.
"""

from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from clash_sos.domain.canonical import (
    BattleOutcome,
    DomainModel,
    PlayerId,
    RecordIssue,
    RecordState,
)
from clash_sos.domain.manifests import SchemaColumnManifest, Sha256
from clash_sos.domain.processed_manifest import (
    INGESTION_ISSUES,
    RECORD_STATES,
    DispositionSummary,
    IssueCount,
    StateCount,
)

DISPOSITION_SCHEMA_VERSION = "kaggle-v6-disposition-schema:v1"

_IDENTITY_FIELDS = (
    "source_id",
    "timestamp",
    "mode",
    "outcome",
    "event_key",
    "fingerprint",
    "side_a_player_id",
    "side_b_player_id",
)

_DISPOSITION_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("archive_member", "VARCHAR", False),
    ("row_number", "BIGINT", False),
    ("state", "VARCHAR", False),
    ("issues", "VARCHAR[]", False),
    ("source_id", "VARCHAR", True),
    ("timestamp", "TIMESTAMP WITH TIME ZONE", True),
    ("mode", "VARCHAR", True),
    ("balance_era_id", "VARCHAR", True),
    ("outcome", "VARCHAR", True),
    ("event_key", "VARCHAR", True),
    ("fingerprint", "VARCHAR", True),
    ("side_a_player_id", "VARCHAR", True),
    ("side_b_player_id", "VARCHAR", True),
    ("representative_archive_member", "VARCHAR", True),
    ("representative_row_number", "BIGINT", True),
)

DISPOSITION_SCHEMA: tuple[SchemaColumnManifest, ...] = tuple(
    SchemaColumnManifest(name=name, physical_type=physical_type, nullable=nullable)
    for name, physical_type, nullable in _DISPOSITION_COLUMNS
)

IngestionState = Literal[
    RecordState.INVALID,
    RecordState.QUARANTINED,
    RecordState.UNSUPPORTED,
    RecordState.VALID,
]


class DispositionRow(DomainModel):
    """One final disposition for a single source location after grouping."""

    archive_member: str = Field(min_length=1)
    row_number: int = Field(ge=0)
    state: IngestionState
    issues: tuple[RecordIssue, ...] = ()
    source_id: str | None = None
    timestamp: datetime | None = None
    mode: str | None = None
    balance_era_id: str | None = None
    outcome: Literal[BattleOutcome.SIDE_A_WIN] | None = None
    event_key: Sha256 | None = None
    fingerprint: Sha256 | None = None
    side_a_player_id: PlayerId | None = None
    side_b_player_id: PlayerId | None = None
    representative_archive_member: str | None = None
    representative_row_number: int | None = None

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamp must be timezone-aware")
        return value

    @field_validator("balance_era_id")
    @classmethod
    def validate_balance_era_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("balance_era_id must be non-empty when present")
        return value

    @field_validator("representative_archive_member")
    @classmethod
    def validate_representative_member(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("representative_archive_member must be non-empty when present")
        return value

    @field_validator("issues")
    @classmethod
    def validate_issues(cls, value: tuple[RecordIssue, ...]) -> tuple[RecordIssue, ...]:
        if len(value) != len(set(value)):
            raise ValueError("issues must be unique")
        sorted_values = tuple(sorted(value, key=lambda issue: issue.value))
        if value != sorted_values:
            raise ValueError("issues must be sorted by issue code")
        return value

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        if self.state is RecordState.VALID and self.issues:
            raise ValueError("valid records cannot carry issues")
        if self.state is not RecordState.VALID and not self.issues:
            raise ValueError("non-valid records must identify at least one issue")
        identity_values = tuple(getattr(self, name) for name in _IDENTITY_FIELDS)
        identity_present = all(value is not None for value in identity_values)
        identity_absent = all(value is None for value in identity_values)
        if not identity_present and not identity_absent:
            raise ValueError("identity fields must be present together or all absent")
        if identity_absent and self.balance_era_id is not None:
            raise ValueError("identity fields must be present together or all absent")
        representative_paired = (self.representative_archive_member is None) == (
            self.representative_row_number is None
        )
        if not representative_paired:
            raise ValueError("representative location fields must be paired")
        if identity_absent and self.representative_archive_member is not None:
            raise ValueError("unadaptable rows cannot carry a representative")
        if identity_absent and self.state not in {RecordState.INVALID, RecordState.QUARANTINED}:
            raise ValueError("unadaptable rows must be invalid or quarantined")
        return self


def summarize_disposition_rows(
    rows: Sequence[DispositionRow],
    *,
    source_row_count: int,
) -> DispositionSummary:
    """Build a processed-manifest disposition summary that must cover every source row."""
    if len(rows) != source_row_count:
        raise ValueError("disposition row count must equal source_row_count")
    state_counts = Counter(row.state for row in rows)
    issue_counts: Counter[RecordIssue] = Counter()
    for row in rows:
        issue_counts.update(row.issues)
    return DispositionSummary(
        source_row_count=source_row_count,
        states=tuple(StateCount(state=state, count=state_counts[state]) for state in RECORD_STATES),
        issues=tuple(
            IssueCount(issue=issue, count=issue_counts[issue]) for issue in INGESTION_ISSUES
        ),
        duplicate_row_count=issue_counts[RecordIssue.DUPLICATE_BATTLE],
        conflict_row_count=issue_counts[RecordIssue.CONFLICTING_BATTLE],
    )
