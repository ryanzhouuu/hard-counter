"""Split identity schema and temporal assignment for processed datasets."""

from datetime import datetime
from typing import Literal

from clash_sos.domain.manifests import SchemaColumnManifest

_SPLIT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("timestamp", "TIMESTAMP WITH TIME ZONE"),
    ("fingerprint", "VARCHAR"),
    ("archive_member", "VARCHAR"),
    ("row_number", "BIGINT"),
    ("partition", "VARCHAR"),
)

SPLIT_SCHEMA: tuple[SchemaColumnManifest, ...] = tuple(
    SchemaColumnManifest(name=name, physical_type=physical_type, nullable=False)
    for name, physical_type in _SPLIT_COLUMNS
)

VERIFICATION_CHECK_IDS: tuple[str, ...] = (
    "canonical_identity",
    "canonical_schema",
    "disposition_reconciliation",
    "player_disjoint_bridge_count",
    "player_disjoint_no_leakage",
    "temporal_boundaries",
    "temporal_coverage",
)


def assign_temporal_partition(
    timestamp: datetime, *, train_end: datetime, validation_end: datetime
) -> Literal["train", "validation", "test"]:
    """Assign a battle using half-open UTC cutovers. Raises on naive timestamps."""
    for value in (timestamp, train_end, validation_end):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("temporal split timestamps must be timezone-aware")
    if train_end >= validation_end:
        raise ValueError("train_end must be earlier than validation_end")
    if timestamp < train_end:
        return "train"
    if timestamp < validation_end:
        return "validation"
    return "test"
