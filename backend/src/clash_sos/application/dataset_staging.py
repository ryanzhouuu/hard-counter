"""Application workflow that stages adapted Kaggle v6 rows without publishing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from clash_sos.domain.canonical import BattleOutcome, BattleSide, DomainModel, RecordState
from clash_sos.domain.canonical_dataset import deck_content_hash
from clash_sos.domain.manifests import DatasetManifest
from clash_sos.domain.staged_dataset import (
    POPULATION_OBSERVATION_ISSUES,
    StagedBattleRow,
    UnadaptableRow,
)
from clash_sos.infrastructure.kaggle_v6.adapter import AdaptedKaggleRecord
from clash_sos.infrastructure.kaggle_v6.audit import audit_kaggle_v6_archive, write_dataset_manifest
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.staging_io import KaggleV6StagingError

__all__ = [
    "KaggleV6StagingError",
    "StagingConfig",
    "StagingResult",
    "adapted_to_staged_row",
    "adapted_to_unadaptable_row",
    "ensure_raw_audit_manifest",
]


class StagingConfig(DomainModel):
    memory_limit: str = Field(min_length=1, default="1GB")
    threads: int = Field(ge=1, default=2)
    chunk_size: int = Field(ge=1, default=8 * 1024 * 1024)
    batch_rows: int = Field(ge=1, default=10_000)
    parquet_row_group_rows: Literal[131072] = 131072


@dataclass(frozen=True)
class StagingResult:
    source_row_count: int
    staged_row_count: int
    unadaptable_row_count: int
    members: tuple[str, ...]
    staging_files: tuple[Path, ...]
    unadaptable_files: tuple[Path, ...]
    file_sha256: tuple[tuple[str, str], ...]
    logical_staging_sha256: str
    logical_unadaptable_sha256: str
    rows_per_second: float
    temp_disk_bytes: int


def _side_fields(
    side: BattleSide,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], str]:
    card_ids = tuple(card.card_id.value for card in side.deck.cards)
    card_forms = tuple(card.form.value for card in side.deck.cards)
    levels = side.card_levels
    deck_hash = deck_content_hash(card_ids, card_forms, levels)
    return card_ids, card_forms, levels, deck_hash


def adapted_to_staged_row(record: AdaptedKaggleRecord) -> StagedBattleRow:
    """Map an adapted record with a battle into a staging row."""
    if record.battle is None:
        raise KaggleV6StagingError("adapted record has no battle to stage")
    battle = record.battle
    observation_issues = tuple(
        sorted(
            (
                issue
                for issue in record.disposition.issues
                if issue in POPULATION_OBSERVATION_ISSUES
            ),
            key=lambda issue: issue.value,
        )
    )

    side_a_ids, side_a_forms, side_a_levels, side_a_hash = _side_fields(battle.side_a)
    side_b_ids, side_b_forms, side_b_levels, side_b_hash = _side_fields(battle.side_b)

    return StagedBattleRow(
        source_id=battle.source_id,
        timestamp=battle.timestamp,
        mode=battle.mode,
        balance_era_id=record.balance_era_id,
        outcome=BattleOutcome.SIDE_A_WIN,
        event_key=battle.event_key,
        fingerprint=battle.fingerprint,
        side_a_player_id=battle.side_a.player_id,
        side_b_player_id=battle.side_b.player_id,
        side_a_card_ids=side_a_ids,
        side_a_card_forms=side_a_forms,
        side_a_card_levels=side_a_levels,
        side_a_deck_hash=side_a_hash,
        side_b_card_ids=side_b_ids,
        side_b_card_forms=side_b_forms,
        side_b_card_levels=side_b_levels,
        side_b_deck_hash=side_b_hash,
        observation_issues=observation_issues,
        archive_member=record.location.archive_member,
        row_number=record.location.row_number,
    )


def adapted_to_unadaptable_row(record: AdaptedKaggleRecord) -> UnadaptableRow:
    """Map an unadaptable adapted record into a compact ledger row without detail text."""
    if record.battle is not None:
        raise KaggleV6StagingError("adapted record with a battle cannot be unadaptable")
    state = record.disposition.state
    if state not in {RecordState.INVALID, RecordState.QUARANTINED}:
        raise KaggleV6StagingError("unadaptable rows must be invalid or quarantined")
    if state is RecordState.INVALID:
        unadaptable_state = RecordState.INVALID
    else:
        unadaptable_state = RecordState.QUARANTINED
    return UnadaptableRow(
        archive_member=record.location.archive_member,
        row_number=record.location.row_number,
        state=unadaptable_state,
        issues=record.disposition.issues,
    )


def ensure_raw_audit_manifest(
    archive_path: Path,
    raw_manifest_path: Path,
    *,
    temp_directory: Path,
    config: StagingConfig,
    expected_archive_size: int | None,
    expected_archive_sha256: str | None,
) -> DatasetManifest:
    """Reuse or create the raw audit manifest after verifying archive identity."""
    size, digest = hash_file(archive_path, config.chunk_size)
    if expected_archive_size is not None and size != expected_archive_size:
        raise KaggleV6StagingError("archive size does not match the pinned Kaggle v6 source")
    if expected_archive_sha256 is not None and digest != expected_archive_sha256:
        raise KaggleV6StagingError("archive checksum does not match the pinned Kaggle v6 source")
    if raw_manifest_path.exists():
        manifest = DatasetManifest.model_validate_json(
            raw_manifest_path.read_text(encoding="utf-8")
        )
        if manifest.archive.size_bytes != size or manifest.archive.sha256 != digest:
            raise KaggleV6StagingError("raw audit manifest does not match archive")
        return manifest
    manifest = audit_kaggle_v6_archive(
        archive_path,
        temp_directory=temp_directory,
        memory_limit=config.memory_limit,
        threads=config.threads,
        chunk_size=config.chunk_size,
        expected_archive_size=expected_archive_size,
        expected_archive_sha256=expected_archive_sha256,
    )
    write_dataset_manifest(manifest, raw_manifest_path)
    return manifest
