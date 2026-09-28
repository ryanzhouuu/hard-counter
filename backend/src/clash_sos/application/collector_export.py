"""Export conflict-free local matches into the existing official snapshot boundary."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from clash_sos.domain.attention_dataset import TowerBattleRowV2
from clash_sos.domain.canonical_dataset import deck_content_hash
from clash_sos.infrastructure.clash_royale.collector_normalize import (
    CollectedBattle,
    CollectedSide,
)
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore

CURRENT_RANKED_MODE = "Ranked1v1_NewArena2"


@dataclass(frozen=True)
class ExportResult:
    """Count written matches and older-mode matches outside the v2 snapshot."""

    written: int
    skipped_mode: int


def _side_fields(side: CollectedSide, prefix: str) -> dict[str, object]:
    """Keep card IDs, forms, and levels aligned after winner-first orientation."""
    ids = tuple(identity.rsplit(":", 1)[0] for identity, _ in side.cards)
    forms = tuple(identity.rsplit(":", 1)[1] for identity, _ in side.cards)
    levels = tuple(level for _, level in side.cards)
    return {
        f"{prefix}_player_id": side.tag,
        f"{prefix}_card_ids": ids,
        f"{prefix}_card_forms": forms,
        f"{prefix}_card_levels": levels,
        f"{prefix}_deck_hash": deck_content_hash(ids, forms, levels),
        f"{prefix}_tower": side.tower,
        f"{prefix}_tower_level": side.tower_level,
    }


def _snapshot_row(
    battle: CollectedBattle,
    *,
    dataset_version: str,
    balance_era_id: str,
    filename: str,
    row_number: int,
) -> TowerBattleRowV2:
    """Verify the collector's normalized payload against the training row contract."""
    if battle.winner_tag is None or battle.exclusion_reason is not None:
        raise ValueError("only eligible decisive battles may be exported")
    winner = next(side for side in battle.sides if side.tag == battle.winner_tag)
    loser = next(side for side in battle.sides if side.tag != battle.winner_tag)
    return TowerBattleRowV2.model_validate(
        {
            "dataset_version": dataset_version,
            "source_id": "official-api",
            "timestamp": battle.timestamp,
            "mode": battle.mode,
            "balance_era_id": balance_era_id,
            "outcome": "side_a_win",
            "event_key": battle.event_key,
            "fingerprint": battle.variant_hash,
            **_side_fields(winner, "side_a"),
            **_side_fields(loser, "side_b"),
            "archive_member": filename,
            "row_number": row_number,
        }
    )


def export_collected_matches(
    store: CollectorStore,
    destination: Path,
    *,
    dataset_version: str,
    balance_era_id: str,
    start: datetime,
    end: datetime,
) -> ExportResult:
    """Atomically publish v2 JSONL without modifying retained collection state."""
    if not dataset_version or not balance_era_id:
        raise ValueError("dataset version and balance era are required")
    if destination.exists():
        raise ValueError("collector export destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.building-{uuid4().hex}")
    written = skipped_mode = 0
    try:
        with temporary.open("x") as output:
            for _, battle in store.export_battles(start, end):
                if battle.mode != CURRENT_RANKED_MODE:
                    skipped_mode += 1
                    continue
                row = _snapshot_row(
                    battle,
                    dataset_version=dataset_version,
                    balance_era_id=balance_era_id,
                    filename=destination.name,
                    row_number=written,
                )
                output.write(row.model_dump_json() + "\n")
                written += 1
        if not written:
            raise ValueError("no eligible current-ranked matches in export window")
        temporary.replace(destination)
        return ExportResult(written, skipped_mode)
    finally:
        temporary.unlink(missing_ok=True)
