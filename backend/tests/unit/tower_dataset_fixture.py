"""Normalized current-era battles with all released identities and recorded towers."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from attention_cache_fixture import FORMS, LEVELS, LOSE, WIN
from test_canonical_dataset import valid_row_payload

from clash_sos.domain.attention_dataset import TowerBattleRow
from clash_sos.domain.canonical_dataset import deck_content_hash
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_TOWER_CATALOG

VERSION = "official-tiny-v1"


def stamp(day: int) -> datetime:
    return datetime(2026, 9, day, tzinfo=UTC)


def tower_rows() -> tuple[TowerBattleRow, ...]:
    new_ids = ("ronin", "minion-giant", "elite-barbarians", "valkyrie", "berserker", "ice-wizard")
    new_forms = ("base", "base", "evolution", "hero", "hero", "hero")
    towers = tuple(entry.identity for entry in CURRENT_TOWER_CATALOG.entries)
    rows: list[TowerBattleRow] = []
    for index, day in enumerate((2, 4, 9, 10, 19, 25)):
        cards = (*new_ids, *WIN[:2]) if index == 0 else WIN
        forms = (*new_forms, *FORMS[:2]) if index == 0 else FORMS
        rows.append(
            TowerBattleRow.model_validate(
                valid_row_payload(
                    dataset_version=VERSION,
                    source_id="official-api",
                    balance_era_id="2026-09",
                    timestamp=stamp(day),
                    event_key=sha256(f"event-{index}".encode()).hexdigest(),
                    fingerprint=sha256(f"battle-{index}".encode()).hexdigest(),
                    side_a_card_ids=cards,
                    side_a_card_forms=forms,
                    side_a_deck_hash=deck_content_hash(cards, forms, LEVELS),
                    side_b_card_ids=LOSE,
                    side_b_card_forms=FORMS,
                    side_b_deck_hash=deck_content_hash(LOSE, FORMS, LEVELS),
                    archive_member="normalized.jsonl",
                    row_number=index,
                    side_a_tower=towers[index % 4],
                    side_b_tower=towers[(index + 1) % 4],
                    side_a_tower_level=16,
                    side_b_tower_level=16,
                )
            )
        )
    return tuple(rows)


def write_tower_source(path: Path, rows: tuple[TowerBattleRow, ...] | None = None) -> None:
    path.write_text("".join(row.model_dump_json() + "\n" for row in (rows or tower_rows())))
