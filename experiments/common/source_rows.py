"""Frozen official-row encoding and population inventories shared by source adapters."""

from datetime import datetime
from hashlib import sha256

from clash_sos.domain.attention_dataset import TowerBattleRow, TowerBattleRowV2
from clash_sos.domain.attention_protocol import digest_row_keys
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.matchup_baseline import should_mirror_sides
from experiments.common.cache import oriented_digest
from experiments.common.contracts import FileRecord, PopulationIdentity
from experiments.common.data_access import ResearchRow, validate_rows


def encode_official_row(
    source: TowerBattleRow | TowerBattleRowV2,
    schema: AttentionCardSchema,
    mirror_seed: int,
) -> ResearchRow:
    expected_mode = (
        "Ranked1v1_NewArena2"
        if schema.canonical_schema_version == "official-ranked16-schema:v2"
        else "Ranked1v1_NewArena"
    )
    if source.mode != expected_mode or source.balance_era_id != schema.balance_era_id:
        raise ValueError("official row mode or era is incompatible with frozen schema")
    if schema.tower_catalog is None or mirror_seed < 0:
        raise ValueError("official encoding requires frozen towers and nonnegative mirror seed")
    tokens_a = schema.encode_side(
        tuple(
            f"{card}:{form}"
            for card, form in zip(source.side_a_card_ids, source.side_a_card_forms, strict=True)
        ),
        levels=source.side_a_card_levels,
        tower=source.side_a_tower,
        tower_level=source.side_a_tower_level,
    )
    tokens_b = schema.encode_side(
        tuple(
            f"{card}:{form}"
            for card, form in zip(source.side_b_card_ids, source.side_b_card_forms, strict=True)
        ),
        levels=source.side_b_card_levels,
        tower=source.side_b_tower,
        tower_level=source.side_b_tower_level,
    )
    row = ResearchRow(
        (source.timestamp, source.fingerprint, source.archive_member, source.row_number),
        source.event_key,
        source.side_a_player_id.value,
        source.side_b_player_id.value,
        1,
        (tokens_a, tokens_b),
    )
    return row.swapped() if should_mirror_sides(source.fingerprint, seed=mirror_seed) else row


def event_mapping_digest(rows: tuple[ResearchRow, ...]) -> str:
    validate_rows(rows)
    return sha256(canonical_json_bytes(tuple((row.key, row.event_key) for row in rows))).hexdigest()


def population_identity(
    rows: tuple[ResearchRow, ...],
    files: tuple[FileRecord, ...],
    schema: AttentionCardSchema,
    mirror_seed: int,
    *,
    start: datetime,
    end: datetime,
) -> PopulationIdentity:
    validate_rows(rows)
    return PopulationIdentity.model_validate(
        {
            "snapshot_files": files,
            "row_keys_sha256": digest_row_keys(row.key for row in rows),
            "event_mapping_sha256": event_mapping_digest(rows),
            "oriented_sha256": oriented_digest(rows),
            "mode": "Ranked1v1_NewArena2"
            if schema.canonical_schema_version == "official-ranked16-schema:v2"
            else "Ranked1v1_NewArena",
            "era": schema.balance_era_id,
            "mirror_seed": mirror_seed,
            "start": start,
            "end": end,
            "catalog_sha256": schema.catalog_sha256,
        }
    )
