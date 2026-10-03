"""Native deterministic populations exercise study contracts without collector data."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from clash_sos.domain.attention_protocol import AttentionProtocol, AttentionSlice, digest_row_keys
from clash_sos.domain.attention_schema import (
    AttentionCardSchema,
    AttentionModelConfig,
    build_attention_schema,
)
from clash_sos.domain.canonical import CardForm, CardId, CardRef
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.card_attributes import CardAttribute, CardAttributeTable
from clash_sos.domain.card_catalog import CardCatalog, CardCatalogEntry
from clash_sos.domain.tower_catalog import TowerCatalog, TowerEntry
from experiments.common.data_access import ResearchRow, RoleAccess
from experiments.mechanics.contracts import MechanicField, MechanicsCatalog, MechanicsEntry
from experiments.mechanics.load import bind_tokens, synthetic_catalog

START = datetime(2026, 9, 1, tzinfo=UTC)


def _schema() -> tuple[MechanicsCatalog, AttentionCardSchema]:
    catalog = synthetic_catalog()
    entries = dict(catalog.entries)
    entries[8] = replace(entries[8], identity="mirror:base")
    form_fields = dict(entries[10].fields)
    form_fields["multi_unit"] = replace(form_fields["multi_unit"], value=True)
    entries[10] = replace(entries[10], fields=form_fields)
    for token in range(17, 21):
        entries[token] = replace(entries[6], identity=f"fixture-{token}:base")
    entries[9] = replace(
        entries[9],
        fields={
            **entries[9].fields,
            **{
                name: MechanicField(value, unit, "synthetic")
                for name, value, unit in (
                    ("conditional_airborne", True, "flag"),
                    ("conditional_air_damage", True, "flag"),
                    ("conditional_air_trigger", "available_elixir_at_least_6", "category"),
                    ("conditional_air_response_scope", "flying_form_attack", "category"),
                )
            },
        },
    )
    catalog = replace(catalog, entries=entries)
    cards: list[CardCatalogEntry] = []
    attributes: dict[str, CardAttribute] = {}
    towers: list[TowerEntry] = []
    for token, entry in catalog.entries.items():
        card_id, form = entry.identity.rsplit(":", 1)
        if entry.kind == "tower":
            towers.append(
                TowerEntry(
                    tower_id=CardId(card_id),
                    name=card_id,
                    api_id=159000000 + token,
                    max_api_level=16,
                )
            )
        else:
            cards.append(
                CardCatalogEntry(
                    len(cards),
                    entry.identity,
                    CardRef(card_id=CardId(card_id), form=CardForm(form)),
                )
            )
            if entry.base_identity is None:
                attributes[card_id] = _attribute(entry)
    schema = build_attention_schema(
        CardCatalog("synthetic-cards:v1", tuple(cards)).serialize(),
        attributes=CardAttributeTable("synthetic-attributes:v1", attributes),
        network=AttentionModelConfig(
            embedding_width=8,
            attention_heads=2,
            within_deck_blocks=1,
            cross_deck_blocks=1,
            feed_forward_width=16,
            neural_component=False,
        ),
        balance_era_id="synthetic:v1",
        tower_catalog=TowerCatalog(catalog_version="synthetic-towers:v1", entries=tuple(towers)),
        official_schema_version="official-ranked16-schema:v2",
    )
    return catalog, schema


def _attribute(entry: MechanicsEntry) -> CardAttribute:
    """Production attributes are a separate synthetic representation of conditional costs."""
    roles = frozenset(
        role
        for role, predicate in (
            ("win_condition", "building_targeting"),
            ("building", "defensive_building"),
            ("spell", "spell"),
            ("cycle", "ordinary_cycle"),
            ("air_defense", "targets_air"),
            ("bait", "multi_unit"),
        )
        if entry.flag(predicate)
    )
    cost = entry.number("deploy_cost")
    return CardAttribute(None if entry.identity == "mirror:base" else int(cost or 3), roles)


def _protocol(rows: tuple[ResearchRow, ...], schema: AttentionCardSchema) -> AttentionProtocol:
    count = len(rows)
    step = count // 4
    final_end = rows[-1].key[0] + timedelta(minutes=1)

    def part(first: int, stop: int, training: bool) -> AttentionSlice:
        return AttentionSlice(
            partition="train" if training else "validation",
            start=rows[first].key[0],
            end=rows[stop].key[0] if stop < count else final_end,
            row_count=stop - first,
            row_keys_sha256=digest_row_keys(r.key for r in rows[first:stop]),
        )

    canonical = sha256(
        canonical_json_bytes(
            [[row.event_key, row.player_a, row.player_b, row.label, row.tokens] for row in rows]
        )
    ).hexdigest()
    return AttentionProtocol(
        family="temporal",
        dataset_version="synthetic-smoke:v1",
        balance_era_id="synthetic:v1",
        processed_manifest_sha256=canonical,
        canonical_sha256=canonical,
        split_sha256=digest_row_keys(r.key for r in rows),
        split_file="splits-temporal.parquet",
        encoding_sha256=schema.fingerprint(),
        mirror_seed=0,
        selection_fit=part(0, step, True),
        watch=part(step, 2 * step, True),
        refit=part(0, 2 * step, True),
        calibration=part(2 * step, 3 * step, False),
        development=part(3 * step, count, False),
    )


def synthetic_smoke_population(
    count: int = 128,
) -> tuple[RoleAccess, MechanicsCatalog, AttentionCardSchema]:
    """Four repeated matchup contexts have deterministic, nonseparable outcome proportions."""
    if type(count) is not int or count < 64 or count % 32:
        raise ValueError("synthetic row count must be a multiple of 32 and at least 64")
    catalog, schema = _schema()
    standard = (0, 1, 4, 5, 6, 7, 8, 9)
    form = (0, 1, 4, 5, 6, 8, 9, 10)
    sole_answer = (2, 3, 5, 6, 17, 18, 19, 20)
    independent_answer = (2, 3, 5, 11, 17, 18, 19, 20)
    contexts = (
        (standard, sole_answer),
        (form, sole_answer),
        (form, independent_answer),
        (standard, independent_answer),
    )
    rows: list[ResearchRow] = []
    for index in range(count):
        context = index % 4
        deck_a, deck_b = contexts[context]
        sides = tuple(
            schema.encode_side(
                [catalog.entries[token].identity for token in deck],
                tower=catalog.entries[13 + (index // 16 + side) % 4].identity,
                tower_level=16,
            )
            for side, deck in enumerate((deck_a, deck_b))
        )
        label = int((index // 4) % 4 < (2, 3, 2, 1)[context])
        player_a = index % 8
        player_b = (player_a + 1 + (index // 8) % 7) % 8
        row = ResearchRow(
            (START + timedelta(minutes=index // 2), f"{index:064x}", "synthetic", index),
            f"synthetic-event-{index}",
            f"synthetic-player-{player_a}",
            f"synthetic-player-{player_b}",
            label,
            (sides[0], sides[1]),
        )
        rows.append(row.swapped() if index % 2 else row)
    population = tuple(rows)
    return (
        RoleAccess(_protocol(population, schema), population),
        bind_tokens(catalog, schema.identity_vocab),
        schema,
    )
