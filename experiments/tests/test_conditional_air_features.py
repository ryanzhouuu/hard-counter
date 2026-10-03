"""Conditional air features count potential channels, preserve restrictions, and obey symmetries."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from experiments.common.components import components
from experiments.common.contracts import StudyConfig
from experiments.common.synthetic import synthetic_smoke_population
from experiments.matchup_features.features import extract
from experiments.mechanics.contracts import MechanicField, MechanicsCatalog, unknown
from experiments.mechanics.load import load_partial_catalog
from hypothesis import given
from hypothesis import strategies as st


def population(identity: str) -> tuple[list[list[int]], MechanicsCatalog]:
    catalog = load_partial_catalog()
    tokens = {e.identity: t for t, e in catalog.entries.items()}
    neutral = [
        "knight",
        "goblins",
        "skeletons",
        "barbarians",
        "mini-pekka",
        "giant",
        "ice-golem",
        "prince",
    ]
    left = [tokens[f"{n}:base"] for n in neutral]
    right = [tokens[f"{n}:base"] for n in ["minions", "bats", *neutral[:6]]]
    left[0] = tokens[identity]
    return [
        [*left, tokens["tower-princess:tower"]],
        [*right, tokens["tower-princess:tower"]],
    ], catalog


@pytest.mark.parametrize(
    ("identity", "flight", "damage", "control", "reflection", "additional"),
    [
        ("spirit-empress:base", 1, 1, 0, 0, 1),
        ("wizard:hero", 1, 1, 1, 0, 0),
        ("ice-golem:hero", 0, 1, 1, 0, 1),
        ("giant:hero", 0, 1, 1, 0, 1),
        ("mighty-miner:champion", 0, 1, 1, 0, 1),
        ("monk:champion", 0, 0, 0, 1, 1),
    ],
)
def test_candidate_counts_separate_channels_and_do_not_duplicate_ordinary_answers(
    identity: str, flight: int, damage: int, control: int, reflection: int, additional: int
) -> None:
    pair, catalog = population(identity)
    result = extract(pair, catalog, response=False, cost=False, conditional_air=True)
    values = dict(zip(result.names, result.values, strict=True))
    for name, expected in zip(
        ("airborne", "air_damage", "air_control", "air_reflection"),
        (flight, damage, control, reflection),
        strict=True,
    ):
        assert values[f"conditional_air.{name}"] == expected
    assert values["conditional_air.additional_response_cards"] == additional
    assert values["conditional_air.air_damage.opposing_ordinary_threats"] == damage * 2
    assert values["conditional_air.air_reflection.opposing_ordinary_threats"] == reflection * 2
    baseline = extract(pair, catalog)
    combined = extract(pair, catalog, conditional_air=True)
    assert combined.names[: len(baseline.names)] == baseline.names
    assert combined.values[: len(baseline.values)] == baseline.values
    assert combined.formulas[: len(baseline.formulas)] == baseline.formulas


@pytest.mark.parametrize(
    ("identity", "scope", "control"),
    [
        ("spirit-empress:base", "flying_form_attack", None),
        ("wizard:hero", "enhanced_attack_area", "pull"),
        ("ice-golem:hero", "moving_area", "slow"),
        ("giant:hero", "selected_troop_ground_only_splash", "throw_and_stun"),
        ("mighty-miner:champion", "departure_bomb_area", "knockback"),
        ("monk:champion", "eligible_incoming_projectiles", None),
    ],
)
def test_registered_scope_and_control_survive_extraction(
    identity: str, scope: str, control: str | None
) -> None:
    pair, catalog = population(identity)
    result = extract(pair, catalog, conditional_air=True)
    values = dict(zip(result.names, result.values, strict=True))
    assert values[f"conditional_air.scope.{scope}"] == 1
    if control is not None:
        assert values[f"conditional_air.control.{control}"] == 1
    trigger = "available_elixir_at_least_6" if identity == "spirit-empress:base" else "paid_ability"
    assert values[f"conditional_air.trigger.{trigger}"] == 1


def test_unknown_capability_and_incomplete_trigger_are_not_silent_absence() -> None:
    pair, catalog = population("wizard:hero")
    token = pair[0][0]
    entry = catalog.for_token(token)
    altered = replace(
        entry,
        fields={**entry.fields, "conditional_air_trigger": unknown("conditional_air_trigger")},
    )
    incomplete = replace(catalog, entries={**catalog.entries, token: altered})
    result = extract(pair, incomplete, conditional_air=True)
    values = dict(zip(result.names, result.values, strict=True))
    assert values["conditional_air.airborne"] == 0
    original = extract(pair, catalog, conditional_air=True)
    assert (
        values["conditional_air.airborne.unavailable"]
        == dict(zip(original.names, original.values, strict=True))[
            "conditional_air.airborne.unavailable"
        ]
        + 1
    )
    knight = pair[1][2]
    entry = catalog.for_token(knight)
    false = replace(
        entry,
        fields={
            **entry.fields,
            "conditional_air_reflection": MechanicField(False, "flag", "synthetic"),
        },
    )
    explicit = replace(catalog, synthetic=True, entries={**catalog.entries, knight: false})
    observed = extract(pair, explicit, conditional_air=True)
    assert observed.values != original.values


@given(st.permutations(range(8)), st.permutations(range(8)))
def test_conditional_features_preserve_permutation_swap_and_identical_decks(
    left_order: list[int], right_order: list[int]
) -> None:
    pair, catalog = population("giant:hero")
    original = extract(pair, catalog, conditional_air=True)
    permuted = [
        [*[pair[0][i] for i in left_order], pair[0][8]],
        [*[pair[1][i] for i in right_order], pair[1][8]],
    ]
    assert extract(permuted, catalog, conditional_air=True) == original
    swapped = extract(pair[::-1], catalog, conditional_air=True)
    assert swapped.names == original.names and swapped.formulas == original.formulas
    assert swapped.values == tuple(-v for v in original.values)
    assert not any(extract([pair[0], pair[0]], catalog, conditional_air=True).values)


def test_registered_a4_builds_nonconstant_synthetic_features() -> None:
    config = StudyConfig.model_validate_json(
        Path("experiments/configs/response-cycle.json").read_bytes()
    )
    variant = next(v for v in config.variants if v.variant_id == "A4")
    access, catalog, schema = synthetic_smoke_population()
    rows = access.read("refit", "fit")
    recipe = components(config, variant, catalog, schema, rows[0])
    matrix = recipe.builder(rows, rows)
    columns = [i for i, name in enumerate(recipe.names) if name.startswith("conditional_air.")]
    assert columns and np.isfinite(matrix).all()
    assert np.any(np.ptp(matrix[:, columns], axis=0) > 0)


def test_missing_ability_cost_does_not_erase_potential_capability() -> None:
    pair, catalog = population("wizard:hero")
    token = pair[0][0]
    entry = catalog.for_token(token)
    changed = replace(entry, fields={**entry.fields, "ability_cost": unknown("ability_cost")})
    catalog = replace(catalog, entries={**catalog.entries, token: changed})
    result = extract(pair, catalog, conditional_air=True)
    values = dict(zip(result.names, result.values, strict=True))
    assert values["conditional_air.airborne"] == 1
    assert values["conditional_air.ability_cost"] == 0
    assert values["conditional_air.ability_cost_missing"] == 1
