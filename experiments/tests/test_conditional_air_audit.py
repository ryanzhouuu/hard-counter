"""Hybrid readiness requires complete alternative channels without guessing deployment state."""

from dataclasses import replace

import pytest
from experiments.mechanics.air_audit import air_audit, require_air_mechanics
from experiments.mechanics.conditional_air import has_channel
from experiments.mechanics.contracts import MechanicsCatalog, MechanicsUnavailable, Value, unknown
from experiments.mechanics.load import load_partial_catalog


def changed_catalog(identity: str, field: str, value: Value = None) -> MechanicsCatalog:
    catalog = load_partial_catalog()
    token = next(t for t, e in catalog.entries.items() if e.identity == identity)
    entry = catalog.for_token(token)
    original = entry.field(field)
    altered = unknown(field) if value is None else replace(original, value=value)
    return replace(
        catalog,
        entries={**catalog.entries, token: replace(entry, fields={**entry.fields, field: altered})},
    )


def test_documented_hybrid_passes_only_the_conditional_aware_gate() -> None:
    catalog = load_partial_catalog()
    tokens = list(catalog.entries)
    assert air_audit(catalog, tokens) == {"spirit-empress:base": ("targets_air", "airborne")}
    assert air_audit(catalog, tokens, include_conditional=True) == {}
    require_air_mechanics(catalog, tokens, include_conditional=True)
    with pytest.raises(MechanicsUnavailable, match="spirit-empress"):
        require_air_mechanics(catalog, tokens)


@pytest.mark.parametrize(
    "field",
    [
        "conditional_airborne",
        "conditional_air_damage",
        "airborne_condition",
        "targets_air_condition",
        "conditional_air_trigger",
        "conditional_air_response_scope",
    ],
)
def test_a_hybrid_label_alone_cannot_bypass_missing_evidence(field: str) -> None:
    catalog = changed_catalog("spirit-empress:base", field)
    gaps = air_audit(catalog, list(catalog.entries), include_conditional=True)
    assert field in gaps["spirit-empress:base"]
    with pytest.raises(MechanicsUnavailable, match=field):
        require_air_mechanics(catalog, list(catalog.entries), include_conditional=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conditional_airborne", False),
        ("conditional_air_damage", False),
        ("airborne_condition", "paid_ability"),
        ("targets_air_condition", "paid_ability"),
        ("conditional_air_trigger", "paid_ability"),
        ("conditional_air_response_scope", "moving_area"),
    ],
)
def test_contradictory_hybrid_conditions_are_reported(field: str, value: Value) -> None:
    catalog = changed_catalog("spirit-empress:base", field, value)
    assert "spirit-empress:base" in air_audit(
        catalog, list(catalog.entries), include_conditional=True
    )


def test_unknown_ordinary_targeting_is_not_covered_by_an_ability() -> None:
    catalog = changed_catalog("ice-golem:hero", "targets_air")
    assert (
        "targets_air"
        in air_audit(catalog, list(catalog.entries), include_conditional=True)["ice-golem:hero"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "conditional_air_trigger",
        "conditional_air_response_scope",
        "conditional_air_control_kind",
        "ability_usage",
    ],
)
def test_recorded_ability_channels_require_their_metadata(field: str) -> None:
    catalog = changed_catalog("ice-golem:hero", field)
    assert (
        field
        in air_audit(catalog, list(catalog.entries), include_conditional=True)["ice-golem:hero"]
    )


def test_ability_cost_and_deployment_cost_remain_separate_gates() -> None:
    catalog = changed_catalog("ice-golem:hero", "ability_cost")
    token = next(t for t, e in catalog.entries.items() if e.identity == "ice-golem:hero")
    assert air_audit(catalog, [token], include_conditional=True) == {}
    assert (
        "ability_cost"
        in air_audit(catalog, [token], include_conditional=True, include_cost=True)[
            "ice-golem:hero"
        ]
    )
    empress = next(t for t, e in catalog.entries.items() if e.identity == "spirit-empress:base")
    assert air_audit(catalog, [empress], include_conditional=True, include_cost=True) == {}
    towers = [t for t, e in catalog.entries.items() if e.kind == "tower"]
    assert air_audit(catalog, towers, include_conditional=True, include_cost=True) == {}


@pytest.mark.parametrize("field", ["conditional_air_trigger", "ability_usage", "ability_cost"])
def test_potential_channels_need_metadata_but_do_not_imply_known_cost(field: str) -> None:
    catalog = changed_catalog("wizard:hero", field)
    entry = next(e for e in catalog.entries.values() if e.identity == "wizard:hero")
    assert has_channel(entry, "conditional_airborne") is (field == "ability_cost")
    with pytest.raises(ValueError, match="unregistered"):
        has_channel(entry, "airborne")
