"""Air readiness uses only relevant predicates and keeps conditional costs separate."""

from dataclasses import replace

import pytest
from experiments.mechanics.air_audit import air_audit, require_air_mechanics
from experiments.mechanics.contracts import MechanicsUnavailable, unknown
from experiments.mechanics.load import load_partial_catalog, synthetic_catalog


def test_complete_air_fixture_and_towers_ignore_unrelated_unknowns() -> None:
    catalog = synthetic_catalog()
    entries = dict(catalog.entries)
    for token, entry in entries.items():
        entries[token] = replace(
            entry, fields={**entry.fields, "area_damage": unknown("area_damage")}
        )
    catalog = replace(catalog, entries=entries)
    require_air_mechanics(catalog, list(entries))
    assert air_audit(catalog, [13, 14, 15, 16], include_cost=True) == {}


def test_unknown_and_verified_false_have_different_readiness() -> None:
    catalog = synthetic_catalog()
    token = 6
    assert air_audit(catalog, [token]) == {}
    entries = dict(catalog.entries)
    entries[token] = replace(
        entries[token], fields={**entries[token].fields, "targets_air": unknown("targets_air")}
    )
    catalog = replace(catalog, entries=entries)
    assert air_audit(catalog, [token]) == {"fixture-6:base": ("targets_air",)}
    with pytest.raises(MechanicsUnavailable, match="targets_air"):
        require_air_mechanics(catalog, [token])


def test_separate_spell_damage_control_and_cost_gates() -> None:
    catalog = synthetic_catalog()
    entry = catalog.entries[4]
    entries = {
        **catalog.entries,
        4: replace(
            entry,
            fields={
                **entry.fields,
                "spell_control": unknown("spell_control"),
                "deploy_cost": unknown("deploy_cost"),
            },
        ),
    }
    catalog = replace(catalog, entries=entries)
    assert air_audit(catalog, [4]) == {entry.identity: ("spell_control",)}
    assert air_audit(catalog, [4], include_cost=True) == {
        entry.identity: ("spell_control", "deploy_cost")
    }


def test_conditional_cost_never_requires_a_fixed_cost_or_silently_infers_airborne() -> None:
    catalog = load_partial_catalog()
    token = next(t for t, e in catalog.entries.items() if e.identity == "spirit-empress:base")
    gaps = air_audit(catalog, [token], include_cost=True)["spirit-empress:base"]
    assert "airborne" in gaps
    assert "deploy_cost" not in gaps
    assert "conditional_cost" not in gaps
    with pytest.raises(MechanicsUnavailable, match="spirit-empress"):
        require_air_mechanics(catalog, [token])


def test_current_catalog_cannot_pass_a_full_air_gate() -> None:
    catalog = load_partial_catalog()
    with pytest.raises(MechanicsUnavailable, match="air mechanics unavailable"):
        require_air_mechanics(catalog, list(catalog.entries))
