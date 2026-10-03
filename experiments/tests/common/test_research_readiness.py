import json
from dataclasses import replace
from datetime import timedelta, timezone
from typing import cast

import pytest
from experiments.common.data_access import RoleAccess
from experiments.common.readiness import readiness
from experiments.mechanics.contracts import MechanicsCatalog, MechanicsEntry, unknown
from experiments.mechanics.load import load_partial_catalog, synthetic_catalog
from experiments.tests.common.research_fixture import protocol, rows


def test_readiness_uses_covariates_and_provenance_without_outcomes() -> None:
    population = rows()
    original = protocol(population)
    catalog = synthetic_catalog()
    result = readiness(RoleAccess(original, population), catalog)
    changed = tuple(replace(row, label=1 - row.label) for row in population)
    assert result == readiness(RoleAccess(original, changed), catalog)
    json.dumps(result, allow_nan=False)
    roles = result["roles"]
    assert isinstance(roles, dict)
    assert roles["refit"]["row_count"] == 12
    assert roles["calibration"]["row_count"] == 6
    assert roles["development"]["row_count"] == 6
    assert roles["refit"]["tower_support"]
    assert roles["refit"]["form_support"]
    assert roles["development"]["complete_player_windows"] == {"10": 0, "25": 0}
    overall = result["population"]
    assert isinstance(overall, dict)
    assert overall["player_count"] == 12
    assert overall["max_player_match_fraction"] == 5 / 24
    assert overall["repeated_player_count"] == 12
    assert overall["players_with_multiple_lineups"] == 12


def test_readiness_keeps_missing_mechanics_and_unavailable_patterns_visible() -> None:
    population = rows()
    catalog = synthetic_catalog()
    entries = dict(catalog.entries)
    original_entry = entries[0]
    entries[0] = MechanicsEntry(original_entry.identity, "card", {})
    missing = MechanicsCatalog(catalog.version, catalog.era_start, catalog.era_end, entries, True)
    report = readiness(RoleAccess(protocol(population), population), missing)
    mechanics = report["mechanics"]
    assert isinstance(mechanics, dict)
    assert mechanics["unknown_field_count"] > 0
    patterns = report["patterns"]
    assert isinstance(patterns, dict)
    assert patterns["status"] == "unavailable"
    assert patterns["disabled_reasons"]


def test_readiness_reports_cost_gaps_separately_from_air_predicates() -> None:
    population = rows()
    catalog = synthetic_catalog()
    entry = catalog.for_token(2)
    changed = replace(entry, fields={**entry.fields, "deploy_cost": unknown("deploy_cost")})
    catalog = replace(catalog, entries={**catalog.entries, 2: changed})
    report = readiness(RoleAccess(protocol(population), population), catalog)
    mechanics = cast(dict[str, object], report["mechanics"])
    audits = cast(dict[str, dict[str, object]], mechanics["air_audits"])
    assert audits["static"]["status"] == audits["conditional"]["status"] == "available"
    assert audits["conditional_cost"]["gaps"] == {entry.identity: ("deploy_cost",)}
    assert audits["conditional_cost"]["status"] == "unavailable"


def test_air_readiness_cannot_pass_when_a_population_identity_is_missing() -> None:
    population = rows()
    catalog = synthetic_catalog()
    catalog = replace(catalog, entries={t: e for t, e in catalog.entries.items() if t != 13})
    report = readiness(RoleAccess(protocol(population), population), catalog)
    mechanics = cast(dict[str, object], report["mechanics"])
    audits = cast(dict[str, dict[str, object]], mechanics["air_audits"])
    assert all(audit["status"] == "unavailable" for audit in audits.values())


@pytest.mark.parametrize("offset", [-5, 14])
def test_era_checks_and_daily_covariates_use_utc(offset: int) -> None:
    catalog = replace(load_partial_catalog(), era_start="2026-01-01", era_end="2026-01-01")
    tower_tokens = [t for t, entry in catalog.entries.items() if entry.kind == "tower"]
    population = tuple(
        replace(row, tokens=tuple((*side[:8], tower_tokens[side[8] - 13]) for side in row.tokens))
        for row in rows()
    )
    expected = readiness(RoleAccess(protocol(population), population), catalog)
    local = timezone(timedelta(hours=offset))
    shifted = tuple(
        replace(row, key=(row.key[0].astimezone(local), *row.key[1:])) for row in population
    )
    actual = readiness(RoleAccess(protocol(shifted), shifted), catalog)
    assert actual == expected
    mechanics = cast(dict[str, object], actual["mechanics"])
    assert mechanics["compatibility_issues"] == []
    outside = replace(catalog, era_start="2026-01-02", era_end="2026-01-02")
    rejected = readiness(RoleAccess(protocol(shifted), shifted), outside)
    issues = cast(dict[str, object], rejected["mechanics"])["compatibility_issues"]
    assert "population battle dates exceed verified mechanics era" in cast(list[str], issues)
