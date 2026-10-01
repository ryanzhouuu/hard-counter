import json
from dataclasses import replace

from experiments.common.data_access import RoleAccess
from experiments.common.readiness import readiness
from experiments.mechanics.contracts import MechanicsCatalog, MechanicsEntry
from experiments.mechanics.load import synthetic_catalog
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
