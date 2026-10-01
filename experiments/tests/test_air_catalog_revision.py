"""Follow-up evidence extends the catalog without rewriting reviewed snapshots."""

from hashlib import sha256
from pathlib import Path

from experiments.matchup_features.responses import answers
from experiments.mechanics.air_audit import air_audit
from experiments.mechanics.load import load, load_partial_catalog


def test_original_october_snapshot_is_frozen_and_new_sources_keep_unknowns() -> None:
    path = Path("experiments/mechanics/inputs/2026-10-01-partial.json")
    assert sha256(path.read_bytes()).hexdigest() == (
        "89705936311adc913b8e35c02fdffefec7a7e75d4b93157e16cfd3b04bbaf3bf"
    )
    original = load(path)
    catalog = load_partial_catalog()
    assert original.digest != catalog.digest
    entries = {entry.identity: entry for entry in catalog.entries.values()}
    assert entries["rage:base"].flag("spell_damage")
    assert entries["rage:base"].flag("targets_air")
    assert answers(entries["rage:base"], "airborne")
    assert entries["freeze:base"].flag("spell_control")
    assert entries["freeze:base"].field("targets_air").status == "user_reported"
    assert entries["cannoneer:tower"].flag("targets_air")
    assert answers(entries["cannoneer:tower"], "airborne")
    assert entries["cannoneer:tower"].field("damage_per_hit").status == "unknown"
    token = next(t for t, e in catalog.entries.items() if e.identity == "minion-giant:base")
    assert air_audit(catalog, [token]) == {}
