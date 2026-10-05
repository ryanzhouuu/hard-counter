"""Applicability revisions preserve evidence and enforce inclusive UTC date bounds."""

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from experiments.common.data_access import RoleAccess
from experiments.common.readiness import readiness
from experiments.mechanics.load import load, load_partial_catalog
from experiments.tests.common.research_fixture import protocol, rows


def test_applicability_revision_preserves_october_second_evidence() -> None:
    path = Path("experiments/mechanics/inputs/2026-10-02-partial-r8.json")
    assert sha256(path.read_bytes()).hexdigest() == (
        "9a345c1b1f9db63b723c092c175eb185b1c5ca6d1e9d027a59af3509a94009e6"
    )
    previous, current = load(path), load_partial_catalog()
    assert current.version == "mechanics-partial:2026-10-05-r9"
    assert (current.era_start, current.era_end) == ("2026-09-24", "2026-10-05")
    assert previous.era_end == "2026-10-01"
    assert current.entries == previous.entries
    assert current.synthetic is previous.synthetic is False
    assert current.digest != previous.digest


@pytest.mark.parametrize("day", [2, 5, 6])
def test_readiness_accepts_extended_dates_and_rejects_later_dates(day: int) -> None:
    catalog = load_partial_catalog()
    tower_tokens = [t for t, entry in catalog.entries.items() if entry.kind == "tower"]
    start = datetime(2026, 10, day, 12, tzinfo=UTC)
    original = rows()
    population = tuple(
        replace(
            row,
            key=(start + (row.key[0] - original[0].key[0]), *row.key[1:]),
            tokens=tuple((*side[:8], tower_tokens[side[8] - 13]) for side in row.tokens),
        )
        for row in original
    )
    access = RoleAccess(protocol(population), population)
    report = readiness(access, catalog)
    mechanics = cast(dict[str, object], report["mechanics"])
    issues = cast(list[str], mechanics["compatibility_issues"])
    date_issue = "population battle dates exceed verified mechanics era"
    assert (date_issue in issues) == (day > 5)
    historical = load("experiments/mechanics/inputs/2026-10-02-partial-r8.json")
    old_report = readiness(access, historical)
    old_mechanics = cast(dict[str, object], old_report["mechanics"])
    assert date_issue in cast(list[str], old_mechanics["compatibility_issues"])
