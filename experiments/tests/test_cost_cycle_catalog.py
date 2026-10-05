"""Reviewed deployment costs enable static proxies without guessing conditional states."""

from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.matchup_features.cycle import return_cost
from experiments.matchup_features.responses import fixed_cost
from experiments.mechanics.air_audit import air_audit
from experiments.mechanics.contracts import Value, unknown
from experiments.mechanics.load import load, load_partial_catalog

from clash_sos.infrastructure.clash_royale.catalog import CURRENT_CARD_ATTRIBUTES


def test_cost_revision_preserves_the_previous_snapshot_and_every_field() -> None:
    path = Path("experiments/mechanics/inputs/2026-10-02-partial-r7.json")
    assert sha256(path.read_bytes()).hexdigest() == (
        "7ef2757a7cd68227d7541a5ae2099beb9ec4c43e5d16e5da2807bbe8136f1d84"
    )
    previous = load(path)
    current = load("experiments/mechanics/inputs/2026-10-02-partial-r8.json")
    assert (current.era_start, current.era_end) == (previous.era_start, previous.era_end)
    for token, old in previous.entries.items():
        assert all(current.for_token(token).field(n) == f for n, f in old.fields.items())


def test_all_costs_have_evidence_and_match_reviewed_attributes() -> None:
    catalog = load_partial_catalog()
    kinds: Counter[Value] = Counter()
    for token in catalog.entries:
        entry = catalog.for_token(token)
        if entry.kind == "tower":
            assert all(
                entry.field(n).status == "not_applicable"
                for n in ("cost_kind", "deploy_cost", "conditional_cost", "ordinary_cycle")
            )
            continue
        kinds[entry.field("cost_kind").value] += 1
        assert entry.flag("ordinary_cycle")
        for name in ("cost_kind", "ordinary_cycle"):
            assert entry.field(name).status == "verified"
            assert entry.field(name).source_url
        if entry.field("cost_kind").value == "fixed":
            assert entry.field("deploy_cost").status == "verified"
            assert fixed_cost(entry) == CURRENT_CARD_ATTRIBUTES.for_identity(entry.identity).elixir
        else:
            assert fixed_cost(entry) is None
            assert entry.field("deploy_cost").status == "not_applicable"
    assert kinds == {"fixed": 180, "conditional": 2}
    assert (
        air_audit(catalog, list(catalog.entries), include_conditional=True, include_cost=True) == {}
    )


def test_return_proxy_uses_champions_and_excludes_conditional_costs() -> None:
    entries = {e.identity: e for e in load_partial_catalog().entries.values()}
    deck = [
        entries[n]
        for n in (
            "balloon:base",
            "skeletons:base",
            "ice-spirit:base",
            "little-prince:champion",
            "ice-golem:hero",
            "mirror:base",
            "spirit-empress:base",
            "golem:base",
        )
    ]
    assert return_cost(deck, 0) == 7  # 1 + 1 + 3 + 2; no variable-cost shortcuts.
    assert entries["mirror:base"].field("conditional_cost").value == "previous_card_plus_one"
    assert entries["spirit-empress:base"].field("conditional_cost").value == "ground_3_air_6"


@pytest.mark.parametrize("identity", ["knight:base", "mirror:base", "spirit-empress:base"])
def test_cost_gate_requires_cycle_applicability(identity: str) -> None:
    catalog = load_partial_catalog()
    token = next(t for t, e in catalog.entries.items() if e.identity == identity)
    entry = catalog.for_token(token)
    altered = replace(entry, fields={**entry.fields, "ordinary_cycle": unknown("ordinary_cycle")})
    catalog = replace(catalog, entries={**catalog.entries, token: altered})
    assert air_audit(catalog, [token], include_conditional=True) == {}
    assert air_audit(catalog, [token], include_conditional=True, include_cost=True) == {
        identity: ("ordinary_cycle",)
    }
