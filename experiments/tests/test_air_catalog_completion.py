"""Ordinary air evidence stays complete without inventing conditional deployment state."""

from hashlib import sha256
from pathlib import Path

import pytest
from experiments.matchup_features.responses import answers
from experiments.mechanics.air_audit import air_audit, require_air_mechanics
from experiments.mechanics.contracts import MechanicsUnavailable
from experiments.mechanics.load import bind_tokens, load, load_partial_catalog


@pytest.mark.parametrize(
    ("revision", "digest"),
    [
        ("r2", "d6d591a71788a55b4f0d22d4ae0ebe1961c4c3a953095e8198554f6aef2bccc3"),
        ("r3", "3918a9a5ba83eab0e7e26d557bb8aad12a2a060d967832c9eb358d1c4dd0a09d"),
        ("r4", "e9cc00058f54661e418a57556c7481d97366138d91b082dc8da488e6b221550d"),
    ],
)
def test_reviewed_october_revisions_remain_frozen(revision: str, digest: str) -> None:
    path = Path(f"experiments/mechanics/inputs/2026-10-01-partial-{revision}.json")
    assert sha256(path.read_bytes()).hexdigest() == digest
    previous = load(path)
    current = load_partial_catalog()
    assert current.digest != previous.digest
    for token, entry in previous.entries.items():
        for name, field in entry.fields.items():
            if field.known:
                assert current.for_token(token).field(name) == field


def test_air_coverage_survives_model_vocabulary_reordering() -> None:
    catalog = load_partial_catalog()
    identities = tuple(reversed([entry.identity for entry in catalog.entries.values()]))
    bound = bind_tokens(catalog, identities)
    assert air_audit(bound, list(bound.entries)) == {
        "spirit-empress:base": ("targets_air", "airborne")
    }
    ordinary = [t for t, e in bound.entries.items() if e.identity != "spirit-empress:base"]
    require_air_mechanics(bound, ordinary)
    with pytest.raises(MechanicsUnavailable, match="spirit-empress"):
        require_air_mechanics(bound, list(bound.entries))


def test_spirit_empress_records_the_condition_without_claiming_deployment_state() -> None:
    entry = next(
        e for e in load_partial_catalog().entries.values() if e.identity == "spirit-empress:base"
    )
    mode = entry.field("airborne_mode")
    assert mode.known
    assert mode.value == "hybrid"
    assert mode.source_effective_date == "2025-07-07"
    for name in ("airborne", "targets_air"):
        assert entry.field(name).status == "unknown"
        condition = entry.field(f"{name}_condition")
        assert condition.known
        assert condition.value == "available_elixir_at_least_6"
        assert condition.source_url == (
            "https://royaleapi.com/blog/spirit-empress-new-card-2025-july?lang=en"
        )


@pytest.mark.parametrize(
    ("identity", "airborne", "air_answer"),
    [
        ("knight:base", False, False),
        ("archers:base", False, True),
        ("minions:base", True, True),
        ("lava-hound:base", True, False),
        ("cannon:base", False, False),
        ("goblin-giant:base", False, True),
        ("goblin-machine:base", False, True),
        ("electro-giant:base", False, True),
        ("ram-rider:base", False, True),
        ("goblinstein:champion", False, True),
        ("night-witch:base", False, False),
        ("royal-ghost:evolution", False, False),
        ("royal-hogs:evolution", True, False),
        ("wizard:hero", False, True),
        ("ice-wizard:hero", False, True),
    ],
)
def test_primary_transport_and_ordinary_answers(
    identity: str, airborne: bool, air_answer: bool
) -> None:
    entry = next(e for e in load_partial_catalog().entries.values() if e.identity == identity)
    assert entry.field("airborne").value is airborne
    assert entry.field("airborne").known
    assert answers(entry, "airborne") is air_answer
    assert entry.field("damage_per_hit").status == "unknown"


@pytest.mark.parametrize("identity", ["mirror:base", "clone:base", "graveyard:base"])
def test_copying_and_summoning_are_not_direct_spell_answers(identity: str) -> None:
    entry = next(e for e in load_partial_catalog().entries.values() if e.identity == identity)
    assert entry.flag("spell")
    assert entry.field("airborne").status == "not_applicable"
    assert not answers(entry, "airborne")


def test_community_evidence_records_review_dates_without_replacing_manual_facts() -> None:
    catalog = load_partial_catalog()
    community = [
        field
        for entry in catalog.entries.values()
        for field in entry.fields.values()
        if field.source_url and "clashroyale.fandom.com" in field.source_url
    ]
    assert community
    assert all(field.source_effective_date == "2026-10-01" for field in community)
    freeze = next(e for e in catalog.entries.values() if e.identity == "freeze:base")
    assert freeze.field("targets_air").status == "user_reported"
    assert freeze.field("targets_air").evidence_sha256


@pytest.mark.parametrize(
    ("identity", "control"),
    [
        ("rage:base", False),
        ("royal-delivery:base", False),
        ("goblin-curse:base", False),
        ("arrows:base", False),
        ("void:base", False),
        ("freeze:base", True),
        ("lightning:base", True),
        ("poison:base", True),
    ],
)
def test_control_means_restricting_enemy_movement_or_actions(identity: str, control: bool) -> None:
    entry = next(e for e in load_partial_catalog().entries.values() if e.identity == identity)
    assert entry.field("spell_control").known
    assert entry.field("spell_control").value is control
    assert answers(entry, "airborne")
