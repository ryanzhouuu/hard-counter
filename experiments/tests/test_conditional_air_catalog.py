"""Potential air channels retain evidence without changing ordinary deployment predicates."""

from hashlib import sha256
from pathlib import Path

import pytest
from experiments.matchup_features.responses import answers
from experiments.mechanics.contracts import MechanicField, MechanicsEntry
from experiments.mechanics.load import from_payload, load, load_partial_catalog, to_payload


@pytest.mark.parametrize(
    ("identity", "capabilities", "trigger", "scope", "control", "cost"),
    [
        (
            "spirit-empress:base",
            ("airborne", "air_damage"),
            "available_elixir_at_least_6",
            "flying_form_attack",
            None,
            None,
        ),
        (
            "wizard:hero",
            ("airborne", "air_damage", "air_control"),
            "paid_ability",
            "enhanced_attack_area",
            "pull",
            1,
        ),
        ("ice-golem:hero", ("air_damage", "air_control"), "paid_ability", "moving_area", "slow", 2),
        (
            "giant:hero",
            ("air_damage", "air_control"),
            "paid_ability",
            "selected_troop_ground_only_splash",
            "throw_and_stun",
            2,
        ),
        (
            "mighty-miner:champion",
            ("air_damage", "air_control"),
            "paid_ability",
            "departure_bomb_area",
            "knockback",
            1,
        ),
        (
            "monk:champion",
            ("air_reflection",),
            "paid_ability",
            "eligible_incoming_projectiles",
            None,
            1,
        ),
    ],
)
def test_six_candidates_have_sourced_channels_and_activation_semantics(
    identity: str,
    capabilities: tuple[str, ...],
    trigger: str,
    scope: str,
    control: str | None,
    cost: int | None,
) -> None:
    catalog = load_partial_catalog()
    restored = from_payload(to_payload(catalog))
    assert restored.digest == catalog.digest
    entry = next(e for e in restored.entries.values() if e.identity == identity)
    expected: dict[str, object] = {f"conditional_{name}": True for name in capabilities}
    expected.update(conditional_air_trigger=trigger, conditional_air_response_scope=scope)
    if control is not None:
        expected["conditional_air_control_kind"] = control
    if cost is not None:
        expected.update(ability_cost=cost, ability_usage="single_use_per_deployment")
    for name, value in expected.items():
        field = entry.field(name)
        assert field.status == "verified"
        assert field.value == value
        assert field.source_url and field.source_effective_date
        assert field.inherited_from is None
    if cost is not None:
        assert entry.field("ability_usage").source_effective_date == "2026-08-04"
    else:
        assert entry.field("ability_usage").status == "not_applicable"
        assert entry.field("ability_cost").status == "not_applicable"


def test_revision_preserves_every_existing_field_and_ordinary_response() -> None:
    path = Path("experiments/mechanics/inputs/2026-10-01-partial-r5.json")
    assert sha256(path.read_bytes()).hexdigest() == (
        "07cb7b8f3bc96a4d4847f2adb600f1fdc213dbff73308d5c630d4f7a0b78605a"
    )
    previous = load(path)
    current = load("experiments/mechanics/inputs/2026-10-02-partial-r6.json")
    assert current.era_start == previous.era_start
    assert current.era_end == previous.era_end
    assert set(current.entries) == set(previous.entries)
    for token, old in previous.entries.items():
        new = current.for_token(token)
        assert (new.identity, new.kind, new.base_identity) == (
            old.identity,
            old.kind,
            old.base_identity,
        )
        assert all(new.field(name) == field for name, field in old.fields.items())
        assert answers(new, "airborne") == answers(old, "airborne")


def test_missing_channels_and_combat_measurements_remain_unknown() -> None:
    entries = {e.identity: e for e in load_partial_catalog().entries.values()}
    assert entries["knight:base"].field("conditional_air_damage").status == "unknown"
    assert entries["knight:hero"].field("conditional_air_control").status == "unknown"
    assert entries["monk:champion"].field("conditional_air_damage").status == "unknown"
    assert entries["spirit-empress:base"].field("airborne").status == "unknown"
    assert entries["spirit-empress:base"].field("targets_air").status == "unknown"
    assert not answers(entries["giant:hero"], "airborne")
    assert answers(entries["wizard:hero"], "airborne")
    for entry in entries.values():
        assert entry.field("damage_per_hit").status == "unknown"


@pytest.mark.parametrize(
    "name",
    [
        "conditional_air_trigger",
        "conditional_air_response_scope",
        "conditional_air_control_kind",
        "ability_usage",
    ],
)
def test_conditional_categories_reject_unregistered_values(name: str) -> None:
    with pytest.raises(ValueError, match="unregistered"):
        MechanicsEntry(
            "fixture:base", "card", {name: MechanicField("unreviewed", "category", "synthetic")}
        )


def test_conditional_capabilities_require_boolean_values() -> None:
    with pytest.raises(ValueError, match="requires a boolean"):
        MechanicsEntry(
            "fixture:base",
            "card",
            {"conditional_air_damage": MechanicField("yes", "flag", "synthetic")},
        )


def test_hybrid_actor_classification_preserves_the_sixth_revision() -> None:
    previous = load("experiments/mechanics/inputs/2026-10-02-partial-r6.json")
    current = load_partial_catalog()
    for token, old in previous.entries.items():
        assert all(current.for_token(token).field(n) == f for n, f in old.fields.items())
    entry = next(e for e in current.entries.values() if e.identity == "spirit-empress:base")
    assert entry.field("spell").known and entry.field("spell").value is False
    assert entry.field("spell").source_url == entry.field("airborne_mode").source_url
