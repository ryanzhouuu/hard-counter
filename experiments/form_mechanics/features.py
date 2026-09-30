"""Potential form capabilities interact with opposing static response composition."""

from collections.abc import Sequence

from experiments.matchup_features.responses import FAMILIES, summarize
from experiments.mechanics.contracts import (
    FeatureResult,
    MechanicsCatalog,
    MechanicsEntry,
    decode_pair,
    difference,
)

DESCRIPTORS = ("targets_air", "area_damage", "shield", "spawns_units", "control", "conditional")


def _directed(
    attackers: Sequence[MechanicsEntry],
    defenders: Sequence[MechanicsEntry],
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    forms = [e for e in attackers[:8] if e.base_identity or e.identity.endswith(":champion")]
    fields: list[tuple[str, float, str]] = []
    for descriptor in DESCRIPTORS:
        capability = sum(e.flag(descriptor) for e in forms)
        fields.extend(
            (
                (descriptor, float(capability), "form cards with potential capability"),
                (
                    f"{descriptor}.opposing_multi_unit",
                    float(capability * sum(e.flag("multi_unit") for e in defenders[:8])),
                    "potential form capability count * opposing multi-unit deployment count",
                ),
                (
                    f"{descriptor}.unknown",
                    float(sum(e.field(descriptor).status == "unknown" for e in forms)),
                    "unknown form descriptor count",
                ),
            )
        )
    for name in ("activation_cycles", "ability_cost"):
        values = [v for e in forms if (v := e.number(name)) is not None]
        fields.extend(
            (
                (
                    name,
                    sum(values),
                    f"sum(verified potential {name}); no activation-state inference",
                ),
                (
                    f"{name}.unknown",
                    float(sum(e.field(name).status == "unknown" for e in forms)),
                    "unknown potential activation/cost count",
                ),
            )
        )
    for family in FAMILIES:
        s = summarize(attackers, defenders, family)
        form_threats = sum(e.flag(family) for e in forms)
        answer_count = len(s.deployable) + len(s.spells)
        fields.extend(
            (
                (
                    f"{family}.unanswered_form_exposure",
                    float(form_threats * (answer_count == 0)),
                    "potential form threat count * no known deployable/spell response",
                ),
                (
                    f"{family}.shared_response_pressure",
                    float(form_threats * (s.threats > 1) * (len(s.deployable) == 1)),
                    "form threat count * multiple threats * sole deployable answer",
                ),
            )
        )
    return (
        tuple(f"form.{name}" for name, _, _ in fields),
        tuple(value for _, value, _ in fields),
        tuple(f"directed({formula}) - reverse" for _, _, formula in fields),
    )


def extract(
    tokens: Sequence[Sequence[int]],
    catalog: MechanicsCatalog,
    *,
    inherited: bool = False,
) -> FeatureResult:
    actual_left, actual_right = decode_pair(tokens, catalog)
    left, right = decode_pair(tokens, catalog, inherited=inherited)
    if inherited:
        left = tuple(
            MechanicsEntry(e.identity, e.kind, e.fields, actual.base_identity)
            for e, actual in zip(left, actual_left, strict=True)
        )
        right = tuple(
            MechanicsEntry(e.identity, e.kind, e.fields, actual.base_identity)
            for e, actual in zip(right, actual_right, strict=True)
        )
    names, forward, formulas = _directed(left, right)
    _, reverse, _ = _directed(right, left)
    return difference(names, forward, reverse, formulas)
