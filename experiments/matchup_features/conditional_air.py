"""Potential air channels keep triggers and restricted effects separate from ordinary responses."""

from collections.abc import Sequence

from experiments.matchup_features.responses import answers
from experiments.mechanics.conditional_air import (
    CAPABILITIES,
    RESPONSES,
    has_channel,
    hybrid_issues,
)
from experiments.mechanics.contracts import MechanicsEntry
from experiments.mechanics.fields import CONDITIONAL_CATEGORIES


def directed(
    attackers: Sequence[MechanicsEntry], defenders: Sequence[MechanicsEntry]
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    """Count potential channels and opposing threat composition without assuming ability use."""
    cards, opposing = attackers[:8], defenders[:8]
    channels = {name: tuple(e for e in cards if has_channel(e, name)) for name in CAPABILITIES}
    active = tuple(e for e in cards if any(has_channel(e, name) for name in CAPABILITIES))
    ordinary_threats = sum(e.flag("airborne") for e in opposing)
    conditional_threats = sum(
        has_channel(e, "conditional_airborne") and not e.flag("airborne") for e in opposing
    )
    fields: list[tuple[str, float, str]] = []
    for capability, entries in channels.items():
        name = capability.removeprefix("conditional_")
        fields.append(
            (name, float(len(entries)), f"count(complete potential {capability} channels)")
        )
        unavailable = sum(
            e.field(capability).status == "unknown"
            or (e.flag(capability) and not has_channel(e, capability))
            for e in cards
        )
        fields.append(
            (
                f"{name}.unavailable",
                float(unavailable),
                "unknown flag or incomplete positive channel count",
            )
        )
        if capability in RESPONSES:
            for kind, threats in (
                ("ordinary", ordinary_threats),
                ("conditional", conditional_threats),
            ):
                fields.append(
                    (
                        f"{name}.opposing_{kind}_threats",
                        float(len(entries) * threats),
                        f"complete potential {capability} count * opposing {kind} air threats",
                    )
                )
    for category, label in (
        ("conditional_air_trigger", "trigger"),
        ("conditional_air_response_scope", "scope"),
        ("conditional_air_control_kind", "control"),
    ):
        eligible = channels["conditional_air_control"] if label == "control" else active
        for value in sorted(CONDITIONAL_CATEGORIES[category]):
            count = sum(e.field(category).value == value for e in eligible)
            fields.append(
                (
                    f"{label}.{value}",
                    float(count),
                    f"complete channel cards with {category}={value}",
                )
            )
    additional = sum(
        any(has_channel(e, name) for name in RESPONSES)
        and not answers(e, "airborne")
        and (
            e.field("targets_air").value is False
            or (e.field("airborne_mode").value == "hybrid" and not hybrid_issues(e))
        )
        for e in cards
    )
    fields.append(
        (
            "additional_response_cards",
            float(additional),
            "complete potential response cards with documented lack of ordinary response; "
            "restricted channels retained",
        )
    )
    paid = tuple(e for e in active if e.field("conditional_air_trigger").value == "paid_ability")
    fields.append(
        (
            "ability_cost",
            sum(e.number("ability_cost") or 0 for e in paid),
            "sum(known paid conditional-air ability cost); not deployment or observed expenditure",
        )
    )
    fields.append(
        (
            "ability_cost_missing",
            float(sum(e.number("ability_cost") is None for e in paid)),
            "paid conditional-air cards with unavailable ability cost",
        )
    )
    threats = sum(not e.flag("airborne") for e in channels["conditional_airborne"])
    for label, candidates in (
        ("deployable", tuple(e for e in opposing if e.field("spell").value is False)),
        ("spell", tuple(e for e in opposing if e.flag("spell"))),
    ):
        count = sum(answers(e, "airborne") for e in candidates)
        fields.append(
            (
                f"airborne.opposing_{label}_answers",
                float(threats * min(count, 2)),
                "additional potential airborne threat count * "
                f"min(opposing ordinary {label} answers, 2)",
            )
        )
    return (
        tuple(f"conditional_air.{name}" for name, _, _ in fields),
        tuple(value for _, value, _ in fields),
        tuple(f"directed({formula}) - reverse" for _, _, formula in fields),
    )
