"""Plausible static response channels, without simulated defensive outcomes."""

from collections.abc import Sequence
from dataclasses import dataclass

from experiments.mechanics.contracts import MechanicsEntry

FAMILIES = ("airborne", "multi_unit", "building_targeting")
ANSWER_FIELDS = (
    "targets_air",
    "targets_ground",
    "area_damage",
    "defensive_building",
    "spell",
    "spell_damage",
    "spell_control",
    "building_disruption",
)
REQUIRED_FIELDS = (
    *FAMILIES,
    *ANSWER_FIELDS,
    "cost_kind",
    "deploy_cost",
)


@dataclass(frozen=True)
class ResponseSummary:
    threats: int
    deployable: tuple[MechanicsEntry, ...]
    spells: tuple[MechanicsEntry, ...]
    towers: tuple[MechanicsEntry, ...]
    minimum_cost: float | None
    unknown: int


def answers(entry: MechanicsEntry, family: str) -> bool:
    if family not in FAMILIES:
        raise ValueError("unsupported threat family")
    if family == "airborne":
        return entry.flag("targets_air") and (
            not entry.flag("spell") or entry.flag("spell_damage") or entry.flag("spell_control")
        )
    if family == "multi_unit":
        return entry.flag("area_damage")
    if entry.kind == "tower":
        return entry.flag("targets_ground")
    return entry.flag("defensive_building") or (
        entry.flag("spell") and entry.flag("building_disruption")
    )


def fixed_cost(entry: MechanicsEntry) -> float | None:
    if entry.kind == "card" and entry.field("cost_kind").value == "fixed":
        return entry.number("deploy_cost")
    return None


def summarize(
    attackers: Sequence[MechanicsEntry],
    defenders: Sequence[MechanicsEntry],
    family: str,
) -> ResponseSummary:
    threats = sum(entry.flag(family) for entry in attackers[:8])
    candidates = tuple(entry for entry in defenders if answers(entry, family))
    deployable = tuple(e for e in candidates if e.kind == "card" and not e.flag("spell"))
    spells = tuple(e for e in candidates if e.kind == "card" and e.flag("spell"))
    towers = tuple(e for e in candidates if e.kind == "tower")
    costs = [cost for e in (*deployable, *spells) if (cost := fixed_cost(e)) is not None]
    unknown = sum(e.field(family).status == "unknown" for e in attackers[:8]) + sum(
        e.field(name).status == "unknown" for e in defenders for name in ANSWER_FIELDS
    )
    return ResponseSummary(
        threats, deployable, spells, towers, min(costs) if costs else None, unknown
    )


def directed(
    attackers: Sequence[MechanicsEntry],
    defenders: Sequence[MechanicsEntry],
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    names: list[str] = []
    values: list[float] = []
    formulas: list[str] = []
    for family in FAMILIES:
        s = summarize(attackers, defenders, family)
        channel_counts = (len(s.deployable), len(s.spells), len(s.towers))
        any_response = sum(channel_counts) > 0
        entries = (
            ("threat_count", s.threats, "count(attacker cards with threat predicate)"),
            ("deployable_answers", min(channel_counts[0], 2), "min(deployable answers, 2)"),
            ("spell_answers", min(channel_counts[1], 2), "min(spell answers, 2)"),
            ("tower_answers", channel_counts[2], "static tower capability indicator"),
            (
                "minimum_response_cost",
                s.minimum_cost or 0,
                "min(known fixed answer cost); 0 if missing",
            ),
            ("response_cost_missing", s.minimum_cost is None, "no known fixed answer cost"),
            (
                "unanswered_exposure",
                s.threats * (not any_response),
                "threat_count * no known answer",
            ),
            (
                "sole_answer_pressure",
                max(s.threats - 1, 0) * (channel_counts[0] == 1),
                "max(threat_count - 1, 0) * exactly one deployable answer",
            ),
            ("response_context", s.threats * any_response, "threat_count * any known response"),
            (
                "unknown_fields",
                s.unknown,
                f"count(unknown attacker {family}) + "
                f"count(unknown opposing answer fields {ANSWER_FIELDS})",
            ),
        )
        for name, value, formula in entries:
            names.append(f"response.{family}.{name}")
            values.append(float(value))
            formulas.append(f"directed({formula}) - reverse")
    return tuple(names), tuple(values), tuple(formulas)
