"""Capped package interactions and fit-role-only support eligibility."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from experiments.higher_order.patterns import PATTERNS, Pattern
from experiments.mechanics.contracts import (
    FeatureResult,
    MechanicsCatalog,
    MechanicsEntry,
    MechanicsUnavailable,
    decode_pair,
    difference,
)


def _directed(
    attackers: Sequence[MechanicsEntry],
    defenders: Sequence[MechanicsEntry],
    registry: Sequence[Pattern],
) -> tuple[float, ...]:
    values: list[float] = []
    for pattern in registry:
        threats = [i for i, e in enumerate(attackers[:8]) if e.flag(pattern.threat)]
        supports = [
            i for i, e in enumerate(attackers[:8]) if e.flag("spell") and e.flag(pattern.support)
        ]
        answers = sum(e.flag(pattern.defense) for e in defenders[:8] if not e.flag("spell"))
        has_separate_support = any(s not in threats for s in supports)
        active = len(threats) >= pattern.minimum_threats and has_separate_support
        active = active and 1 <= answers <= pattern.maximum_answers
        values.append(float(min(len(threats), pattern.cap) if active else 0))
    return tuple(values)


def extract(
    tokens: Sequence[Sequence[int]],
    catalog: MechanicsCatalog,
    *,
    registry: Sequence[Pattern] = PATTERNS,
) -> FeatureResult:
    if not registry or len(registry) > 6 or len({p.name for p in registry}) != len(registry):
        raise ValueError("registry requires one to six unique frozen patterns")
    left, right = decode_pair(tokens, catalog)
    for side in (left, right):
        for entry in side:
            required = {p.defense for p in registry}
            if entry.kind == "card":
                required |= {"spell", *(p.threat for p in registry), *(p.support for p in registry)}
            for name in sorted(required):
                if entry.field(name).status == "unknown":
                    raise MechanicsUnavailable(
                        f"pattern field unavailable: {entry.identity}/{name}"
                    )
    formulas = tuple(
        f"min(count({p.threat}),{p.cap}) * [count >= {p.minimum_threats}] * "
        f"[separate spell.{p.support}] * "
        f"[1 <= nonspell {p.defense} answers <= {p.maximum_answers}] - reverse"
        for p in registry
    )
    return difference(
        tuple(f"pattern.{p.name}" for p in registry),
        _directed(left, right, registry),
        _directed(right, left, registry),
        formulas,
    )


@dataclass(frozen=True)
class PatternEligibility:
    role: Literal["selection_fit", "refit"]
    names: tuple[str, ...]
    support: tuple[int, ...]
    active: tuple[bool, ...]
    reasons: tuple[str, ...]


def eligibility(
    rows: Sequence[FeatureResult],
    *,
    role: Literal["selection_fit", "refit"],
    minimum_support: int,
) -> PatternEligibility:
    if role not in {"selection_fit", "refit"} or minimum_support < 1 or not rows:
        raise ValueError(
            "support eligibility requires nonempty fit-role rows and a positive threshold"
        )
    names = rows[0].names
    if any(row.names != names for row in rows):
        raise ValueError("pattern registry changed between rows")
    columns = tuple(tuple(row.values[i] for row in rows) for i in range(len(names)))
    support = tuple(sum(value != 0 for value in column) for column in columns)
    reasons = tuple(
        "constant_column"
        if len(set(column)) < 2
        else "insufficient_support"
        if count < minimum_support
        else "eligible"
        for column, count in zip(columns, support, strict=True)
    )
    return PatternEligibility(
        role, names, support, tuple(r == "eligible" for r in reasons), reasons
    )
