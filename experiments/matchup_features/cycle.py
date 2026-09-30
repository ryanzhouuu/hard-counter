"""Deployment commitment and return-cost proxies exclude towers and unknown costs."""

from collections.abc import Sequence

from experiments.matchup_features.responses import FAMILIES, answers, fixed_cost
from experiments.mechanics.contracts import MechanicsEntry


def return_cost(cards: Sequence[MechanicsEntry], excluded: int) -> float | None:
    """Sum the four cheapest other fixed-cost ordinary-cycle cards, if available."""
    costs = sorted(
        cost
        for index, entry in enumerate(cards[:8])
        if index != excluded
        and entry.flag("ordinary_cycle")
        and (cost := fixed_cost(entry)) is not None
    )
    return sum(costs[:4]) if len(costs) >= 4 else None


def directed(
    attackers: Sequence[MechanicsEntry],
    defenders: Sequence[MechanicsEntry],
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    costs = [c for e in attackers[:8] if (c := fixed_cost(e)) is not None]
    mean = sum(costs) / len(costs) if costs else 0.0
    fields: list[tuple[str, float, str]] = [
        ("mean_fixed_cost", mean, "mean(known fixed deployment costs)"),
        (
            "cost_dispersion",
            sum((c - mean) ** 2 for c in costs) / len(costs) if costs else 0,
            "population variance of known fixed costs in elixir squared",
        ),
        ("cheap_count", float(sum(c <= 2 for c in costs)), "count(fixed cost <= 2)"),
        ("expensive_count", float(sum(c >= 5 for c in costs)), "count(fixed cost >= 5)"),
        (
            "conditional_count",
            float(sum(e.field("cost_kind").value == "conditional" for e in attackers[:8])),
            "count(conditional deploy cost)",
        ),
        (
            "cost_unknown_count",
            float(
                sum(
                    e.field("cost_kind").status == "unknown"
                    or (e.field("cost_kind").value == "fixed" and fixed_cost(e) is None)
                    for e in attackers[:8]
                )
            ),
            "count(unknown cost kind or missing fixed cost)",
        ),
        (
            "special_cycle_count",
            float(
                sum(
                    e.field("ordinary_cycle").known and not e.flag("ordinary_cycle")
                    for e in attackers[:8]
                )
            ),
            "count(verified nonordinary cycle cards)",
        ),
        (
            "cycle_unknown_count",
            float(sum(e.field("ordinary_cycle").status == "unknown" for e in attackers[:8])),
            "count(unknown cycle applicability)",
        ),
    ]
    for family in FAMILIES:
        threats = [(i, e) for i, e in enumerate(attackers[:8]) if e.flag(family)]
        responses = [(i, e) for i, e in enumerate(defenders[:8]) if answers(e, family)]
        threat_costs = [c for _, e in threats if (c := fixed_cost(e)) is not None]
        response_costs = [c for _, e in responses if (c := fixed_cost(e)) is not None]
        threat_returns = [c for i, _ in threats if (c := return_cost(attackers, i)) is not None]
        response_returns = [c for i, _ in responses if (c := return_cost(defenders, i)) is not None]
        missing = not threat_returns or not response_returns
        fields.extend(
            (
                (
                    f"{family}.threat_commitment",
                    sum(threat_costs),
                    "sum(known fixed threat deploy costs)",
                ),
                (
                    f"{family}.response_commitment",
                    min(response_costs) if response_costs else 0,
                    "minimum known fixed opposing answer cost; 0 if missing",
                ),
                (
                    f"{family}.commitment_missing",
                    float(not response_costs),
                    "opposing answer cost missing",
                ),
                (
                    f"{family}.return_cost_gap",
                    0 if missing else min(threat_returns) - min(response_returns),
                    "min(threat return proxy) - min(answer return proxy); 0 if unavailable",
                ),
                (
                    f"{family}.return_cost_missing",
                    float(missing),
                    "either return proxy unavailable",
                ),
            )
        )
    return (
        tuple(f"cycle.{name}" for name, _, _ in fields),
        tuple(float(value) for _, value, _ in fields),
        tuple(f"directed({formula}) - reverse" for _, _, formula in fields),
    )
