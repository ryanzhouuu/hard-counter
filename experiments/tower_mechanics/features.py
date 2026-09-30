"""Ninth-slot tower descriptors interact with threats, without battle DPS claims."""

from collections.abc import Sequence

from experiments.mechanics.contracts import (
    FeatureResult,
    MechanicsCatalog,
    MechanicsEntry,
    MechanicsUnavailable,
    decode_pair,
    difference,
)

CATEGORICAL = ("targets_air", "targets_ground", "area_damage", "tower_support", "burst")
QUANTITATIVE = ("damage_per_hit", "attack_interval", "recharge")


def require_quantitative(catalog: MechanicsCatalog, tokens: Sequence[int]) -> None:
    """Missing values disable D2; preregistered inapplicable values have an indicator."""
    for token in sorted(set(tokens)):
        entry = catalog.for_token(token)
        if entry.kind != "tower":
            raise ValueError("quantitative tower gate received a card")
        for name in QUANTITATIVE:
            field = entry.field(name)
            if field.status == "unknown":
                raise MechanicsUnavailable(
                    f"quantitative tower field unavailable: {entry.identity}/{name}"
                )
            if name in {"damage_per_hit", "attack_interval"} and field.status == "not_applicable":
                raise MechanicsUnavailable(
                    f"tower requires damage/cadence: {entry.identity}/{name}"
                )


def _directed(
    own: Sequence[MechanicsEntry],
    opposing: Sequence[MechanicsEntry],
    quantitative: bool,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    tower = own[8]
    threat_counts = {
        "airborne": sum(e.flag("airborne") for e in opposing[:8]),
        "multi_unit": sum(e.flag("multi_unit") for e in opposing[:8]),
        "building_targeting": sum(e.flag("building_targeting") for e in opposing[:8]),
        "own_troops": sum(not e.flag("spell") for e in own[:8]),
    }
    fields: list[tuple[str, float, str]] = []
    for descriptor in CATEGORICAL:
        for threat, count in threat_counts.items():
            fields.append(
                (
                    f"{descriptor}.{threat}",
                    float(tower.flag(descriptor) * count),
                    f"tower {descriptor} potential * {threat} card count",
                )
            )
        fields.append(
            (
                f"{descriptor}.unknown",
                float(tower.field(descriptor).status == "unknown"),
                "tower categorical descriptor unknown indicator",
            )
        )
    for threat in ("airborne", "multi_unit", "building_targeting"):
        fields.append(
            (
                f"{threat}.unknown",
                float(sum(e.field(threat).status == "unknown" for e in opposing[:8])),
                "opposing threat descriptor unknown count",
            )
        )
    if quantitative:
        for descriptor in QUANTITATIVE:
            field = tower.field(descriptor)
            value = tower.number(descriptor) or 0.0
            for threat, count in threat_counts.items():
                fields.append(
                    (
                        f"{descriptor}.{threat}",
                        value * count,
                        f"static tower {descriptor} ({field.unit}) * {threat} count",
                    )
                )
            fields.append(
                (
                    f"{descriptor}.not_applicable",
                    float(field.status == "not_applicable"),
                    "inapplicable numeric descriptor",
                )
            )
    return (
        tuple(f"tower.{name}" for name, _, _ in fields),
        tuple(value for _, value, _ in fields),
        tuple(f"directed({formula}) - reverse" for _, _, formula in fields),
    )


def extract(
    tokens: Sequence[Sequence[int]],
    catalog: MechanicsCatalog,
    *,
    quantitative: bool = False,
) -> FeatureResult:
    left, right = decode_pair(tokens, catalog)
    if quantitative:
        require_quantitative(catalog, [tokens[0][8], tokens[1][8]])
    names, forward, formulas = _directed(left, right, quantitative)
    _, reverse, _ = _directed(right, left, quantitative)
    return difference(names, forward, reverse, formulas)
