"""Air-only preparation checks do not score outcomes or infer missing capabilities."""

from collections.abc import Sequence

from experiments.mechanics.contracts import MechanicsCatalog, MechanicsEntry, MechanicsUnavailable


def required_air_fields(entry: MechanicsEntry, *, include_cost: bool = False) -> tuple[str, ...]:
    """Require ordinary targeting; towers have neither deck threats nor deployment costs."""
    fields = ["targets_air"]
    if entry.kind == "tower":
        return tuple(fields)
    fields.append("airborne")
    if entry.flag("targets_air"):
        fields.append("spell")
        if entry.flag("spell"):
            fields.extend(("spell_damage", "spell_control"))
    if include_cost:
        fields.append("cost_kind")
        if entry.field("cost_kind").value == "fixed":
            fields.append("deploy_cost")
        elif entry.field("cost_kind").value == "conditional":
            fields.append("conditional_cost")
    return tuple(fields)


def air_audit(
    catalog: MechanicsCatalog,
    tokens: Sequence[int],
    *,
    include_cost: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Return incomplete identities; costs are a separate gate from response scarcity."""
    gaps: dict[str, tuple[str, ...]] = {}
    for token in sorted(set(tokens)):
        entry = catalog.for_token(token)
        missing = tuple(
            name
            for name in required_air_fields(entry, include_cost=include_cost)
            if not entry.field(name).known
        )
        if missing:
            gaps[entry.identity] = missing
    return gaps


def require_air_mechanics(
    catalog: MechanicsCatalog,
    tokens: Sequence[int],
    *,
    include_cost: bool = False,
) -> None:
    """Reject an incomplete preparation input instead of treating unknown flags as false."""
    gaps = air_audit(catalog, tokens, include_cost=include_cost)
    if gaps:
        examples = "; ".join(
            f"{identity}: {', '.join(fields)}" for identity, fields in gaps.items()
        )
        raise MechanicsUnavailable(
            f"air mechanics unavailable ({len(gaps)} identities): {examples}"
        )
