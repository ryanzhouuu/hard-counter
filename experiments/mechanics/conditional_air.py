"""Conditional channels require documented triggers; they never establish observed activation."""

from experiments.mechanics.contracts import MechanicsEntry

CAPABILITIES = (
    "conditional_airborne",
    "conditional_air_damage",
    "conditional_air_control",
    "conditional_air_reflection",
)
RESPONSES = CAPABILITIES[1:]
HYBRID_VALUES = {
    "airborne_mode": "hybrid",
    "conditional_airborne": True,
    "conditional_air_damage": True,
    "airborne_condition": "available_elixir_at_least_6",
    "targets_air_condition": "available_elixir_at_least_6",
    "conditional_air_trigger": "available_elixir_at_least_6",
    "conditional_air_response_scope": "flying_form_attack",
    "spell": False,
}


def channel_fields(
    entry: MechanicsEntry, capability: str, *, include_cost: bool = False
) -> tuple[str, ...]:
    """Require metadata for a positive channel; costs are an optional additional gate."""
    if capability not in CAPABILITIES:
        raise ValueError("unregistered conditional air capability")
    fields = [capability, "conditional_air_trigger", "conditional_air_response_scope"]
    if capability == "conditional_air_control":
        fields.append("conditional_air_control_kind")
    if entry.field("conditional_air_trigger").value == "paid_ability":
        fields.append("ability_usage")
        if include_cost:
            fields.append("ability_cost")
    return tuple(fields)


def hybrid_issues(entry: MechanicsEntry) -> tuple[str, ...]:
    """Validate the registered six-Elixir alternative rather than accepting a hybrid label alone."""
    return tuple(
        name
        for name, value in HYBRID_VALUES.items()
        if not entry.field(name).known or entry.field(name).value != value
    )


def has_channel(entry: MechanicsEntry, capability: str) -> bool:
    """Count potential capability only when its activation and response metadata are known."""
    fields = channel_fields(entry, capability)
    if entry.kind != "card" or not entry.flag(capability):
        return False
    if entry.field("airborne_mode").value == "hybrid" and hybrid_issues(entry):
        return False
    return all(entry.field(name).known for name in fields)
