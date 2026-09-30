"""Versioned static hypotheses; patterns are not card-specific counter labels."""

from dataclasses import dataclass

PATTERN_VERSION = "mechanical-packages:v1"


@dataclass(frozen=True)
class Pattern:
    name: str
    threat: str
    support: str
    defense: str
    minimum_threats: int
    maximum_answers: int
    cap: int = 2
    condition: str = "supporting spell must be a separate card"

    def __post_init__(self) -> None:
        if not self.name or self.minimum_threats < 1 or self.maximum_answers < 1 or self.cap < 1:
            raise ValueError("invalid pattern thresholds")
        if self.condition != "supporting spell must be a separate card":
            raise ValueError("unregistered support condition")
        if self.threat not in {"multi_unit", "building_targeting", "airborne"}:
            raise ValueError("unregistered threat predicate")
        if self.support not in {"spell_damage", "spell_control", "building_disruption"}:
            raise ValueError("unregistered supporting capability")
        if self.defense not in {"area_damage", "defensive_building", "targets_air"}:
            raise ValueError("unregistered defensive package")


PATTERNS = (
    Pattern("swarm_spell_single_area", "multi_unit", "spell_damage", "area_damage", 1, 1),
    Pattern(
        "building_pressure_disruption",
        "building_targeting",
        "building_disruption",
        "defensive_building",
        1,
        2,
    ),
    Pattern("shared_area_answer", "multi_unit", "spell_control", "area_damage", 2, 1),
)
