"""Units and verification vocabulary shared by research mechanics entries."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

Value = bool | int | float | str | None
Status = Literal["verified", "user_reported", "unknown", "not_applicable", "synthetic"]
FIELD_UNITS: Mapping[str, str] = MappingProxyType(
    {
        **{
            name: "flag"
            for name in (
                "airborne",
                "targets_air",
                "targets_ground",
                "building_targeting",
                "multi_unit",
                "spawns_units",
                "area_damage",
                "defensive_building",
                "spell",
                "spell_damage",
                "spell_control",
                "building_disruption",
                "ordinary_cycle",
                "shield",
                "control",
                "conditional",
                "tower_support",
                "burst",
            )
        },
        "cost_kind": "category",
        "deploy_cost": "elixir",
        "conditional_cost": "category",
        "airborne_mode": "category",
        "airborne_condition": "category",
        "targets_air_condition": "category",
        "activation_cycles": "deployments",
        "ability_cost": "elixir",
        "damage_per_hit": "damage",
        "attack_interval": "seconds",
        "recharge": "seconds",
    }
)
