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
                "conditional_airborne",
                "conditional_air_damage",
                "conditional_air_control",
                "conditional_air_reflection",
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
        "conditional_air_trigger": "category",
        "conditional_air_response_scope": "category",
        "conditional_air_control_kind": "category",
        "ability_usage": "category",
        "activation_cycles": "deployments",
        "ability_cost": "elixir",
        "damage_per_hit": "damage",
        "attack_interval": "seconds",
        "recharge": "seconds",
    }
)

CONDITIONAL_CATEGORIES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "conditional_air_trigger": frozenset({"available_elixir_at_least_6", "paid_ability"}),
        "conditional_air_response_scope": frozenset(
            {
                "flying_form_attack",
                "enhanced_attack_area",
                "moving_area",
                "selected_troop_ground_only_splash",
                "departure_bomb_area",
                "eligible_incoming_projectiles",
            }
        ),
        "conditional_air_control_kind": frozenset({"pull", "slow", "throw_and_stun", "knockback"}),
        "ability_usage": frozenset({"single_use_per_deployment"}),
    }
)
