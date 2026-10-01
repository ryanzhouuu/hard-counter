"""Hybrid mobility is explicit catalog evidence rather than a boolean deployment state."""

import pytest
from experiments.mechanics.contracts import MechanicField, MechanicsEntry
from experiments.mechanics.load import from_payload, load_partial_catalog, to_payload


@pytest.mark.parametrize("mode", ["ground", "air", "hybrid"])
def test_registered_airborne_modes(mode: str) -> None:
    entry = MechanicsEntry(
        "fixture:base", "card", {"airborne_mode": MechanicField(mode, "category", "synthetic")}
    )
    assert entry.field("airborne_mode").value == mode


def test_unregistered_airborne_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unregistered airborne mode"):
        MechanicsEntry(
            "fixture:base",
            "card",
            {"airborne_mode": MechanicField("sometimes", "category", "synthetic")},
        )


def test_hybrid_classification_and_conditions_survive_serialization() -> None:
    catalog = load_partial_catalog()
    restored = from_payload(to_payload(catalog))
    assert restored.digest == catalog.digest
    entry = next(e for e in restored.entries.values() if e.identity == "spirit-empress:base")
    assert entry.field("airborne_mode").value == "hybrid"
    assert entry.field("airborne_mode").known
    assert entry.field("airborne").status == "unknown"
    assert entry.field("targets_air").status == "unknown"
    assert entry.field("airborne_condition").value == "available_elixir_at_least_6"
