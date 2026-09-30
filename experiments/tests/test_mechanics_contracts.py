import pytest
from experiments.mechanics.contracts import MechanicField, MechanicsEntry


def test_unknown_false_zero_and_units_are_distinct() -> None:
    unknown = MechanicField(None, "flag", "unknown")
    false = MechanicField(False, "flag", "synthetic")
    zero = MechanicField(0, "elixir", "synthetic")
    assert not unknown.known and false.known and zero.known
    assert unknown.value is None and false.value is False and zero.value == 0
    with pytest.raises(ValueError, match="unit"):
        MechanicsEntry("fixture:base", "card", {"ability_cost": unknown})
