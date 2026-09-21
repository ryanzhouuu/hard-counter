from collections.abc import Sequence

import pytest

from clash_sos.domain.matchup_lgbm import (
    PresenceRow,
    PresenceSchema,
    lightgbm_training_params,
    past_only_skill_gaps,
    predict_equal_skill,
    symmetrized_probability,
    watch_row_count,
)
from clash_sos.domain.player_skill import PlayerSkillTracker

IDENTITIES = ("archers:base", "knight:base")


def test_presence_row_keeps_shared_cards_and_orders_columns() -> None:
    schema = PresenceSchema(IDENTITIES)
    row = schema.row(
        ("knight:base", "knight:base", "missing:base"),
        ("knight:base", "archers:base"),
        skill_diff=1.5,
    )
    assert schema.skill_column == 4
    assert schema.feature_count == 5
    assert row.indices == (1, 2, 3, 4)
    assert row.values == (1.0, 1.0, 1.0, 1.5)


def test_presence_schema_rejects_duplicate_identities() -> None:
    with pytest.raises(ValueError, match="unique"):
        PresenceSchema(("knight:base", "knight:base"))


def test_catalog_width_matches_the_176_identity_layout() -> None:
    schema = PresenceSchema(tuple(f"card-{index}:base" for index in range(176)))
    assert schema.skill_column == 352
    assert schema.feature_count == 353


def test_symmetrized_probability_inverts_when_sides_swap() -> None:
    forward = symmetrized_probability(0.8, 0.3)
    backward = symmetrized_probability(0.3, 0.8)
    assert forward == pytest.approx(0.75)
    assert backward == pytest.approx(1.0 - forward)
    assert symmetrized_probability(0.62, 0.62) == pytest.approx(0.5)


def test_equal_skill_predict_zeroes_skill_and_cancels_equal_decks() -> None:
    schema = PresenceSchema(IDENTITIES)
    seen: list[float] = []

    def raw_predict(rows: Sequence[PresenceRow]) -> tuple[float, ...]:
        seen.extend(row.values[-1] for row in rows)
        return tuple(0.8 if 1 in row.indices else 0.2 for row in rows)

    forward = predict_equal_skill(raw_predict, schema, ("knight:base",), ("archers:base",))
    backward = predict_equal_skill(raw_predict, schema, ("archers:base",), ("knight:base",))
    equal = predict_equal_skill(raw_predict, schema, ("knight:base",), ("knight:base",))
    assert seen == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert forward == pytest.approx(0.8)
    assert backward == pytest.approx(1.0 - forward)
    assert equal == pytest.approx(0.5)


def test_skill_gap_ignores_the_current_and_later_outcome() -> None:
    first = past_only_skill_gaps((("a", "b", 1), ("a", "c", 1)), PlayerSkillTracker())
    later_flipped = past_only_skill_gaps((("a", "b", 1), ("a", "c", 0)), PlayerSkillTracker())
    assert first[0] == 0.0
    assert first[1] > 0.0
    assert later_flipped[0] == first[0]
    assert later_flipped[1] == first[1]


def test_watch_tail_is_ten_percent_and_leaves_a_fit_prefix() -> None:
    assert watch_row_count(3_809_835) == 380_983
    assert watch_row_count(2) == 1
    with pytest.raises(ValueError, match="two training rows"):
        watch_row_count(1)


def test_training_params_constrain_only_the_skill_column() -> None:
    params = lightgbm_training_params(identity_count=176)
    constraints = params["monotone_constraints"]
    assert params["objective"] == "binary"
    assert params["learning_rate"] == 0.05
    assert params["num_leaves"] == 63
    assert params["min_data_in_leaf"] == 200
    assert params["num_threads"] == 4
    assert isinstance(constraints, list)
    assert constraints == [0] * 352 + [1]
