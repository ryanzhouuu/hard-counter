import pytest

from clash_sos.domain.matchup_baseline import card_log_odds, laplace_probability, logit
from clash_sos.domain.matchup_pair import (
    DEFAULT_PAIR_EPOCHS,
    DEFAULT_PAIR_INIT_SCALE,
    DEFAULT_PAIR_LEARNING_RATE,
    CardPairPredictor,
    accumulate_pair_counts,
    initialize_card_pair_predictor,
)


def test_pair_fit_defaults_are_conservative() -> None:
    assert DEFAULT_PAIR_EPOCHS == 1
    assert DEFAULT_PAIR_LEARNING_RATE == 0.001
    assert DEFAULT_PAIR_INIT_SCALE == 8.0


def test_equal_decks_are_neutral() -> None:
    predictor = CardPairPredictor(
        identities=("the-log:base", "goblin-barrel:base"),
        additive=[0.4, -0.1],
        pair_upper=[0.8],
    )
    deck = ("the-log:base", "goblin-barrel:base")
    assert predictor.predict(deck, deck) == pytest.approx(0.5)


def test_side_swap_inverts_probability() -> None:
    predictor = CardPairPredictor(
        identities=("the-log:base", "goblin-barrel:base"),
        additive=[0.2, -0.3],
        pair_upper=[0.5],
    )
    probability = predictor.predict(("the-log:base",), ("goblin-barrel:base",))
    swapped = predictor.predict(("goblin-barrel:base",), ("the-log:base",))
    assert probability != pytest.approx(0.5)
    assert swapped == pytest.approx(1.0 - probability)


def test_same_identity_pair_weight_is_zero() -> None:
    predictor = CardPairPredictor(
        identities=("the-log:base", "goblin-barrel:base"),
        additive=[0.0, 0.0],
        pair_upper=[1.25],
    )
    assert predictor.pair_weight(0, 0) == 0.0
    assert predictor.pair_weight(1, 1) == 0.0
    assert predictor.pair_weight(0, 1) == pytest.approx(-predictor.pair_weight(1, 0))


def test_unknown_identities_contribute_zero() -> None:
    predictor = CardPairPredictor(
        identities=("the-log:base",),
        additive=[0.9],
        pair_upper=[],
    )
    assert predictor.predict(("missing:base",), ("also-missing:base",)) == pytest.approx(0.5)


def test_pair_score_divides_interaction_by_init_scale() -> None:
    unscaled = CardPairPredictor(
        identities=("the-log:base", "goblin-barrel:base"),
        additive=[0.0, 0.0],
        pair_upper=[0.8],
        pair_init_scale=1.0,
    )
    scaled = CardPairPredictor(
        identities=("the-log:base", "goblin-barrel:base"),
        additive=[0.0, 0.0],
        pair_upper=[0.8],
        pair_init_scale=8.0,
    )
    side_a = ("the-log:base",)
    side_b = ("goblin-barrel:base",)
    assert unscaled.score(side_a, side_b) == pytest.approx(0.8)
    assert scaled.score(side_a, side_b) == pytest.approx(0.1)


def test_payload_without_scale_scores_like_v1() -> None:
    loaded = CardPairPredictor.from_payload(
        {
            "additive": [0.0, 0.0],
            "identities": ["the-log:base", "goblin-barrel:base"],
            "pair_upper": [0.8],
        }
    )
    assert loaded.pair_init_scale == 1.0
    assert loaded.score(("the-log:base",), ("goblin-barrel:base",)) == pytest.approx(0.8)


def test_accumulate_pair_counts_tracks_directed_wins() -> None:
    counts = accumulate_pair_counts(
        (
            (1, ("the-log:base",), ("goblin-barrel:base",)),
            (0, ("the-log:base",), ("goblin-barrel:base",)),
            (1, ("the-log:base",), ("goblin-barrel:base",)),
        )
    )
    assert counts[("the-log:base", "goblin-barrel:base")] == (2, 3)


def test_laplace_init_is_antisymmetric() -> None:
    identities = ("the-log:base", "goblin-barrel:base")
    predictor = initialize_card_pair_predictor(
        identities,
        card_counts={"the-log:base": (6, 10), "goblin-barrel:base": (4, 10)},
        pair_counts={
            ("the-log:base", "goblin-barrel:base"): (8, 10),
            ("goblin-barrel:base", "the-log:base"): (2, 10),
        },
        alpha=1.0,
        pair_init_scale=DEFAULT_PAIR_INIT_SCALE,
    )
    forward = logit(laplace_probability(8, 10, alpha=1.0))
    reverse = logit(laplace_probability(2, 10, alpha=1.0))
    expected = 0.5 * (forward - reverse)
    assert predictor.pair_weight(0, 1) == pytest.approx(expected)
    assert predictor.pair_weight(1, 0) == pytest.approx(-expected)
    assert predictor.pair_init_scale == DEFAULT_PAIR_INIT_SCALE
    assert predictor.score(("the-log:base",), ("goblin-barrel:base",)) == pytest.approx(
        predictor.additive[0] - predictor.additive[1] + expected / DEFAULT_PAIR_INIT_SCALE
    )
    assert predictor.additive[0] == pytest.approx(card_log_odds(6, 10, alpha=1.0))
    assert predictor.additive[1] == pytest.approx(card_log_odds(4, 10, alpha=1.0))


def test_sgd_step_moves_probability_toward_label() -> None:
    predictor = CardPairPredictor(
        identities=("the-log:base", "goblin-barrel:base"),
        additive=[0.0, 0.0],
        pair_upper=[0.0],
    )
    side_a = ("the-log:base",)
    side_b = ("goblin-barrel:base",)
    before = predictor.predict(side_a, side_b)
    predictor.sgd_step(side_a, side_b, label=1, learning_rate=0.5, l2=0.0)
    after = predictor.predict(side_a, side_b)
    assert before == pytest.approx(0.5)
    assert after > before


def test_predictor_json_round_trip_preserves_prediction() -> None:
    predictor = CardPairPredictor(
        identities=("the-log:base", "goblin-barrel:base", "knight:base"),
        additive=[0.1, -0.2, 0.05],
        pair_upper=[0.3, -0.4, 0.15],
        pair_init_scale=8.0,
    )
    side_a = ("the-log:base", "knight:base")
    side_b = ("goblin-barrel:base",)
    payload = predictor.to_payload()
    loaded = CardPairPredictor.from_payload(payload)
    assert loaded.identities == predictor.identities
    assert loaded.pair_init_scale == predictor.pair_init_scale
    assert loaded.predict(side_a, side_b) == pytest.approx(predictor.predict(side_a, side_b))
    assert loaded.pair_weight(0, 2) == pytest.approx(predictor.pair_weight(0, 2))
