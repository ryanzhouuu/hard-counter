import pytest

from clash_sos.domain.matchup_baseline import (
    card_identity_key,
    card_log_odds,
    exact_matchup_probability,
    laplace_probability,
    mirrored_side_a_win,
    predict_card_log_odds,
    should_mirror_sides,
)


def test_card_identity_key_joins_id_and_form() -> None:
    assert card_identity_key("knight", "evolution") == "knight:evolution"


def test_should_mirror_sides_is_stable_for_fingerprint_and_seed() -> None:
    first = should_mirror_sides("abc", seed=0)
    second = should_mirror_sides("abc", seed=0)
    other_seed = should_mirror_sides("abc", seed=1)
    other_fingerprint = should_mirror_sides("def", seed=0)

    assert first is second
    assert first is not other_seed or first is not other_fingerprint


def test_should_mirror_sides_uses_both_orientations_across_fingerprints() -> None:
    decisions = {should_mirror_sides(f"fp-{index}", seed=0) for index in range(64)}
    assert decisions == {False, True}


def test_mirrored_side_a_win_flips_winner_first_label() -> None:
    assert mirrored_side_a_win(mirrored=False) is True
    assert mirrored_side_a_win(mirrored=True) is False


def test_laplace_probability_is_half_without_trials() -> None:
    assert laplace_probability(0, 0, alpha=1.0) == 0.5


def test_laplace_probability_shrinks_toward_half() -> None:
    assert laplace_probability(1, 1, alpha=1.0) == pytest.approx(2 / 3)


def test_card_log_odds_is_zero_without_trials() -> None:
    assert card_log_odds(0, 0, alpha=1.0) == pytest.approx(0.0)


def test_predict_card_log_odds_is_neutral_for_equal_decks() -> None:
    effects = {"knight:base": 1.2, "archers:base": -0.4}
    deck = ("knight:base", "archers:base")
    assert predict_card_log_odds(deck, deck, effects) == pytest.approx(0.5)


def test_predict_card_log_odds_inverts_when_sides_swap() -> None:
    effects = {"knight:base": 0.8}
    probability = predict_card_log_odds(("knight:base",), ("archers:base",), effects)
    swapped = predict_card_log_odds(("archers:base",), ("knight:base",), effects)
    assert probability > 0.5
    assert swapped == pytest.approx(1 - probability)


def test_predict_card_log_odds_ignores_unknown_identities() -> None:
    assert predict_card_log_odds(("missing:base",), ("also-missing:base",), {}) == pytest.approx(
        0.5
    )


def test_exact_matchup_probability_uses_smoothed_counts() -> None:
    counts = {("deck-a", "deck-b"): (3, 3)}
    assert exact_matchup_probability("deck-a", "deck-b", counts, alpha=1.0) == pytest.approx(0.8)


def test_exact_matchup_probability_falls_back_when_unseen() -> None:
    assert exact_matchup_probability("deck-a", "deck-b", {}, alpha=1.0) == 0.5
