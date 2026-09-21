import pytest

from clash_sos.domain.matchup_baseline import card_log_odds
from clash_sos.domain.player_skill import DEFAULT_SKILL_ALPHA, PlayerSkillTracker


def test_skill_alpha_is_eight_pseudo_counts() -> None:
    assert DEFAULT_SKILL_ALPHA == 8.0


def test_unseen_player_rating_is_zero() -> None:
    tracker = PlayerSkillTracker()
    assert tracker.rating("#NEW") == pytest.approx(0.0)


def test_rating_ignores_the_battle_until_observe() -> None:
    tracker = PlayerSkillTracker()
    before = tracker.rating("#A") - tracker.rating("#B")
    tracker.observe("#A", won=True)
    tracker.observe("#B", won=False)
    after = tracker.rating("#A") - tracker.rating("#B")
    assert before == pytest.approx(0.0)
    assert after > 0
    assert tracker.rating("#A") == pytest.approx(card_log_odds(1, 1, alpha=8.0))
    assert tracker.rating("#B") == pytest.approx(card_log_odds(0, 1, alpha=8.0))


def test_skill_tracker_rejects_blank_player_ids() -> None:
    tracker = PlayerSkillTracker()
    with pytest.raises(ValueError, match="player id"):
        tracker.rating("")
    with pytest.raises(ValueError, match="player id"):
        tracker.observe("", won=True)


def test_skill_tracker_rejects_non_positive_alpha() -> None:
    with pytest.raises(ValueError, match="skill alpha"):
        PlayerSkillTracker(alpha=0)
