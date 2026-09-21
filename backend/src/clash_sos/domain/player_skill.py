"""Past-only Laplace player ratings for matchup training.

Ratings include only battles passed to observe. Callers rate a battle before
observing it so the current outcome cannot explain itself.
"""

from clash_sos.domain.matchup_baseline import card_log_odds

DEFAULT_SKILL_ALPHA = 8.0


class PlayerSkillTracker:
    """Shrunk log-odds win rate per player."""

    def __init__(self, *, alpha: float = DEFAULT_SKILL_ALPHA) -> None:
        if alpha <= 0:
            raise ValueError("skill alpha must be positive")
        self.alpha = alpha
        self._wins: dict[str, int] = {}
        self._trials: dict[str, int] = {}

    def rating(self, player_id: str) -> float:
        """Return the log-odds win rate from prior observe calls. Unseen players are 0."""
        if not player_id:
            raise ValueError("player id is required")
        return card_log_odds(
            self._wins.get(player_id, 0),
            self._trials.get(player_id, 0),
            alpha=self.alpha,
        )

    def observe(self, player_id: str, *, won: bool) -> None:
        """Record one finished battle for a player."""
        if not player_id:
            raise ValueError("player id is required")
        self._trials[player_id] = self._trials.get(player_id, 0) + 1
        if won:
            self._wins[player_id] = self._wins.get(player_id, 0) + 1
