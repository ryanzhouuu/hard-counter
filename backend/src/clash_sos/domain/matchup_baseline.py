"""Deterministic side mirroring and card log-odds matchup baseline.

Canonical Kaggle rows are winner-first, so training labels are always side-A wins
until this module mirrors sides. Card identities are `card_id:form` keys.
"""

from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import exp, log

DEFAULT_SMOOTHING_ALPHA = 1.0
DEFAULT_MIRROR_SEED = 0
PROBABILITY_FLOOR = 1e-15


def card_identity_key(card_id: str, form: str) -> str:
    """Return the catalog identity used as a baseline feature key."""
    if not card_id or not form:
        raise ValueError("card identity requires card_id and form")
    return f"{card_id}:{form}"


def should_mirror_sides(fingerprint: str, *, seed: int = DEFAULT_MIRROR_SEED) -> bool:
    """Return whether a winner-first row should swap sides for a seed."""
    if not fingerprint:
        raise ValueError("fingerprint is required")
    digest = sha256(f"{seed}:{fingerprint}".encode()).digest()
    return digest[0] % 2 == 1


def mirrored_side_a_win(*, mirrored: bool) -> bool:
    """Winner-first rows label side A as the winner unless the row is mirrored."""
    return not mirrored


def _clip_probability(probability: float) -> float:
    return min(max(probability, PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)


def laplace_probability(wins: int, trials: int, *, alpha: float = DEFAULT_SMOOTHING_ALPHA) -> float:
    """Return a Laplace-smoothed win rate. Unseen pairs are 0.5 when alpha is 1."""
    if wins < 0 or trials < 0 or wins > trials:
        raise ValueError("wins must be between 0 and trials inclusive")
    if alpha <= 0:
        raise ValueError("smoothing alpha must be positive")
    return (wins + alpha) / (trials + 2 * alpha)


def logit(probability: float) -> float:
    """Return log-odds for a clipped probability in (0, 1)."""
    clipped = _clip_probability(probability)
    return log(clipped / (1.0 - clipped))


def sigmoid(log_odds: float) -> float:
    """Return a probability from log-odds without overflowing exp."""
    if log_odds >= 0:
        return 1.0 / (1.0 + exp(-log_odds))
    exponential = exp(log_odds)
    return exponential / (1.0 + exponential)


def card_log_odds(wins: int, trials: int, *, alpha: float = DEFAULT_SMOOTHING_ALPHA) -> float:
    """Return the additive card effect used by the promoted baseline."""
    return logit(laplace_probability(wins, trials, alpha=alpha))


def predict_card_log_odds(
    side_a: Sequence[str],
    side_b: Sequence[str],
    effects: Mapping[str, float],
) -> float:
    """Predict P(side A wins) from additive card effects. Missing keys contribute 0."""
    score = sum(effects.get(key, 0.0) for key in side_a) - sum(
        effects.get(key, 0.0) for key in side_b
    )
    return sigmoid(score)


def exact_matchup_probability(
    deck_a_hash: str,
    deck_b_hash: str,
    counts: Mapping[tuple[str, str], tuple[int, int]],
    *,
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> float:
    """Return a smoothed exact-deck probability, or 0.5 when the pair is unseen."""
    wins, trials = counts.get((deck_a_hash, deck_b_hash), (0, 0))
    if trials == 0:
        return 0.5
    return laplace_probability(wins, trials, alpha=alpha)
