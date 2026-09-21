"""Antisymmetric card-pair logistic matchup model.

Fits additive card effects plus an antisymmetric card-vs-card matrix so swapping
sides inverts P(side A wins) and equal decks are 0.5. Training streams labeled
decks; this module does not read Parquet or DuckDB.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Self, cast

from clash_sos.domain.matchup_baseline import (
    DEFAULT_SMOOTHING_ALPHA,
    card_log_odds,
    laplace_probability,
    logit,
    sigmoid,
)

DEFAULT_PAIR_LEARNING_RATE = 0.01
DEFAULT_PAIR_L2 = 1e-5
DEFAULT_PAIR_EPOCHS = 2
DEFAULT_PAIR_INIT_SCALE = 8.0
PAIR_FEATURE_SCHEMA_VERSION = "card-pair:v1"

LabeledDeck = tuple[int, Sequence[str], Sequence[str]]


def _string_list(value: object, field: str) -> tuple[str, ...]:
    """Parse a JSON string array. Raises ValueError when the field is malformed."""
    if not isinstance(value, list):
        raise ValueError(f"predictor {field} must be a list of strings")
    items: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            raise ValueError(f"predictor {field} must be a list of strings")
        items.append(item)
    return tuple(items)


def _number_list(value: object, field: str) -> list[float]:
    """Parse a JSON number array. Raises ValueError when the field is malformed."""
    if not isinstance(value, list):
        raise ValueError(f"predictor {field} must be a list of numbers")
    items: list[float] = []
    for item in cast(list[object], value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"predictor {field} must be a list of numbers")
        items.append(float(item))
    return items


def pair_upper_length(n: int) -> int:
    """Return the number of i < j entries in an n-by-n antisymmetric matrix."""
    if n < 0:
        raise ValueError("identity count must be non-negative")
    return n * (n - 1) // 2


def pair_upper_index(i: int, j: int, n: int) -> int:
    """Return the row-major index of pair (i, j) with i < j."""
    if not 0 <= i < j < n:
        raise ValueError("pair index requires 0 <= i < j < n")
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def accumulate_pair_counts(
    labeled_decks: Iterable[LabeledDeck],
) -> dict[tuple[str, str], tuple[int, int]]:
    """Return directed (wins, trials) for every side-A vs side-B identity pair."""
    wins: dict[tuple[str, str], int] = defaultdict(int)
    trials: dict[tuple[str, str], int] = defaultdict(int)
    for label, side_a, side_b in labeled_decks:
        if label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        for card_a in side_a:
            for card_b in side_b:
                key = (card_a, card_b)
                trials[key] += 1
                wins[key] += label
    return {key: (wins[key], trials[key]) for key in trials}


class CardPairPredictor:
    """P(side A wins) from additive effects plus antisymmetric pair weights."""

    def __init__(
        self,
        identities: Sequence[str],
        additive: Sequence[float],
        pair_upper: Sequence[float],
    ) -> None:
        keys = tuple(identities)
        if len(keys) != len(set(keys)):
            raise ValueError("identities must be unique")
        n = len(keys)
        if len(additive) != n:
            raise ValueError("additive length must match identities")
        if len(pair_upper) != pair_upper_length(n):
            raise ValueError("pair_upper length must match n choose 2")
        self.identities = keys
        self.additive = [float(value) for value in additive]
        self.pair_upper = [float(value) for value in pair_upper]
        self._index = {key: index for index, key in enumerate(keys)}

    def pair_weight(self, i: int, j: int) -> float:
        """Return W_ij. Diagonal is 0 and W_ji is -W_ij."""
        n = len(self.identities)
        if not 0 <= i < n or not 0 <= j < n:
            raise ValueError("pair indices must be in range")
        if i == j:
            return 0.0
        if i < j:
            return self.pair_upper[pair_upper_index(i, j, n)]
        return -self.pair_upper[pair_upper_index(j, i, n)]

    def _resolved(self, side: Sequence[str]) -> list[int]:
        return [self._index[key] for key in side if key in self._index]

    def score(self, side_a: Sequence[str], side_b: Sequence[str]) -> float:
        """Return matchup log-odds. Missing identities contribute 0."""
        a_idx = self._resolved(side_a)
        b_idx = self._resolved(side_b)
        total = 0.0
        for index in a_idx:
            total += self.additive[index]
        for index in b_idx:
            total -= self.additive[index]
        for i in a_idx:
            for j in b_idx:
                total += self.pair_weight(i, j)
        return total

    def predict(self, side_a: Sequence[str], side_b: Sequence[str]) -> float:
        """Return P(side A wins). Equal decks are 0.5; swapping sides inverts p."""
        return sigmoid(self.score(side_a, side_b))

    def sgd_step(
        self,
        side_a: Sequence[str],
        side_b: Sequence[str],
        *,
        label: int,
        learning_rate: float = DEFAULT_PAIR_LEARNING_RATE,
        l2: float = DEFAULT_PAIR_L2,
    ) -> None:
        """Take one log-loss + L2 step. Updates only identities present in the row."""
        if label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        error = self.predict(side_a, side_b) - label
        a_idx = self._resolved(side_a)
        b_idx = self._resolved(side_b)
        n = len(self.identities)
        for index in a_idx:
            self.additive[index] -= learning_rate * (error + l2 * self.additive[index])
        for index in b_idx:
            self.additive[index] -= learning_rate * (-error + l2 * self.additive[index])
        for i in a_idx:
            for j in b_idx:
                if i == j:
                    continue
                low, high = (i, j) if i < j else (j, i)
                slot = pair_upper_index(low, high, n)
                sign = 1.0 if i < j else -1.0
                self.pair_upper[slot] -= learning_rate * (sign * error + l2 * self.pair_upper[slot])

    def to_payload(self) -> dict[str, object]:
        """Return canonical predictor JSON fields."""
        return {
            "additive": list(self.additive),
            "identities": list(self.identities),
            "pair_upper": list(self.pair_upper),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Self:
        """Load a predictor written by to_payload. Raises ValueError if malformed."""
        return cls(
            _string_list(payload.get("identities"), "identities"),
            _number_list(payload.get("additive"), "additive"),
            _number_list(payload.get("pair_upper"), "pair_upper"),
        )


def initialize_card_pair_predictor(
    identities: Sequence[str],
    card_counts: Mapping[str, tuple[int, int]],
    pair_counts: Mapping[tuple[str, str], tuple[int, int]],
    *,
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
    pair_init_scale: float = DEFAULT_PAIR_INIT_SCALE,
) -> CardPairPredictor:
    """Initialize u from card Laplace log-odds and W from antisymmetric pair logits / scale."""
    if pair_init_scale <= 0:
        raise ValueError("pair_init_scale must be positive")
    keys = tuple(identities)
    additive = [card_log_odds(*card_counts.get(key, (0, 0)), alpha=alpha) for key in keys]
    pair_upper: list[float] = []
    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            wins_ij, trials_ij = pair_counts.get((keys[i], keys[j]), (0, 0))
            wins_ji, trials_ji = pair_counts.get((keys[j], keys[i]), (0, 0))
            forward = logit(laplace_probability(wins_ij, trials_ij, alpha=alpha))
            reverse = logit(laplace_probability(wins_ji, trials_ji, alpha=alpha))
            pair_upper.append(0.5 * (forward - reverse) / pair_init_scale)
    return CardPairPredictor(keys, additive, pair_upper)
