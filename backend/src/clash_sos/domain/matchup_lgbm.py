"""Card-presence features for a side-symmetric LightGBM matchup probability.

Each deck keeps its own presence bits so a card played on both sides stays visible.
An attached attribute table adds seven deck summaries per side. The skill column
is a training control. Published probabilities set it to zero and average the two
orientations, so swapping decks inverts the probability.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from clash_sos.domain.card_attributes import SUMMARY_COLUMNS, CardAttributeTable
from clash_sos.domain.player_skill import PlayerSkillTracker

LIGHTGBM_FEATURE_SCHEMA_VERSION = "lightgbm-presence:v1"
DEFAULT_WATCH_FRACTION = 0.1
DEFAULT_NUM_THREADS = 4
DEFAULT_MAX_ROUNDS = 500
DEFAULT_EARLY_STOPPING_ROUNDS = 30
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_NUM_LEAVES = 63
DEFAULT_MIN_DATA_IN_LEAF = 200
DEFAULT_FEATURE_FRACTION = 0.8
DEFAULT_BAGGING_FRACTION = 0.8
DEFAULT_BAGGING_FREQ = 1
DEFAULT_LAMBDA_L2 = 1.0
DEFAULT_SEED = 0
CARDS_PER_SIDE = 8
RawPredict = Callable[[Sequence["PresenceRow"]], Sequence[float]]


@dataclass(frozen=True)
class PresenceRow:
    """One sparse row. Card columns are 1 and the last entry is the skill gap."""

    indices: tuple[int, ...]
    values: tuple[float, ...]


class PresenceSchema:
    """Column layout: side A, side B, optional deck summaries, then skill."""

    def __init__(
        self, identities: Sequence[str], attributes: CardAttributeTable | None = None
    ) -> None:
        keys = tuple(identities)
        if not keys:
            raise ValueError("identities are required")
        if len(keys) != len(set(keys)):
            raise ValueError("identities must be unique")
        if attributes is not None:
            for key in keys:
                attributes.for_identity(key)
        self.identities = keys
        self.attributes = attributes
        self._index = {key: index for index, key in enumerate(keys)}
        self.summary_width = 0 if attributes is None else len(SUMMARY_COLUMNS)
        self.summary_column = len(keys) * 2
        self.skill_column = self.summary_column + self.summary_width * 2
        self.feature_count = self.skill_column + 1

    def row(
        self, side_a: Sequence[str], side_b: Sequence[str], *, skill_diff: float
    ) -> PresenceRow:
        """Return presence bits for known identities plus the skill column.

        Unknown identities and repeated cards on one side are omitted. When an
        attribute table is attached, each side also stores its seven summaries,
        including zeros. The skill column is always stored, including when the
        gap is zero.
        """
        columns: list[tuple[int, float]] = []
        included_a = self._append_presence(side_a, columns, offset=0, seen=set())
        included_b = self._append_presence(side_b, columns, offset=len(self.identities), seen=set())
        if self.attributes is not None:
            self._append_summaries(columns, self.summary_column, included_a)
            self._append_summaries(columns, self.summary_column + self.summary_width, included_b)
        columns.append((self.skill_column, float(skill_diff)))
        columns.sort()
        return PresenceRow(
            indices=tuple(column for column, _value in columns),
            values=tuple(value for _column, value in columns),
        )

    def _append_presence(
        self,
        identities: Sequence[str],
        columns: list[tuple[int, float]],
        *,
        offset: int,
        seen: set[int],
    ) -> tuple[str, ...]:
        """Add presence bits and return the identities that were stored."""
        included: list[str] = []
        for key in identities:
            index = self._index.get(key)
            if index is None:
                continue
            column = offset + index
            if column in seen:
                continue
            seen.add(column)
            included.append(key)
            columns.append((column, 1.0))
        return tuple(included)

    def _append_summaries(
        self, columns: list[tuple[int, float]], start: int, identities: Sequence[str]
    ) -> None:
        """Store one summary block, including explicit zeros."""
        if self.attributes is None:
            return
        for slot, value in enumerate(self.attributes.summaries(identities)):
            columns.append((start + slot, value))


def symmetrized_probability(probability_ab: float, probability_ba: float) -> float:
    """Return 0.5 * (p(A, B) + 1 - p(B, A)). Equal raw scores cancel to 0.5."""
    return 0.5 * (probability_ab + 1.0 - probability_ba)


def predict_equal_skill(
    raw_predict: RawPredict,
    schema: PresenceSchema,
    side_a: Sequence[str],
    side_b: Sequence[str],
) -> float:
    """Predict P(side A wins) at skill gap 0, averaged across both orientations."""
    probabilities = raw_predict(
        (
            schema.row(side_a, side_b, skill_diff=0.0),
            schema.row(side_b, side_a, skill_diff=0.0),
        )
    )
    if len(probabilities) != 2:
        raise ValueError("raw predict must return one probability per orientation")
    return symmetrized_probability(probabilities[0], probabilities[1])


def skill_gap_then_observe(
    tracker: PlayerSkillTracker,
    player_a: str,
    player_b: str,
    label: int,
) -> float:
    """Return r_a - r_b from battles already observed, then record this outcome.

    Side A won when label is 1. The current battle is not part of the returned gap.
    """
    if label not in (0, 1):
        raise ValueError("labels must be 0 or 1")
    gap = tracker.rating(player_a) - tracker.rating(player_b)
    tracker.observe(player_a, won=label == 1)
    tracker.observe(player_b, won=label == 0)
    return gap


def past_only_skill_gaps(
    rows: Iterable[tuple[str, str, int]],
    tracker: PlayerSkillTracker,
) -> list[float]:
    """Return one past-only skill gap per (player_a, player_b, label) row."""
    return [
        skill_gap_then_observe(tracker, player_a, player_b, label)
        for player_a, player_b, label in rows
    ]


def watch_row_count(row_count: int, fraction: float = DEFAULT_WATCH_FRACTION) -> int:
    """Return how many trailing temporal-train rows choose the boosting round count.

    The tail is only a watch set. At least one row stays in the fit prefix.
    """
    if row_count < 2:
        raise ValueError("early stopping requires at least two training rows")
    if not 0 < fraction < 1:
        raise ValueError("watch fraction must be between 0 and 1")
    watch = int(row_count * fraction)
    if watch < 1:
        return 1
    if watch >= row_count:
        return row_count - 1
    return watch


def monotone_constraints(identity_count: int, summary_count: int = 0) -> list[int]:
    """Return LightGBM constraints. Card and summary columns are free.

    The skill gap is the last column and is nondecreasing.
    """
    if identity_count < 1:
        raise ValueError("identity count must be positive")
    if summary_count < 0:
        raise ValueError("summary count must be non-negative")
    return [0] * (identity_count * 2 + summary_count * 2) + [1]


def lightgbm_training_params(
    *,
    identity_count: int,
    summary_count: int = 0,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_leaves: int = DEFAULT_NUM_LEAVES,
    min_data_in_leaf: int = DEFAULT_MIN_DATA_IN_LEAF,
    feature_fraction: float = DEFAULT_FEATURE_FRACTION,
    bagging_fraction: float = DEFAULT_BAGGING_FRACTION,
    bagging_freq: int = DEFAULT_BAGGING_FREQ,
    lambda_l2: float = DEFAULT_LAMBDA_L2,
    num_threads: int = DEFAULT_NUM_THREADS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Return the fixed binary-classification preset. Validation is not a search target."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if num_leaves < 2:
        raise ValueError("num_leaves must be at least 2")
    if min_data_in_leaf < 1:
        raise ValueError("min_data_in_leaf must be positive")
    if not 0 < feature_fraction <= 1:
        raise ValueError("feature_fraction must be in (0, 1]")
    if not 0 < bagging_fraction <= 1:
        raise ValueError("bagging_fraction must be in (0, 1]")
    if bagging_freq < 0:
        raise ValueError("bagging_freq must be non-negative")
    if lambda_l2 < 0:
        raise ValueError("lambda_l2 must be non-negative")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    return {
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "deterministic": True,
        "feature_fraction": feature_fraction,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "lambda_l2": lambda_l2,
        "learning_rate": learning_rate,
        "metric": "binary_logloss",
        "min_data_in_leaf": min_data_in_leaf,
        "monotone_constraints": monotone_constraints(identity_count, summary_count),
        "num_leaves": num_leaves,
        "num_threads": num_threads,
        "objective": "binary",
        "seed": seed,
        "verbose": -1,
    }
