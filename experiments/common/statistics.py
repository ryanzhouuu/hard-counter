"""Production probability metrics and shared-player paired uncertainty."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

from clash_sos.domain.model_artifact import log_loss
from clash_sos.domain.model_evaluation import (
    MatchupEvaluationAccumulator,
    MatchupScoreSummary,
)
from experiments.common.predictions import Prediction, pair_predictions


@dataclass(frozen=True)
class DyadicInterval:
    mean_difference: float
    variance: float
    row_count: int
    player_count: int
    unordered_pair_count: int
    max_player_match_fraction: float
    small_player_count: bool
    status: Literal["ok", "zero_variance_estimate", "negative_variance_estimate"]
    standard_error: float | None
    ci95: tuple[float, float] | None


def dyadic_interval(
    differences: Sequence[float], player_a: Sequence[str], player_b: Sequence[str]
) -> DyadicInterval:
    """Unadjusted intercept-only sandwich; day/common-meta dependence is not modeled.

    Player score sums minus unordered-pair score sums count shared-player terms once.
    https://arxiv.org/abs/1312.3398
    """
    count = len(differences)
    if count < 2 or len(player_a) != count or len(player_b) != count:
        raise ValueError("dyadic intervals require at least two aligned rows")
    if any(not isfinite(value) for value in differences):
        raise ValueError("paired differences must be finite")
    mean = sum(differences) / count
    if not isfinite(mean):
        raise ValueError("paired difference mean is nonfinite")
    players: defaultdict[str, float] = defaultdict(float)
    pairs: defaultdict[tuple[str, str], float] = defaultdict(float)
    incidence: Counter[str] = Counter()
    for value, first, second in zip(differences, player_a, player_b, strict=True):
        if not first or not second or first == second:
            raise ValueError("dyadic rows require distinct nonempty players")
        residual = value - mean
        players[first] += residual
        players[second] += residual
        pair = (first, second) if first < second else (second, first)
        pairs[pair] += residual
        incidence.update((first, second))
    variance = (
        sum(value**2 for value in players.values()) - sum(value**2 for value in pairs.values())
    ) / count**2
    if not isfinite(variance):
        raise ValueError("dyadic variance is nonfinite")
    standard_error = sqrt(variance) if variance >= 0 else None
    interval = (
        None
        if standard_error is None
        else (
            mean - 1.959963984540054 * standard_error,
            mean + 1.959963984540054 * standard_error,
        )
    )
    status = (
        "negative_variance_estimate"
        if variance < 0
        else "zero_variance_estimate"
        if variance == 0
        else "ok"
    )
    return DyadicInterval(
        mean,
        variance,
        count,
        len(players),
        len(pairs),
        max(incidence.values()) / count,
        len(players) < 30,
        status,
        standard_error,
        interval,
    )


def score_predictions(rows: Sequence[Prediction]) -> MatchupScoreSummary:
    accumulator = MatchupEvaluationAccumulator()
    for row in rows:
        accumulator.update(
            label=row.label,
            probability=row.probability,
            deck_a_support=0,
            deck_b_support=0,
            unordered_pair_support=0,
        )
    return accumulator.finalize().overall


@dataclass(frozen=True)
class PairedComparison:
    candidate: MatchupScoreSummary
    comparator: MatchupScoreSummary
    log_loss: DyadicInterval | None
    brier_score: DyadicInterval | None


def paired_comparison(
    candidate: Sequence[Prediction], comparator: Sequence[Prediction]
) -> PairedComparison:
    aligned = pair_predictions(candidate, comparator)
    if not aligned:
        raise ValueError("paired comparison requires nonempty rows")
    first = tuple(pair[0] for pair in aligned)
    second = tuple(pair[1] for pair in aligned)
    log_differences = tuple(
        log_loss((a.label,), (a.probability,)) - log_loss((b.label,), (b.probability,))
        for a, b in aligned
    )
    brier_differences = tuple(
        (a.probability - a.label) ** 2 - (b.probability - b.label) ** 2 for a, b in aligned
    )
    players_a = tuple(row.player_a for row in first)
    players_b = tuple(row.player_b for row in first)
    return PairedComparison(
        score_predictions(first),
        score_predictions(second),
        dyadic_interval(log_differences, players_a, players_b) if len(aligned) > 1 else None,
        dyadic_interval(brier_differences, players_a, players_b) if len(aligned) > 1 else None,
    )


def holm_adjust(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    """Keep unsupported tests in the predeclared family without claiming significance."""
    if any(
        value is not None and (not isfinite(value) or not 0 <= value <= 1)
        for value in pvalues.values()
    ):
        raise ValueError("p-values must be finite and within zero/one")
    numeric = {name: 1.0 if value is None else value for name, value in pvalues.items()}
    ordered = sorted(numeric, key=lambda name: numeric[name])
    adjusted: dict[str, float | None] = {}
    previous = 0.0
    for index, name in enumerate(ordered):
        value = pvalues[name]
        previous = max(
            previous, min(1.0, (len(ordered) - index) * (1.0 if value is None else value))
        )
        adjusted[name] = None if value is None else previous
    return adjusted
