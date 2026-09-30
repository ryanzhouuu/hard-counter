"""Common final-N player windows using production expected-win arithmetic."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from clash_sos.application.rolling_sos import summarize_observations
from clash_sos.domain.analytics import (
    PredictionProvenance,
    RollingSoSObservation,
    RollingSoSResult,
)
from experiments.common.predictions import Prediction, pair_predictions


@dataclass(frozen=True)
class PlayerWindowShift:
    player: str
    window_size: int
    candidate: RollingSoSResult
    comparator: RollingSoSResult
    expected_win_shift: float | None
    difficulty_shift: float | None


@dataclass(frozen=True)
class WindowSummary:
    window_size: int
    player_count: int
    complete_count: int
    incomplete_count: int
    mean_signed_expected_win_shift: float | None
    mean_absolute_expected_win_shift: float | None
    p50_absolute_expected_win_shift: float | None
    p90_absolute_expected_win_shift: float | None
    mean_difficulty_shift: float | None


def _observation(
    row: Prediction, player: str, provenance: PredictionProvenance
) -> RollingSoSObservation:
    first = player == row.player_a
    probability = row.probability if first else 1 - row.probability
    return RollingSoSObservation(
        timestamp=row.timestamp,
        battle_fingerprint=row.row_key[1],
        player_win_probability=probability,
        actual_win=row.label if first else 1 - row.label,
        difficulty=1 - probability,
        provenance=provenance,
    )


def compare_windows(
    candidate: Sequence[Prediction],
    comparator: Sequence[Prediction],
    *,
    candidate_provenance: PredictionProvenance,
    comparator_provenance: PredictionProvenance,
    window_sizes: tuple[int, ...] = (10, 25),
) -> tuple[PlayerWindowShift, ...]:
    """Caller supplies one declared interval; coverage mismatches fail before window selection."""
    if not window_sizes or min(window_sizes) <= 0 or len(set(window_sizes)) != len(window_sizes):
        raise ValueError("window sizes must be unique positive integers")
    aligned = pair_predictions(candidate, comparator)
    players: defaultdict[str, list[tuple[Prediction, Prediction]]] = defaultdict(list)
    for pair in aligned:
        players[pair[0].player_a].append(pair)
        players[pair[0].player_b].append(pair)
    output: list[PlayerWindowShift] = []
    for size in window_sizes:
        for player in sorted(players):
            first = summarize_observations(
                player,
                (_observation(a, player, candidate_provenance) for a, _ in players[player]),
                window_size=size,
            )
            second = summarize_observations(
                player,
                (_observation(b, player, comparator_provenance) for _, b in players[player]),
                window_size=size,
            )
            win_shift = (
                None
                if first.expected_wins is None or second.expected_wins is None
                else (first.expected_wins - second.expected_wins)
            )
            difficulty_shift = (
                None
                if (first.strength_of_schedule is None or second.strength_of_schedule is None)
                else first.strength_of_schedule - second.strength_of_schedule
            )
            output.append(
                PlayerWindowShift(player, size, first, second, win_shift, difficulty_shift)
            )
    return tuple(output)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_windows(rows: Sequence[PlayerWindowShift]) -> tuple[WindowSummary, ...]:
    output: list[WindowSummary] = []
    for size in sorted({row.window_size for row in rows}):
        selected = tuple(row for row in rows if row.window_size == size)
        shifts = tuple(
            row.expected_win_shift for row in selected if row.expected_win_shift is not None
        )
        difficulty = tuple(
            row.difficulty_shift for row in selected if row.difficulty_shift is not None
        )
        absolute = tuple(abs(value) for value in shifts)
        count = len(shifts)
        output.append(
            WindowSummary(
                size,
                len(selected),
                count,
                len(selected) - count,
                sum(shifts) / count if count else None,
                sum(absolute) / count if count else None,
                _quantile(absolute, 0.5) if count else None,
                _quantile(absolute, 0.9) if count else None,
                sum(difficulty) / count if count else None,
            )
        )
    return tuple(output)


@dataclass(frozen=True)
class WindowSeedVariability:
    player: str
    window_size: int
    seed_count: int
    expected_win_shift_standard_deviation: float | None
    expected_win_shift_span: float | None


def window_seed_variability(
    seeds: Sequence[Sequence[PlayerWindowShift]],
) -> tuple[WindowSeedVariability, ...]:
    """Seeds share the evaluation population; dispersion is not sampling uncertainty."""
    if not seeds:
        raise ValueError("window variability requires at least one seed")
    indexed: list[dict[tuple[str, int], PlayerWindowShift]] = []
    for rows in seeds:
        mapping = {(row.player, row.window_size): row for row in rows}
        if len(mapping) != len(rows) or (indexed and mapping.keys() != indexed[0].keys()):
            raise ValueError("seed windows must have identical unique player/window memberships")
        indexed.append(mapping)
    output: list[WindowSeedVariability] = []
    for key in sorted(indexed[0]):
        rows = tuple(seed[key] for seed in indexed)
        reference = rows[0]
        for row in rows:
            for result, original in (
                (row.candidate, reference.candidate),
                (row.comparator, reference.comparator),
            ):
                identity = tuple(
                    (item.timestamp, item.battle_fingerprint, item.actual_win)
                    for item in result.window
                )
                expected = tuple(
                    (item.timestamp, item.battle_fingerprint, item.actual_win)
                    for item in original.window
                )
                if identity != expected or result.eligible_count != original.eligible_count:
                    raise ValueError("seed windows must use identical eligible observations")
        shifts = tuple(row.expected_win_shift for row in rows if row.expected_win_shift is not None)
        if shifts and len(shifts) != len(rows):
            raise ValueError("seed windows must share completion status")
        mean = sum(shifts) / len(shifts) if shifts else 0.0
        deviation = (
            (sum((value - mean) ** 2 for value in shifts) / len(shifts)) ** 0.5 if shifts else None
        )
        output.append(
            WindowSeedVariability(
                key[0],
                key[1],
                len(seeds),
                deviation,
                max(shifts) - min(shifts) if shifts else None,
            )
        )
    return tuple(output)
