"""Covariate support and descriptive player-identification limitations."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from clash_sos.domain.manifests import ManifestModel
from experiments.common.data_access import ResearchRow
from experiments.player_adjustment.history import FrozenHistory
from experiments.player_adjustment.joint import FrozenPlayerEffects, training_vocabulary


class PlayerSupport(ManifestModel):
    player: str
    count: int
    distinct_decks: int
    deck_switches: int
    dominant_deck_share: float
    component: int


class TrainingDiagnostics(ManifestModel):
    players: tuple[PlayerSupport, ...]
    component_sizes: tuple[int, ...]
    single_deck_players: int
    warnings: tuple[str, ...]


def training_diagnostics(rows: Sequence[ResearchRow]) -> TrainingDiagnostics:
    """Single-deck assignments expose confounding risk without proving identification."""
    graph = training_vocabulary(rows)
    components = {player: index for index, group in enumerate(graph.components) for player in group}
    assignments: dict[str, list[tuple[ResearchRow, tuple[int, ...]]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: item.key):
        for player, deck in ((row.player_a, row.tokens[0]), (row.player_b, row.tokens[1])):
            lineup = (*sorted(deck[:-1]), deck[-1]) if len(deck) == 9 else tuple(sorted(deck))
            assignments[player].append((row, lineup))
    players: list[PlayerSupport] = []
    for player in graph.players:
        observations = assignments[player]
        decks = Counter(deck for _, deck in observations)
        switches = sum(
            first.key[0] < second.key[0] and old != new
            for (first, old), (second, new) in pairwise(observations)
        )
        players.append(
            PlayerSupport(
                player=player,
                count=len(observations),
                distinct_decks=len(decks),
                deck_switches=switches,
                dominant_deck_share=max(decks.values()) / len(observations),
                component=components[player],
            )
        )
    single = sum(player.distinct_decks == 1 for player in players)
    warnings = ["Shrinkage and centering do not establish separation of player and deck effects."]
    if single:
        warnings.append(
            "Some training players use one lineup; player/deck confounding is possible."
        )
    if len(graph.components) > 1:
        warnings.append(
            "Opponent graph is disconnected; effects across components are constrained."
        )
    return TrainingDiagnostics(
        players=tuple(players),
        component_sizes=tuple(len(group) for group in graph.components),
        single_deck_players=single,
        warnings=tuple(warnings),
    )


class EvaluationSupport(ManifestModel):
    player_a_seen: bool
    player_b_seen: bool
    player_a_prior_count: int
    player_b_prior_count: int
    player_a_missing_history: bool
    player_b_missing_history: bool
    same_training_component: bool | None


class SupportBins(ManifestModel):
    prior_history_edges: tuple[int, ...]
    deck_switching_edges: tuple[int, ...]

    @model_validator(mode="after")
    def validate_edges(self) -> Self:
        for edges in (self.prior_history_edges, self.deck_switching_edges):
            if not edges or any(edge < 0 for edge in edges) or tuple(sorted(set(edges))) != edges:
                raise ValueError("support boundaries must be nonnegative, unique, and increasing")
        return self


def support_bin_counts(
    rows: Sequence[ResearchRow],
    history: FrozenHistory,
    diagnostics: TrainingDiagnostics,
    bins: SupportBins,
) -> dict[str, tuple[int, ...]]:
    """Bins include lower bounds; seen_pair order is neither, B-only, A-only, both."""
    prior = {entry.player: entry.count for entry in history.players}
    switches = {entry.player: entry.deck_switches for entry in diagnostics.players}
    counts = {
        "prior_history": [0] * (len(bins.prior_history_edges) + 1),
        "deck_switching": [0] * (len(bins.deck_switching_edges) + 1),
        "seen_pair": [0] * 4,
    }
    for row in rows:
        counts["seen_pair"][2 * int(row.player_a in switches) + int(row.player_b in switches)] += 1
        for player in (row.player_a, row.player_b):
            counts["prior_history"][
                sum(prior.get(player, 0) >= edge for edge in bins.prior_history_edges)
            ] += 1
            counts["deck_switching"][
                sum(switches.get(player, 0) >= edge for edge in bins.deck_switching_edges)
            ] += 1
    return {name: tuple(values) for name, values in counts.items()}


def matchup_probability_stability(
    reference: Sequence[float], candidate: Sequence[float]
) -> dict[str, float]:
    first, second = np.asarray(reference), np.asarray(candidate)
    if first.ndim != 1 or first.shape != second.shape or not len(first):
        raise ValueError("stability requires nonempty aligned probabilities")
    if any(
        not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
        for values in (first, second)
    ):
        raise ValueError("stability probabilities must be finite and within zero/one")
    delta = second - first
    return {
        "mean_absolute_probability_change": float(np.mean(np.abs(delta))),
        "rms_probability_change": float(np.sqrt(np.mean(delta * delta))),
        "maximum_absolute_probability_change": float(np.max(np.abs(delta))),
    }


def evaluation_support(
    rows: Sequence[ResearchRow], history: FrozenHistory, diagnostics: TrainingDiagnostics
) -> tuple[EvaluationSupport, ...]:
    counts = {entry.player: entry.count for entry in history.players}
    components = {entry.player: entry.component for entry in diagnostics.players}
    return tuple(
        EvaluationSupport(
            player_a_seen=row.player_a in components,
            player_b_seen=row.player_b in components,
            player_a_prior_count=counts.get(row.player_a, 0),
            player_b_prior_count=counts.get(row.player_b, 0),
            player_a_missing_history=row.player_a not in counts,
            player_b_missing_history=row.player_b not in counts,
            same_training_component=(
                components[row.player_a] == components[row.player_b]
                if row.player_a in components and row.player_b in components
                else None
            ),
        )
        for row in rows
    )


def shrinkage_diagnostics(effects: FrozenPlayerEffects) -> dict[str, float]:
    values = np.asarray(effects.effects, dtype=np.float64)
    return {
        "mean_squared_effect": float(np.mean(values * values)),
        "maximum_absolute_effect": float(np.max(np.abs(values))),
        "penalty_normalization": len(values) * 2.0,
    }


class ResidualPlayerDiagnostic(ManifestModel):
    interpretation: Literal["descriptive-development-residual"] = "descriptive-development-residual"
    player: str
    count: int = Field(gt=0)
    mean_residual: float


def residual_player_diagnostic(
    rows: Sequence[ResearchRow], probabilities: Sequence[float], *, role: str
) -> tuple[ResidualPlayerDiagnostic, ...]:
    if role != "development":
        raise ValueError("residual diagnostics may read development outcomes only")
    if len(rows) != len(probabilities) or any(
        not np.isfinite(value) or not 0 <= value <= 1 for value in probabilities
    ):
        raise ValueError("residual probabilities must be finite and row aligned")
    residuals: dict[str, list[float]] = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        residual = row.label - probability
        residuals[row.player_a].append(residual)
        residuals[row.player_b].append(-residual)
    return tuple(
        ResidualPlayerDiagnostic(
            player=player, count=len(values), mean_residual=sum(values) / len(values)
        )
        for player, values in sorted(residuals.items())
    )
