"""Predeclared comparator-defined slice membership, independent of candidate outcomes."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from clash_sos.domain.attention_protocol import RowKey
from clash_sos.domain.model_evaluation import PROBABILITY_BANDS
from experiments.common.predictions import Prediction, index_predictions, pair_predictions
from experiments.common.statistics import PairedComparison, paired_comparison


@dataclass(frozen=True)
class SliceMetadata:
    row_key: RowKey
    era: str
    deck_a_support: int
    deck_b_support: int
    pair_support: int
    player_a_support: int = 0
    player_b_support: int = 0
    history_a_count: int = 0
    history_b_count: int = 0
    forms: tuple[str, ...] = ()
    tower_a: str = ""
    tower_b: str = ""

    def __post_init__(self) -> None:
        if (
            min(
                self.deck_a_support,
                self.deck_b_support,
                self.pair_support,
                self.player_a_support,
                self.player_b_support,
                self.history_a_count,
                self.history_b_count,
            )
            < 0
        ):
            raise ValueError("slice support and history counts must be nonnegative")
        if not self.era:
            raise ValueError("slice era must be declared")


@dataclass(frozen=True)
class SliceSpec:
    days: tuple[date, ...] = ()
    eras: tuple[str, ...] = ()
    forms: tuple[str, ...] = ()
    towers: tuple[str, ...] = ()
    history_boundaries: tuple[int, ...] = (0, 1, 10)

    def __post_init__(self) -> None:
        if (
            not self.history_boundaries
            or self.history_boundaries[0] != 0
            or (tuple(sorted(set(self.history_boundaries))) != self.history_boundaries)
        ):
            raise ValueError("history bins require unique increasing boundaries starting at zero")
        for values in (self.days, self.eras, self.forms, self.towers):
            if len(set(values)) != len(values):
                raise ValueError("slice definitions must be unique")


@dataclass(frozen=True)
class FixedSlices:
    population: tuple[RowKey, ...]
    members: tuple[tuple[str, tuple[RowKey, ...]], ...]


def build_slices(
    metadata: Sequence[SliceMetadata], comparator: Sequence[Prediction], spec: SliceSpec
) -> FixedSlices:
    predictions = index_predictions(comparator)
    indexed = {row.row_key: row for row in metadata}
    if len(indexed) != len(metadata) or indexed.keys() != predictions.keys():
        raise ValueError("slice metadata must join one-to-one to comparator rows")
    members: dict[str, list[RowKey]] = {}
    names = (
        "overall",
        "deck:both_unseen",
        "deck:one_unseen",
        "deck:both_seen",
        "pair:unseen_pair",
        "pair:seen_pair",
        "support:0",
        "support:1_3",
        "support:4_19",
        "support:20_plus",
        "player:both_unseen",
        "player:one_unseen",
        "player:both_seen",
        "form:present",
        "form:absent",
        *(f"probability:band_{index}" for index in range(len(PROBABILITY_BANDS))),
        *(f"history:band_{index}" for index in range(len(spec.history_boundaries))),
        *(f"day:{day.isoformat()}" for day in spec.days),
        *(f"era:{era}" for era in spec.eras),
        *(f"form:identity:{form}" for form in spec.forms),
        *(f"tower:{tower}" for tower in spec.towers),
    )
    for name in names:
        members[name] = []
    novelty = ("both_unseen", "one_unseen", "both_seen")
    for key in sorted(predictions):
        row = indexed[key]
        support = min(row.deck_a_support, row.deck_b_support)
        support_name = (
            "0"
            if support == 0
            else "1_3"
            if support < 4
            else ("4_19" if support < 20 else "20_plus")
        )
        band = next(
            index
            for index, upper in enumerate(PROBABILITY_BANDS)
            if predictions[key].probability < upper or upper == 1
        )
        history = min(row.history_a_count, row.history_b_count)
        history_band = max(
            index for index, lower in enumerate(spec.history_boundaries) if history >= lower
        )
        groups = [
            "overall",
            f"deck:{novelty[int(row.deck_a_support > 0) + int(row.deck_b_support > 0)]}",
            f"pair:{'seen_pair' if row.pair_support else 'unseen_pair'}",
            f"support:{support_name}",
            f"player:{novelty[int(row.player_a_support > 0) + int(row.player_b_support > 0)]}",
            f"form:{'present' if row.forms else 'absent'}",
            f"probability:band_{band}",
            f"history:band_{history_band}",
        ]
        groups += [f"day:{day.isoformat()}" for day in spec.days if key[0].date() == day]
        groups += [f"era:{era}" for era in spec.eras if row.era == era]
        groups += [f"form:identity:{form}" for form in spec.forms if form in row.forms]
        groups += [f"tower:{tower}" for tower in spec.towers if tower in (row.tower_a, row.tower_b)]
        for group in groups:
            members[group].append(key)
    return FixedSlices(
        tuple(sorted(predictions)),
        tuple((name, tuple(keys)) for name, keys in sorted(members.items())),
    )


@dataclass(frozen=True)
class SliceReport:
    name: str
    row_count: int
    comparison: PairedComparison | None


def compare_slices(
    candidate: Sequence[Prediction], comparator: Sequence[Prediction], slices: FixedSlices
) -> tuple[SliceReport, ...]:
    aligned = pair_predictions(candidate, comparator)
    if tuple(row.row_key for row, _ in aligned) != slices.population:
        raise ValueError("slice comparison must use the frozen comparator population")
    first, second = index_predictions(candidate), index_predictions(comparator)
    return tuple(
        SliceReport(
            name,
            len(keys),
            paired_comparison(tuple(first[key] for key in keys), tuple(second[key] for key in keys))
            if keys
            else None,
        )
        for name, keys in slices.members
    )
