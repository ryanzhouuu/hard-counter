"""Past-only histories with simultaneous processing of tied timestamps."""

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from itertools import groupby
from math import isclose
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from clash_sos.domain.manifests import ManifestModel
from clash_sos.domain.matchup_baseline import card_log_odds
from clash_sos.domain.player_skill import DEFAULT_SKILL_ALPHA, PlayerSkillTracker
from experiments.common.data_access import ResearchRow


class PlayerHistory(ManifestModel):
    player: str = Field(min_length=1)
    wins: int = Field(ge=0)
    count: int = Field(gt=0)
    rating: float = Field(allow_inf_nan=False)


class FrozenHistory(ManifestModel):
    alpha: float = Field(default=DEFAULT_SKILL_ALPHA, gt=0, allow_inf_nan=False)
    cutoff: datetime
    players: tuple[PlayerHistory, ...]

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() is None:
            raise ValueError("history cutoff must have a timezone")
        ids = [entry.player for entry in self.players]
        if ids != sorted(set(ids)):
            raise ValueError("history players must be unique and sorted")
        for entry in self.players:
            expected = card_log_odds(entry.wins, entry.count, alpha=self.alpha)
            if entry.wins > entry.count or not isclose(entry.rating, expected, abs_tol=1e-12):
                raise ValueError("history counts and ratings disagree")
        return self

    def gaps(self, rows: Sequence[ResearchRow]) -> NDArray[np.float64]:
        ratings = {entry.player: entry.rating for entry in self.players}
        return np.asarray(
            [ratings.get(row.player_a, 0.0) - ratings.get(row.player_b, 0.0) for row in rows],
            dtype=np.float64,
        )


def training_history(
    rows: Sequence[ResearchRow], *, cutoff: datetime | None = None
) -> tuple[NDArray[np.float64], FrozenHistory]:
    """Return aligned prior gaps; rows after an optional cutoff cannot update history."""
    if not rows:
        raise ValueError("training history requires rows")
    if len({row.event_key for row in rows}) != len(rows):
        raise ValueError("history requires unique events")
    end = cutoff or max(row.key[0] for row in rows)
    if end.tzinfo is None or end.utcoffset() is None:
        raise ValueError("history cutoff must have a timezone")
    tracker = PlayerSkillTracker(alpha=DEFAULT_SKILL_ALPHA)
    counts: dict[str, int] = defaultdict(int)
    wins: dict[str, int] = defaultdict(int)
    gaps = np.empty(len(rows), dtype=np.float64)
    ordered = sorted(enumerate(rows), key=lambda item: item[1].key[0])
    for _, group in groupby(ordered, key=lambda item: item[1].key[0]):
        tied = list(group)
        for index, row in tied:
            if row.player_a == row.player_b or row.label not in (0, 1):
                raise ValueError("history requires distinct players and binary outcomes")
            gaps[index] = tracker.rating(row.player_a) - tracker.rating(row.player_b)
        for _, row in tied:
            if row.key[0] <= end:
                for player, won in ((row.player_a, row.label == 1), (row.player_b, row.label == 0)):
                    tracker.observe(player, won=won)
                    counts[player] += 1
                    wins[player] += int(won)
    state = FrozenHistory(
        cutoff=end,
        players=tuple(
            PlayerHistory(
                player=player,
                wins=wins[player],
                count=counts[player],
                rating=tracker.rating(player),
            )
            for player in sorted(counts)
        ),
    )
    return gaps, state
