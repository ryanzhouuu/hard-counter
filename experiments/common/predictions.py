"""Oriented prediction identities and exact paired-population joins."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, TypeAdapter

from clash_sos.domain.attention_protocol import RowKey


@dataclass(frozen=True)
class Prediction:
    row_key: RowKey
    event_key: str
    timestamp: datetime
    player_a: str
    player_b: str
    label: int
    logit: float
    probability: float
    raw_probability: float | None = None

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp != self.row_key[0]:
            raise ValueError("prediction timestamp must be aware and match the canonical row key")
        if not all(
            (self.event_key, self.row_key[1], self.row_key[2], self.player_a, self.player_b)
        ):
            raise ValueError("prediction identities must be nonempty")
        if self.row_key[3] < 0 or self.player_a == self.player_b:
            raise ValueError("prediction requires a nonnegative row number and distinct players")
        if self.label not in (0, 1) or not isfinite(self.logit):
            raise ValueError("prediction requires a binary label and finite logit")
        for probability in (self.probability, self.raw_probability):
            if probability is not None and (not isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError("prediction probabilities must be finite and in [0, 1]")


def index_predictions(rows: Iterable[Prediction]) -> dict[RowKey, Prediction]:
    indexed: dict[RowKey, Prediction] = {}
    events: set[str] = set()
    for row in rows:
        if row.row_key in indexed or row.event_key in events:
            raise ValueError("duplicate prediction row or event identity")
        indexed[row.row_key] = row
        events.add(row.event_key)
    return indexed


def pair_predictions(
    candidate: Iterable[Prediction], comparator: Iterable[Prediction]
) -> tuple[tuple[Prediction, Prediction], ...]:
    first, second = index_predictions(candidate), index_predictions(comparator)
    if first.keys() != second.keys():
        raise ValueError("paired predictions require identical canonical rows")
    pairs: list[tuple[Prediction, Prediction]] = []
    for key in sorted(first):
        a, b = first[key], second[key]
        if (a.event_key, a.timestamp, a.player_a, a.player_b, a.label) != (
            b.event_key,
            b.timestamp,
            b.player_a,
            b.player_b,
            b.label,
        ):
            raise ValueError("paired prediction event, orientation, or label mismatch")
        pairs.append((a, b))
    return tuple(pairs)


_PREDICTIONS = TypeAdapter(tuple[Prediction, ...])


def write_predictions(path: Path, rows: Iterable[Prediction]) -> None:
    indexed = index_predictions(rows)
    ordered = tuple(indexed[key] for key in sorted(indexed))
    path.write_bytes(_PREDICTIONS.dump_json(ordered) + b"\n")


def read_predictions(path: Path) -> tuple[Prediction, ...]:
    rows = _PREDICTIONS.validate_json(path.read_bytes())
    indexed = index_predictions(rows)
    return tuple(indexed[key] for key in sorted(indexed))
