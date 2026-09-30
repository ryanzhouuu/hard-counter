from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from experiments.common.data_access import ResearchRow
from experiments.player_adjustment.history import FrozenHistory, training_history

START = datetime(2026, 1, 1, tzinfo=UTC)
DECKS = ((0, 2, 3, 4, 5, 6, 7, 8, 10), (1, 2, 3, 4, 5, 6, 7, 8, 10))


def row(index: int, *, a: str = "a", b: str = "b", label: int = 1) -> ResearchRow:
    return ResearchRow(
        (START + timedelta(seconds=index), "a", "b", index),
        f"event-{index}",
        a,
        b,
        label,
        DECKS,
    )


def test_prior_history_excludes_current_future_and_tied_outcomes() -> None:
    rows = [row(0), row(1), replace(row(2), key=row(1).key), row(3)]
    first, _ = training_history(rows)
    changed, _ = training_history([*rows[:1], replace(rows[1], label=0), *rows[2:]])
    np.testing.assert_array_equal(first[:3], changed[:3])
    assert first[0] == 0
    assert first[1] == first[2] > 0
    assert first[3] != changed[3]
    reverse, state = training_history([rows[2], rows[0], rows[3], rows[1]])
    np.testing.assert_array_equal(first, reverse[[1, 3, 0, 2]])
    assert {entry.count for entry in state.players} == {4}


def test_frozen_cutoff_does_not_include_later_players_or_labels() -> None:
    rows = [row(0), row(1), row(2, a="new", b="future")]
    gaps, frozen = training_history(rows, cutoff=rows[0].key[0])
    assert {entry.player for entry in frozen.players} == {"a", "b"}
    assert gaps[1] > 0 and gaps[2] == 0
    changed, same = training_history(
        [rows[0], replace(rows[1], label=0), replace(rows[2], label=0)],
        cutoff=rows[0].key[0],
    )
    np.testing.assert_array_equal(gaps, changed)
    assert same == frozen
    restored = FrozenHistory.model_validate_json(frozen.model_dump_json())
    np.testing.assert_array_equal(frozen.gaps(rows), restored.gaps(rows))
    np.testing.assert_array_equal(restored.gaps([rows[0].swapped()]), -restored.gaps([rows[0]]))
    _, later = training_history(rows, cutoff=rows[1].key[0])
    assert frozen != later


def test_history_rejects_duplicate_events_and_corrupt_serialized_rating() -> None:
    with pytest.raises(ValueError, match="unique events"):
        training_history([row(0), row(0)])
    _, frozen = training_history([row(0)])
    corrupt = frozen.model_dump()
    corrupt["players"][0]["rating"] = 99
    with pytest.raises(ValueError, match="disagree"):
        FrozenHistory.model_validate(corrupt)
