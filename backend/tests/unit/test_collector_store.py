"""Persistent deduplication, conflict exclusion, and restart-safe poll state."""

from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_collector_normalize import battle

from clash_sos.infrastructure.clash_royale.collector_normalize import (
    CollectedBattle,
    normalize_battle,
)
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore

NOW = datetime(2026, 9, 27, 12, tzinfo=UTC)


def record(store: CollectorStore, tag: str, rows: list[CollectedBattle]) -> None:
    """Commit one complete mock log without waiting for a scheduler."""
    store.record_poll(
        tag,
        rows,
        Counter(),
        observed_at=NOW,
        next_due=NOW + timedelta(hours=1),
        log_length=len(rows),
    )


def test_repeated_polls_and_opposite_player_share_one_match(tmp_path: Path) -> None:
    path = tmp_path / "collector.sqlite"
    item = normalize_battle(battle(), "#ABC")
    store = CollectorStore(path)
    try:
        record(store, "#ABC", [item])
        record(store, "#ABC", [item])
    finally:
        store.close()
    reopened = CollectorStore(path)
    try:
        record(reopened, "#DEF", [normalize_battle(battle(), "#DEF")])
        assert reopened.summary()["matches_eligible"] == 1
        assert reopened.summary()["variants"] == 1
        assert reopened.summary()["duplicates"] == 2
        assert reopened.summary()["polls"] == 3
        assert [key for key, _ in reopened.export_battles(NOW, NOW + timedelta(days=1))] == [
            item.event_key
        ]
    finally:
        reopened.close()


def test_conflicts_are_retained_and_removed_from_future_exports(tmp_path: Path) -> None:
    store = CollectorStore(tmp_path / "collector.sqlite")
    raw = battle()
    original = normalize_battle(raw, "#ABC")
    changed = deepcopy(raw)
    changed["opponent"][0]["supportCards"] = [
        {"id": 159000001, "name": "Cannoneer", "level": 11, "maxLevel": 11}
    ]
    variant = normalize_battle(changed, "#DEF")
    try:
        record(store, "#ABC", [original])
        assert len(list(store.export_battles(NOW, NOW + timedelta(days=1)))) == 1
        record(store, "#DEF", [variant])
        assert store.summary()["matches_conflicted"] == 1
        assert store.summary()["variants"] == 2
        assert store.summary()["new_conflicts"] == 1
        assert list(store.export_battles(NOW, NOW + timedelta(days=1))) == []
        record(store, "#ABC", [original])
        assert store.summary()["matches_conflicted"] == 1
    finally:
        store.close()


def test_ineligible_variant_cannot_enter_export(tmp_path: Path) -> None:
    store = CollectorStore(tmp_path / "collector.sqlite")
    raw = battle()
    raw["team"][0]["cards"][0]["level"] -= 1
    try:
        record(store, "#ABC", [normalize_battle(raw, "#ABC")])
        assert store.summary()["matches_ineligible"] == 1
        assert store.summary()["ineligible_non_max_level"] == 1
        assert list(store.export_battles(NOW, NOW + timedelta(days=1))) == []
    finally:
        store.close()


def test_poll_rolls_back_observations_with_failed_checkpoint(tmp_path: Path) -> None:
    store = CollectorStore(tmp_path / "collector.sqlite")
    first = normalize_battle(battle(), "#ABC")
    try:
        with pytest.raises(ValueError, match="timezone-aware"):
            store.record_poll(
                "#ABC",
                [first],
                Counter(),
                observed_at=NOW,
                next_due=datetime(2026, 9, 27, 13),
                log_length=1,
            )
        assert store.summary()["variants"] == 0
        assert store.due_tags(("#ABC",), now=NOW) == ("#ABC",)
    finally:
        store.close()


def test_resume_refetches_due_tags_and_flags_nonoverlap(tmp_path: Path) -> None:
    store = CollectorStore(tmp_path / "collector.sqlite")
    raw = battle()
    first = normalize_battle(raw, "#ABC")
    later = deepcopy(raw)
    later["battleTime"] = "20260927T130000.000Z"
    second = normalize_battle(later, "#ABC")
    try:
        assert store.due_tags(("#ABC",), now=NOW) == ("#ABC",)
        record(store, "#ABC", [first])
        assert store.due_tags(("#ABC",), now=NOW) == ()
        result = store.record_poll(
            "#ABC",
            [second],
            Counter({"unknown_card": 2}),
            observed_at=NOW + timedelta(hours=1),
            next_due=NOW + timedelta(hours=2),
            log_length=3,
        )
        assert result.possible_gap
        assert store.summary()["possible_gaps"] == 1
        assert store.summary()["rejected_unknown_card"] == 2
        store.record_failure("#ABC", "rate_limited", next_due=NOW + timedelta(hours=3))
        assert store.due_tags(("#ABC",), now=NOW + timedelta(hours=2)) == ()
        assert store.summary()["api_rate_limited"] == 1
    finally:
        store.close()
