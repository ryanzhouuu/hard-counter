import sqlite3
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import Barrier

import pytest
from experiments.common.contracts import FileRecord, PopulationIdentity
from experiments.common.reporting_consumption import (
    finish_consumption,
    registry_path,
    reserve_events,
    reserve_reporting,
)
from experiments.tests.common.test_candidate_freeze import candidate, prospective


def population(name: str) -> PopulationIdentity:
    original = prospective(candidate())
    return PopulationIdentity.model_validate(
        {
            **original.model_dump(),
            "row_keys_sha256": sha256(name.encode()).hexdigest(),
        }
    )


def test_consumption_survives_candidate_source_path_and_status_changes(tmp_path: Path) -> None:
    frozen, future = candidate(), population("first")
    reserve_reporting(tmp_path, frozen, future)
    reserve_events(tmp_path, future, ("event-1", "event-2"))
    finish_consumption(tmp_path, future, "failed", tmp_path / "failure.json")
    moved = future.model_copy(
        update={"snapshot_files": (FileRecord(path="copy.jsonl", sha256="c" * 64, size_bytes=3),)}
    )
    with pytest.raises(FileExistsError, match="consumed"):
        reserve_reporting(tmp_path, frozen.model_copy(update={"code_sha256": "d" * 64}), moved)
    with sqlite3.connect(registry_path(tmp_path)) as connection:
        assert connection.execute("SELECT status FROM populations").fetchall() == [("failed",)]
        assert connection.execute("SELECT count(*) FROM events").fetchone() == (2,)


def test_overlapping_events_roll_back_without_blocking_new_disjoint_events(tmp_path: Path) -> None:
    frozen = candidate()
    first, overlapping, disjoint = (population(name) for name in ("first", "overlap", "disjoint"))
    reserve_reporting(tmp_path, frozen, first)
    reserve_events(tmp_path, first, ("shared",))
    reserve_reporting(tmp_path, frozen, overlapping)
    with pytest.raises(FileExistsError, match="consumed"):
        reserve_events(tmp_path, overlapping, ("new", "shared"))
    reserve_reporting(tmp_path, frozen, disjoint)
    reserve_events(tmp_path, disjoint, ("new",))
    finish_consumption(tmp_path, disjoint, "complete", tmp_path / "confirmation.json")
    with sqlite3.connect(registry_path(tmp_path)) as connection:
        assert connection.execute("SELECT count(*) FROM events").fetchone() == (2,)
    with pytest.raises(ValueError, match="unfinished"):
        finish_consumption(tmp_path, disjoint, "failed", tmp_path / "failure.json")


def test_concurrent_confirmations_cannot_reserve_the_same_population(tmp_path: Path) -> None:
    frozen, future = candidate(), population("concurrent")
    ready = Barrier(2)

    def reserve() -> str:
        ready.wait(timeout=5)
        try:
            reserve_reporting(tmp_path, frozen, future)
        except FileExistsError:
            return "consumed"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = tuple(workers.submit(reserve) for _ in range(2))
        assert sorted(result.result(timeout=10) for result in results) == ["consumed", "reserved"]
