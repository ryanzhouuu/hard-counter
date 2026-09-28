"""Own durable local match variants and poll checkpoints for one collector process."""

import sqlite3
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from json import dumps, loads
from pathlib import Path
from typing import cast

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.infrastructure.clash_royale.collector_normalize import CollectedBattle


@dataclass(frozen=True)
class PollResult:
    """Counts for one committed poll, including a possible log-overlap gap."""

    new_matches: int
    duplicates: int
    new_conflicts: int
    possible_gap: bool


def _micros(value: datetime) -> int:
    """Use integral UTC microseconds so SQL windows preserve fractional order."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("poll and export times must be timezone-aware")
    delta = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


class CollectorStore:
    """Serialize one poll's observations and checkpoint in a single transaction."""

    def __init__(self, path: Path) -> None:
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if existed and version != 1:
            self.connection.close()
            raise ValueError("unsupported collector database schema version")
        self.connection.execute("PRAGMA journal_mode=WAL")
        if not existed:
            self.connection.executescript(
                """
                CREATE TABLE matches (
                    event_key TEXT PRIMARY KEY,
                    timestamp_us INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('eligible','ineligible','conflicted'))
                );
                CREATE TABLE variants (
                    event_key TEXT NOT NULL REFERENCES matches(event_key),
                    variant_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    reason TEXT,
                    PRIMARY KEY(event_key, variant_hash)
                );
                CREATE TABLE poll_state (
                    tag TEXT PRIMARY KEY,
                    last_success_us INTEGER,
                    next_due_us INTEGER NOT NULL,
                    log_keys TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE metrics (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
                CREATE INDEX matches_export ON matches(status, timestamp_us, event_key);
                PRAGMA user_version=1;
                """
            )

    def close(self) -> None:
        """Close the local connection before copying or moving the database."""
        self.connection.close()

    def _increment(self, name: str, value: int) -> None:
        if value:
            self.connection.execute(
                "INSERT INTO metrics(name,value) VALUES(?,?) "
                "ON CONFLICT(name) DO UPDATE SET value=value+excluded.value",
                (name, value),
            )

    def _record_variant(self, battle: CollectedBattle) -> tuple[int, int, int]:
        """Return new-match, duplicate, and newly conflicted indicators."""
        key = battle.event_key
        existing = self.connection.execute(
            "SELECT status,timestamp_us FROM matches WHERE event_key=?", (key,)
        ).fetchone()
        timestamp_us = _micros(battle.timestamp)
        if existing is not None and existing["timestamp_us"] != timestamp_us:
            raise ValueError("stored match identity has a different timestamp")
        if existing is None:
            self.connection.execute(
                "INSERT INTO matches(event_key,timestamp_us,status) VALUES(?,?,?)",
                (key, timestamp_us, "ineligible"),
            )
        reason = battle.exclusion_reason
        inserted = self.connection.execute(
            "INSERT OR IGNORE INTO variants(event_key,variant_hash,payload,reason) VALUES(?,?,?,?)",
            (
                key,
                battle.variant_hash,
                canonical_json_bytes(battle.model_dump(mode="python")).decode(),
                reason,
            ),
        ).rowcount
        if not inserted:
            return (0, 1, 0)
        count = self.connection.execute(
            "SELECT COUNT(*) FROM variants WHERE event_key=?", (key,)
        ).fetchone()[0]
        status = "conflicted" if count > 1 else "ineligible" if reason else "eligible"
        self.connection.execute("UPDATE matches SET status=? WHERE event_key=?", (status, key))
        return (int(existing is None), 0, int(count == 2))

    def record_poll(
        self,
        tag: str,
        battles: Sequence[CollectedBattle],
        rejections: Counter[str],
        *,
        observed_at: datetime,
        next_due: datetime,
        log_length: int,
    ) -> PollResult:
        """Commit a complete fetched log with its next poll and overlap state."""
        keys = sorted({battle.event_key for battle in battles})
        previous = self.connection.execute(
            "SELECT log_keys FROM poll_state WHERE tag=?", (tag,)
        ).fetchone()
        prior_keys: set[str] = (
            set(cast(list[str], loads(previous["log_keys"]))) if previous is not None else set()
        )
        gap = bool(prior_keys) and bool(keys) and prior_keys.isdisjoint(keys)
        exclusions = Counter(
            battle.exclusion_reason for battle in battles if battle.exclusion_reason is not None
        )
        new_matches = duplicates = conflicts = 0
        with self.connection:
            for battle in battles:
                new, duplicate, conflict = self._record_variant(battle)
                new_matches += new
                duplicates += duplicate
                conflicts += conflict
            self.connection.execute(
                "INSERT INTO poll_state(tag,last_success_us,next_due_us,log_keys) "
                "VALUES(?,?,?,?) ON CONFLICT(tag) DO UPDATE SET "
                "last_success_us=excluded.last_success_us, "
                "next_due_us=excluded.next_due_us, log_keys=excluded.log_keys",
                (tag, _micros(observed_at), _micros(next_due), dumps(keys)),
            )
            for name, value in (
                ("polls", 1),
                ("seen", log_length),
                ("duplicates", duplicates),
                ("possible_gaps", int(gap)),
                ("new_conflicts", conflicts),
                *((f"ineligible_{reason}", count) for reason, count in exclusions.items()),
                *((f"rejected_{reason}", count) for reason, count in rejections.items()),
            ):
                self._increment(name, value)
        return PollResult(new_matches, duplicates, conflicts, gap)

    def record_failure(self, tag: str, code: str, *, next_due: datetime) -> None:
        """Reschedule a failed fetch without changing its last successful log."""
        with self.connection:
            self.connection.execute(
                "INSERT INTO poll_state(tag,next_due_us) VALUES(?,?) "
                "ON CONFLICT(tag) DO UPDATE SET next_due_us=excluded.next_due_us",
                (tag, _micros(next_due)),
            )
            self._increment(f"api_{code}", 1)

    def due_tags(self, cohort: Sequence[str], *, now: datetime) -> tuple[str, ...]:
        """Return only cohort tags whose stored retry or next poll is due."""
        threshold = _micros(now)
        return tuple(
            tag
            for tag in cohort
            if (
                row := self.connection.execute(
                    "SELECT next_due_us FROM poll_state WHERE tag=?", (tag,)
                ).fetchone()
            )
            is None
            or row["next_due_us"] <= threshold
        )

    def summary(self) -> dict[str, int]:
        """Report retained match states and operational counts without raw data."""
        result = {
            f"matches_{status}": count
            for status, count in self.connection.execute(
                "SELECT status,COUNT(*) FROM matches GROUP BY status"
            )
        }
        result.update(self.connection.execute("SELECT name,value FROM metrics").fetchall())
        result["variants"] = self.connection.execute("SELECT COUNT(*) FROM variants").fetchone()[0]
        return result

    def export_battles(
        self, start: datetime, end: datetime
    ) -> Iterator[tuple[str, CollectedBattle]]:
        """Stream conflict-free matches from one consistent database snapshot."""
        if _micros(start) >= _micros(end):
            raise ValueError("export window must increase")
        self.connection.execute("BEGIN")
        try:
            rows = self.connection.execute(
                "SELECT m.event_key,v.payload FROM matches AS m "
                "JOIN variants AS v ON v.event_key=m.event_key "
                "WHERE m.status='eligible' AND m.timestamp_us>=? AND m.timestamp_us<? "
                "ORDER BY m.timestamp_us,m.event_key",
                (_micros(start), _micros(end)),
            )
            for row in rows:
                yield row["event_key"], CollectedBattle.model_validate_json(row["payload"])
        finally:
            self.connection.rollback()
