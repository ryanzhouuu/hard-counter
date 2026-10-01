"""Repository-wide reservations prevent reuse of prospective populations and events."""

import sqlite3
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Literal

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.candidate import CandidateFreeze
from experiments.common.contracts import PopulationIdentity, fingerprint


def registry_path(root: Path) -> Path:
    return root / "data" / "experiments" / "reporting-consumption.sqlite3"


@contextmanager
def _database(root: Path) -> Generator[sqlite3.Connection]:
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=30)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS populations (
                identity TEXT PRIMARY KEY,
                candidate_sha256 TEXT NOT NULL,
                population_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('started', 'complete', 'failed')),
                report TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                event_key TEXT PRIMARY KEY,
                population_identity TEXT NOT NULL REFERENCES populations(identity)
            );
            """
        )
        yield connection


def reserve_reporting(root: Path, frozen: CandidateFreeze, population: PopulationIdentity) -> None:
    """Reserve before loading labels; filenames, orientation, and candidates cannot reset it."""
    with _database(root) as connection:
        try:
            with connection:
                connection.execute(
                    "INSERT INTO populations VALUES (?, ?, ?, 'started', NULL)",
                    (
                        population.row_keys_sha256,
                        fingerprint(frozen),
                        canonical_json_bytes(population.model_dump(mode="json")).decode(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise FileExistsError("reporting population already consumed") from error


def reserve_events(root: Path, population: PopulationIdentity, event_keys: Sequence[str]) -> None:
    """Reject overlapping inventories atomically before computing any reporting predictions."""
    if not event_keys:
        raise ValueError("reporting event reservation requires a nonempty inventory")
    with _database(root) as connection:
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO events VALUES (?, ?)",
                    ((event, population.row_keys_sha256) for event in event_keys),
                )
        except sqlite3.IntegrityError as error:
            raise FileExistsError(
                "reporting events already consumed or population not reserved"
            ) from error


def finish_consumption(
    root: Path,
    population: PopulationIdentity,
    status: Literal["complete", "failed"],
    report: Path,
) -> None:
    """Completion records are informational; failed and interrupted reservations remain consumed."""
    with _database(root) as connection, connection:
        updated = connection.execute(
            "UPDATE populations SET status = ?, report = ? "
            "WHERE identity = ? AND status = 'started'",
            (status, str(report.resolve()), population.row_keys_sha256),
        )
        if updated.rowcount != 1:
            raise ValueError("reporting consumption requires an unfinished reservation")
