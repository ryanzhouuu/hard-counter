"""Local collector command contracts and cohort file validation."""

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_collector_normalize import battle
from typer.testing import CliRunner

from clash_sos.infrastructure.clash_royale.collector_normalize import normalize_battle
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore
from clash_sos.interfaces.cli.collector import load_cohort
from clash_sos.interfaces.cli.main import app


def test_cohort_requires_distinct_tag_lines(tmp_path: Path) -> None:
    path = tmp_path / "players.txt"
    path.write_text("#abc\n\n#DEF\n")
    assert load_cohort(path) == ("#ABC", "#DEF")
    path.write_text("#ABC\n#abc\n")
    with pytest.raises(ValueError, match="distinct"):
        load_cohort(path)
    path.write_text("#ABC extra\n")
    with pytest.raises(ValueError, match="only official player tags"):
        load_cohort(path)


def test_cli_exports_and_reports_existing_database(tmp_path: Path) -> None:
    database = tmp_path / "collector.sqlite"
    store = CollectorStore(database)
    timestamp = datetime(2026, 9, 27, 12, tzinfo=UTC)
    try:
        store.record_poll(
            "#ABC",
            [normalize_battle(battle(), "#ABC")],
            Counter(),
            observed_at=timestamp,
            next_due=timestamp + timedelta(hours=1),
            log_length=1,
        )
    finally:
        store.close()
    runner = CliRunner()
    report = runner.invoke(app, ["collect", "report", "--database", str(database)])
    assert report.exit_code == 0, report.output
    assert '"matches_eligible": 1' in report.output
    destination = tmp_path / "snapshot.jsonl"
    export = runner.invoke(
        app,
        [
            "collect",
            "export",
            "--database",
            str(database),
            "--destination",
            str(destination),
            "--dataset-version",
            "pilot-v1",
            "--balance-era",
            "2026-09",
            "--start",
            "2026-09-27T00:00:00Z",
            "--end",
            "2026-09-28T00:00:00Z",
        ],
    )
    assert export.exit_code == 0, export.output
    assert '"written": 1' in export.output
    assert '"mode":"Ranked1v1_NewArena2"' in destination.read_text()
