"""Local collector command contracts and cohort file validation."""

from collections import Counter
from datetime import UTC, datetime, timedelta
from json import loads
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr
from test_collector_normalize import battle
from typer.testing import CliRunner

from clash_sos.infrastructure.clash_royale.client import RoyaleClient
from clash_sos.infrastructure.clash_royale.collector_normalize import normalize_battle
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore
from clash_sos.interfaces.cli import collector as collector_cli
from clash_sos.interfaces.cli.collector import load_cohort, sweep_report
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
    path.write_text("".join(f"#T{index:03d}\n" for index in range(800)))
    assert len(load_cohort(path)) == 800
    path.write_text("".join(f"#T{index:03d}\n" for index in range(801)))
    with pytest.raises(ValueError, match="1-800"):
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


def test_cli_sweep_polls_once_then_skips_recent_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = tmp_path / "players.txt"
    cohort.write_text("#ABC\n")
    database = tmp_path / "collector.sqlite"
    requests = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=[battle()])

    async_client = httpx.AsyncClient

    def mock_async_client(*, base_url: str, timeout: float) -> httpx.AsyncClient:
        assert base_url == "https://proxy.royaleapi.dev/v1/"
        return async_client(
            transport=httpx.MockTransport(respond), base_url=base_url, timeout=timeout
        )

    monkeypatch.setattr(httpx, "AsyncClient", mock_async_client)
    monkeypatch.setattr(
        collector_cli,
        "get_settings",
        lambda: SimpleNamespace(
            royale_api_token=SecretStr("test-token"),
            royale_api_base_url="https://proxy.royaleapi.dev/v1/",
        ),
    )
    args = ["collect", "sweep", "--cohort", str(cohort), "--database", str(database)]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert requests == 1
    assert loads(first.stdout)["successful_polls"] == 1
    assert loads(first.stdout)["eligible_delta"] == 1

    second = CliRunner().invoke(app, args)
    assert second.exit_code == 0, second.output
    assert requests == 1
    assert loads(second.stdout)["skipped_not_due"] == 1
    assert loads(second.stdout)["attempted"] == 0


def test_cli_run_uses_configured_api_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = tmp_path / "players.txt"
    cohort.write_text("#ABC\n")
    seen: list[str] = []

    async def collect(client: RoyaleClient, *_args: object, **_kwargs: object) -> dict[str, int]:
        seen.append(str(client.client.base_url))
        return {}

    monkeypatch.setattr(collector_cli, "collect_for_duration", collect)
    monkeypatch.setattr(
        collector_cli,
        "get_settings",
        lambda: SimpleNamespace(
            royale_api_token=SecretStr("test-token"),
            royale_api_base_url="https://proxy.royaleapi.dev/v1/",
        ),
    )
    collector_cli.run_collector(
        cohort=cohort,
        duration_minutes=1,
        poll_interval_minutes=1,
        request_spacing_seconds=1,
        failure_backoff_seconds=1,
        rate_limit_backoff_seconds=1,
        database=tmp_path / "collector.sqlite",
    )
    assert seen == ["https://proxy.royaleapi.dev/v1/"]


def test_sweep_report_uses_invocation_deltas() -> None:
    before = {"polls": 10, "matches_eligible": 20, "duplicates": 4}
    after = {
        "polls": 11,
        "api_rate_limited": 1,
        "matches_eligible": 21,
        "matches_ineligible": 2,
        "duplicates": 6,
        "rejected_unknown_card": 3,
        "possible_gaps": 1,
    }
    assert sweep_report(before, after, targeted=3, skipped_not_due=1) == {
        "attempted": 2,
        "successful_polls": 1,
        "api_failures": 1,
        "unfinished": 1,
        "skipped_not_due": 1,
        "new_matches": 3,
        "eligible_delta": 1,
        "duplicates": 2,
        "rejected": 3,
        "possible_gaps": 1,
    }
