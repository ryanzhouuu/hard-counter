"""Local commands for durable official-battle collection and snapshot export."""

import asyncio
from datetime import UTC, datetime
from json import dumps
from pathlib import Path
from typing import Annotated

import httpx
import typer

from clash_sos.application.collector_export import export_collected_matches
from clash_sos.application.collector_run import (
    MAX_COHORT_TAGS,
    CollectorConfig,
    collect_for_duration,
)
from clash_sos.infrastructure.clash_royale.client import RoyaleAPIError, RoyaleClient
from clash_sos.infrastructure.clash_royale.collector_normalize import TAG_PATTERN
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore
from clash_sos.infrastructure.settings import get_settings
from clash_sos.interfaces.cli.dataset_attention import parse_snapshot_datetime

app = typer.Typer(no_args_is_help=True)
DEFAULT_DATABASE = Path("data/collector/official.sqlite")
DEFAULT_COHORT = Path("data/collector/players.txt")


def load_cohort(path: Path) -> tuple[str, ...]:
    """Require one stable, distinct official player tag per nonblank line."""
    tags = tuple(line.strip().upper() for line in path.read_text().splitlines() if line.strip())
    if not 1 <= len(tags) <= MAX_COHORT_TAGS or len(set(tags)) != len(tags):
        raise ValueError(f"cohort must contain 1-{MAX_COHORT_TAGS} distinct tags")
    if any(not TAG_PATTERN.fullmatch(tag) for tag in tags):
        raise ValueError("cohort lines must contain only official player tags like #ABC")
    return tags


def sweep_report(
    before: dict[str, int], after: dict[str, int], targeted: int, skipped_not_due: int
) -> dict[str, int]:
    """Derive invocation counts from cumulative SQLite metrics."""

    def delta(key: str) -> int:
        """Treat metrics absent before a sweep as zero."""
        return after.get(key, 0) - before.get(key, 0)

    successful_polls = delta("polls")
    api_failures = sum(delta(key) for key in after if key.startswith("api_"))
    return {
        "attempted": successful_polls + api_failures,
        "successful_polls": successful_polls,
        "api_failures": api_failures,
        "unfinished": max(0, targeted - successful_polls - api_failures),
        "skipped_not_due": skipped_not_due,
        "new_matches": sum(
            delta(f"matches_{status}") for status in ("eligible", "ineligible", "conflicted")
        ),
        "eligible_delta": delta("matches_eligible"),
        "duplicates": delta("duplicates"),
        "rejected": sum(delta(key) for key in after if key.startswith("rejected_")),
        "possible_gaps": delta("possible_gaps"),
    }


@app.command("sweep")
def sweep_collector(
    cohort: Annotated[Path, typer.Option(help="One player tag per line")] = DEFAULT_COHORT,
    database: Path = DEFAULT_DATABASE,
    max_duration_minutes: float = 30,
    min_gap_minutes: float = 60,
    request_spacing_seconds: float = 2,
    failure_backoff_seconds: float = 60,
    rate_limit_backoff_seconds: float = 300,
) -> None:
    """Attempt each currently due tag once, then exit with per-sweep counts."""
    try:
        tags = load_cohort(cohort)
        config = CollectorConfig(
            duration_seconds=max_duration_minutes * 60,
            poll_interval_seconds=min_gap_minutes * 60,
            request_spacing_seconds=request_spacing_seconds,
            failure_backoff_seconds=failure_backoff_seconds,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        )
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    store = CollectorStore(database)
    try:
        due = store.due_tags(tags, now=datetime.now(UTC))
        before = store.summary()
        if due:
            token = get_settings().royale_api_token
            if token is None:
                raise typer.BadParameter("CLASH_ROYALE_API_TOKEN is required")

            async def sweep() -> dict[str, int]:
                """Share the bounded polling path while stopping after one pass."""
                async with httpx.AsyncClient(
                    base_url="https://api.clashroyale.com/v1/", timeout=12.0
                ) as http:
                    return await collect_for_duration(
                        RoyaleClient(http, token.get_secret_value()),
                        store,
                        due,
                        config,
                        progress=lambda message: typer.echo(message, err=True),
                        single_pass=True,
                    )

            try:
                after = asyncio.run(sweep())
            except RoyaleAPIError as error:
                typer.echo(f"collector stopped: {error.code}", err=True)
                raise typer.Exit(code=1) from error
        else:
            after = before
        report = sweep_report(before, after, len(due), len(tags) - len(due))
        typer.echo(dumps(report, sort_keys=True))
    finally:
        store.close()


@app.command("run")
def run_collector(
    cohort: Annotated[Path, typer.Option(help="One player tag per line")],
    duration_minutes: Annotated[float, typer.Option()],
    poll_interval_minutes: Annotated[float, typer.Option()],
    request_spacing_seconds: Annotated[float, typer.Option()],
    failure_backoff_seconds: Annotated[float, typer.Option()],
    rate_limit_backoff_seconds: Annotated[float, typer.Option()],
    database: Path = DEFAULT_DATABASE,
) -> None:
    """Poll one fixed cohort with one request in flight and resumable checkpoints."""
    try:
        tags = load_cohort(cohort)
        config = CollectorConfig(
            duration_seconds=duration_minutes * 60,
            poll_interval_seconds=poll_interval_minutes * 60,
            request_spacing_seconds=request_spacing_seconds,
            failure_backoff_seconds=failure_backoff_seconds,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        )
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    token = get_settings().royale_api_token
    if token is None:
        raise typer.BadParameter("CLASH_ROYALE_API_TOKEN is required")
    store = CollectorStore(database)

    async def campaign() -> dict[str, int]:
        async with httpx.AsyncClient(
            base_url="https://api.clashroyale.com/v1/", timeout=12.0
        ) as http:
            return await collect_for_duration(
                RoyaleClient(http, token.get_secret_value()),
                store,
                tags,
                config,
                progress=lambda message: typer.echo(message, err=True),
            )

    try:
        typer.echo(dumps(asyncio.run(campaign()), sort_keys=True))
    except RoyaleAPIError as error:
        typer.echo(f"collector stopped: {error.code}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        store.close()


@app.command("export")
def export_collector(
    destination: Annotated[Path, typer.Option(help="New temporary JSONL path")],
    dataset_version: Annotated[str, typer.Option()],
    balance_era: Annotated[str, typer.Option()],
    start: Annotated[datetime, typer.Option(parser=parse_snapshot_datetime)],
    end: Annotated[datetime, typer.Option(parser=parse_snapshot_datetime)],
    database: Path = DEFAULT_DATABASE,
) -> None:
    """Freeze eligible current-mode matches in a deterministic JSONL export."""
    if not database.is_file():
        raise typer.BadParameter("collector database does not exist")
    store = CollectorStore(database)
    try:
        result = export_collected_matches(
            store,
            destination,
            dataset_version=dataset_version,
            balance_era_id=balance_era,
            start=start,
            end=end,
        )
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    finally:
        store.close()
    typer.echo(dumps({"written": result.written, "skipped_mode": result.skipped_mode}))


@app.command("report")
def report_collector(database: Path = DEFAULT_DATABASE) -> None:
    """Print retained status and observation counts for one local database."""
    if not database.is_file():
        raise typer.BadParameter("collector database does not exist")
    store = CollectorStore(database)
    try:
        typer.echo(dumps(store.summary(), sort_keys=True))
    finally:
        store.close()
