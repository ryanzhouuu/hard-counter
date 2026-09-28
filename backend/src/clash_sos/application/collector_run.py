"""Run one bounded local collection campaign against the official battle log."""

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from clash_sos.infrastructure.clash_royale.client import RoyaleAPIError, RoyaleClient
from clash_sos.infrastructure.clash_royale.collector_normalize import (
    BattleRejected,
    CollectedBattle,
    normalize_battle,
)
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore


@dataclass(frozen=True)
class CollectorConfig:
    """Bound local runtime and one in-flight request with explicit pacing."""

    duration_seconds: float
    poll_interval_seconds: float
    request_spacing_seconds: float
    failure_backoff_seconds: float
    rate_limit_backoff_seconds: float

    def __post_init__(self) -> None:
        if (
            min(
                self.duration_seconds,
                self.poll_interval_seconds,
                self.request_spacing_seconds,
                self.failure_backoff_seconds,
                self.rate_limit_backoff_seconds,
            )
            <= 0
        ):
            raise ValueError("collector durations and spacing must be positive")


async def collect_for_duration(
    client: RoyaleClient,
    store: CollectorStore,
    cohort: Sequence[str],
    config: CollectorConfig,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Refetch due logs after failures or restarts; checkpoint only complete fetches."""
    if not cohort or len(cohort) > 200 or len(set(cohort)) != len(cohort):
        raise ValueError("collector cohort must contain 1-200 distinct tags")
    deadline = monotonic() + config.duration_seconds
    next_request_at = monotonic()
    while monotonic() < deadline:
        due = store.due_tags(cohort, now=now())
        if not due:
            await sleep(min(5.0, deadline - monotonic()))
            continue
        for tag in due:
            remaining = deadline - monotonic()
            delay = max(0.0, next_request_at - monotonic())
            if remaining <= delay:
                return store.summary()
            if delay:
                await sleep(delay)
            requested_at = monotonic()
            next_request_at = requested_at + config.request_spacing_seconds
            try:
                raw_battles = await client.battles(tag)
            except RoyaleAPIError as error:
                backoff = (
                    config.rate_limit_backoff_seconds
                    if error.code == "rate_limited"
                    else config.failure_backoff_seconds
                )
                observed_at = now()
                store.record_failure(
                    tag, error.code, next_due=observed_at + timedelta(seconds=backoff)
                )
                if progress:
                    progress(f"{tag}: {error.code}; retry after {backoff:g}s")
                if error.code == "credentials_rejected":
                    raise
                if error.code in {"rate_limited", "upstream_unavailable"}:
                    next_request_at = max(next_request_at, monotonic() + backoff)
                continue
            observed_at = now()
            normalized: list[CollectedBattle] = []
            rejections: Counter[str] = Counter()
            for raw in raw_battles:
                try:
                    normalized.append(normalize_battle(raw, tag))
                except BattleRejected as error:
                    rejections[str(error)] += 1
            result = store.record_poll(
                tag,
                normalized,
                rejections,
                observed_at=observed_at,
                next_due=observed_at + timedelta(seconds=config.poll_interval_seconds),
                log_length=len(raw_battles),
            )
            if progress:
                progress(
                    f"{tag}: {result.new_matches} new, {result.duplicates} duplicate, "
                    f"{result.new_conflicts} conflict, {sum(rejections.values())} rejected, "
                    f"possible_gap={result.possible_gap}"
                )
    return store.summary()
