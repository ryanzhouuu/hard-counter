"""Bounded polling, retry pacing, and persisted API failure behavior."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from test_collector_normalize import battle

from clash_sos.application.collector_run import CollectorConfig, collect_for_duration
from clash_sos.infrastructure.clash_royale.client import RoyaleAPIError, RoyaleClient
from clash_sos.infrastructure.clash_royale.collector_store import CollectorStore

BASE = datetime(2026, 9, 27, tzinfo=UTC)


class FakeClock:
    """Advance both scheduling clocks without a real campaign delay."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def now(self) -> datetime:
        return BASE + timedelta(seconds=self.elapsed)

    def monotonic(self) -> float:
        return self.elapsed

    async def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


def config(duration: float = 17) -> CollectorConfig:
    return CollectorConfig(duration, 10, 1, 4, 7)


def test_poll_refetches_due_tag_and_flags_missing_overlap(tmp_path: Path) -> None:
    clock = FakeClock()
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        assert request.headers["Authorization"] == "Bearer test-token"
        calls += 1
        raw = battle()
        if calls == 2:
            raw["battleTime"] = "20260927T130000.000Z"
            bad = battle()
            bad["battleTime"] = "20260927T140000.000Z"
            bad["team"][0]["cards"][0]["name"] = "Unknown future card"
            return httpx.Response(200, json=[raw, bad])
        return httpx.Response(200, json=[raw])

    async def campaign(store: CollectorStore) -> dict[str, int]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://api.clashroyale.com/v1/"
        ) as http:
            return await collect_for_duration(
                RoyaleClient(http, "test-token"),
                store,
                ("#ABC",),
                config(),
                now=clock.now,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

    store = CollectorStore(tmp_path / "collector.sqlite")
    try:
        summary = asyncio.run(campaign(store))
        assert calls == 2
        assert summary["polls"] == 2
        assert summary["rejected_unknown_card"] == 1
        assert summary["possible_gaps"] == 1
        assert summary["matches_eligible"] == 2
        assert store.due_tags(("#ABC",), now=clock.now()) == ()
    finally:
        store.close()


def test_rate_limit_delays_all_tags_then_retries(tmp_path: Path) -> None:
    clock = FakeClock()
    calls: list[tuple[str, float]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        tag = request.url.path.split("/")[-2]
        calls.append((tag, clock.elapsed))
        if tag == "#ABC" and len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=[battle()])

    async def campaign(store: CollectorStore) -> dict[str, int]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://api.clashroyale.com/v1/"
        ) as http:
            return await collect_for_duration(
                RoyaleClient(http, "test-token"),
                store,
                ("#ABC", "#DEF"),
                config(11),
                now=clock.now,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

    store = CollectorStore(tmp_path / "collector.sqlite")
    try:
        summary = asyncio.run(campaign(store))
        assert len(calls) == 3
        assert calls[1][1] >= 7
        assert summary["api_rate_limited"] == 1
        assert summary["polls"] == 2
        assert summary["matches_eligible"] == 1
        assert summary["duplicates"] == 1
    finally:
        store.close()


def test_single_pass_stops_after_one_attempt_and_resumes_when_due(tmp_path: Path) -> None:
    clock = FakeClock()
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[battle()])

    async def sweep(store: CollectorStore) -> dict[str, int]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://api.clashroyale.com/v1/"
        ) as http:
            return await collect_for_duration(
                RoyaleClient(http, "test-token"),
                store,
                ("#ABC",),
                config(35),
                now=clock.now,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                single_pass=True,
            )

    store = CollectorStore(tmp_path / "collector.sqlite")
    try:
        assert asyncio.run(sweep(store))["polls"] == 1
        assert calls == 1
        assert clock.elapsed == 0
        assert asyncio.run(sweep(store))["polls"] == 1
        assert calls == 1
        clock.elapsed = 11
        assert asyncio.run(sweep(store))["polls"] == 2
        assert calls == 2
    finally:
        store.close()


def test_single_pass_records_failure_without_retrying_it(tmp_path: Path) -> None:
    clock = FakeClock()
    calls: list[tuple[str, float]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path.split("/")[-2], clock.elapsed))
        return httpx.Response(429 if len(calls) == 1 else 200, json=[battle()])

    async def sweep(store: CollectorStore) -> dict[str, int]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://api.clashroyale.com/v1/"
        ) as http:
            return await collect_for_duration(
                RoyaleClient(http, "test-token"),
                store,
                ("#ABC", "#DEF"),
                config(20),
                now=clock.now,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                single_pass=True,
            )

    store = CollectorStore(tmp_path / "collector.sqlite")
    try:
        summary = asyncio.run(sweep(store))
        assert len(calls) == 2
        assert calls[1][1] >= 7
        assert summary["api_rate_limited"] == 1
        assert summary["polls"] == 1
    finally:
        store.close()


def test_single_pass_keeps_unattempted_tags_due_at_deadline(tmp_path: Path) -> None:
    clock = FakeClock()
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[])

    async def sweep(store: CollectorStore) -> dict[str, int]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://api.clashroyale.com/v1/"
        ) as http:
            return await collect_for_duration(
                RoyaleClient(http, "test-token"),
                store,
                ("#ABC", "#DEF", "#GHI"),
                config(1.5),
                now=clock.now,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                single_pass=True,
            )

    store = CollectorStore(tmp_path / "collector.sqlite")
    try:
        assert asyncio.run(sweep(store))["polls"] == 2
        assert calls == 2
        assert store.due_tags(("#ABC", "#DEF", "#GHI"), now=clock.now()) == ("#GHI",)
    finally:
        store.close()


def test_credential_rejection_stops_without_checkpoint(tmp_path: Path) -> None:
    clock = FakeClock()
    store = CollectorStore(tmp_path / "collector.sqlite")

    async def campaign() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(403)),
            base_url="https://api.clashroyale.com/v1/",
        ) as http:
            await collect_for_duration(
                RoyaleClient(http, "test-token"),
                store,
                ("#ABC",),
                config(),
                now=clock.now,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

    try:
        with pytest.raises(RoyaleAPIError, match="credentials_rejected"):
            asyncio.run(campaign())
        assert store.summary()["api_credentials_rejected"] == 1
        assert store.due_tags(("#ABC",), now=BASE + timedelta(seconds=4)) == ("#ABC",)
        assert "polls" not in store.summary()
    finally:
        store.close()


def test_collector_rejects_more_than_eight_hundred_tags(tmp_path: Path) -> None:
    clock = FakeClock()
    store = CollectorStore(tmp_path / "collector.sqlite")

    async def campaign() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[])),
            base_url="https://api.clashroyale.com/v1/",
        ) as http:
            await collect_for_duration(
                RoyaleClient(http, "test-token"),
                store,
                tuple(f"#T{index:03d}" for index in range(801)),
                config(),
                now=clock.now,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

    try:
        with pytest.raises(ValueError, match="1-800"):
            asyncio.run(campaign())
    finally:
        store.close()
