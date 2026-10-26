"""Warm inference reuses only a verified artifact with an unchanged manifest."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from clash_sos.application.live_model import LiveModelError, score_live_decks
from clash_sos.infrastructure.ml import attention_artifact_io


def write_manifest(path: Path, version: str = "first") -> None:
    path.mkdir(exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps({"manifest_type": "matchup_attention", "model_version": version})
    )


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []

    class Schema:
        identity_vocab = ("knight:base",)
        tower_catalog = None

        def encode_side(self, cards: tuple[str, ...], *, tower: str | None) -> tuple[int, ...]:
            return (0,) * len(cards)

    class Model:
        def __call__(self, rows: torch.Tensor) -> torch.Tensor:
            return torch.zeros(rows.shape[0])

    def load(path: Path) -> tuple[Any, Any, Any]:
        calls.append(path)
        sleep(0.01)
        raw = json.loads((path / "manifest.json").read_bytes())
        return (
            SimpleNamespace(
                model_version=raw["model_version"],
                dataset_version="data",
                catalog_version="catalog",
                balance_era_id="2026-06",
            ),
            Schema(),
            Model(),
        )

    monkeypatch.setattr(attention_artifact_io, "load_attention_artifact", load)
    return calls


def test_repeated_predictions_reuse_the_verified_model(tmp_path: Path, loader: list[Path]) -> None:
    write_manifest(tmp_path)
    deck = ("knight:base",) * 8

    first = score_live_decks(tmp_path, [(deck, deck)])
    second = score_live_decks(tmp_path / ".", [(deck, deck)])

    assert first == second
    assert second[1][0].side_a_win_probability == 0.5
    assert loader == [tmp_path]


def test_manifest_change_reloads_the_model(tmp_path: Path, loader: list[Path]) -> None:
    write_manifest(tmp_path)
    first, _ = score_live_decks(tmp_path, [])
    write_manifest(tmp_path, "second")

    second, _ = score_live_decks(tmp_path, [])

    assert first.model_version == "first"
    assert second.model_version == "second"
    assert loader == [tmp_path, tmp_path]


def test_cache_keeps_only_one_artifact(tmp_path: Path, loader: list[Path]) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    write_manifest(first)
    write_manifest(second)

    for path in (first, second, first):
        score_live_decks(path, [])

    assert loader == [first, second, first]


def test_failed_load_can_be_retried(
    tmp_path: Path, loader: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_manifest(tmp_path)
    original = attention_artifact_io.load_attention_artifact

    def unavailable(_path: Path) -> Any:
        raise ValueError("incomplete upload")

    monkeypatch.setattr(attention_artifact_io, "load_attention_artifact", unavailable)
    with pytest.raises(LiveModelError, match="model_unavailable"):
        score_live_decks(tmp_path, [])
    monkeypatch.setattr(attention_artifact_io, "load_attention_artifact", original)

    assert score_live_decks(tmp_path, [])[0].model_version == "first"
    assert loader == [tmp_path]


def test_concurrent_first_requests_share_one_load(tmp_path: Path, loader: list[Path]) -> None:
    write_manifest(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(score_live_decks, tmp_path, []) for _ in range(4)]
        results = [future.result() for future in futures]

    assert len(results) == 4
    assert loader == [tmp_path]


def test_missing_manifest_is_rejected_after_a_successful_load(
    tmp_path: Path, loader: list[Path]
) -> None:
    write_manifest(tmp_path)
    score_live_decks(tmp_path, [])
    (tmp_path / "manifest.json").unlink()

    with pytest.raises(LiveModelError, match="model_unavailable"):
        score_live_decks(tmp_path, [])
    assert loader == [tmp_path]
