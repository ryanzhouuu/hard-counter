"""A configured attention artifact scores a live batch without changing provenance."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from clash_sos.application.live_model import LiveModelError, score_live_decks
from clash_sos.infrastructure.ml import attention_artifact_io


def test_attention_batch_loads_once_and_retains_training_era(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "manifest.json").write_text('{"manifest_type":"matchup_attention"}')
    seen: list[torch.Tensor] = []

    class Schema:
        def encode_deck(self, cards: tuple[str, ...]) -> tuple[int, ...]:
            return tuple(len(card) for card in cards)

    class Model:
        def __call__(self, rows: torch.Tensor) -> torch.Tensor:
            seen.append(rows)
            return torch.tensor([0.0, 1.0])

    manifest = SimpleNamespace(
        model_version="attention-test",
        dataset_version="dataset-test",
        catalog_version="catalog-test",
        balance_era_id="2026-06",
    )
    loads: list[Path] = []

    def load(path: Path) -> tuple[Any, Any, Any]:
        loads.append(path)
        return manifest, Schema(), Model()

    monkeypatch.setattr(attention_artifact_io, "load_attention_artifact", load)

    info, predictions = score_live_decks(
        tmp_path,
        [(("one",) * 8, ("two",) * 8), (("three",) * 8, ("four",) * 8)],
    )

    assert loads == [tmp_path]
    assert len(seen) == 1
    assert seen[0].shape == (2, 2, 8)
    assert info.training_era_id == "2026-06"
    assert predictions[0].side_a_win_probability == 0.5
    assert predictions[1].side_a_win_probability == pytest.approx(0.7310586)
    assert all(item.provenance == info.provenance for item in predictions)


def test_missing_artifact_fails_with_model_unavailable(tmp_path: Path) -> None:
    with pytest.raises(LiveModelError, match="model_unavailable"):
        score_live_decks(tmp_path, [])
