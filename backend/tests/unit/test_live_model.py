"""A configured attention artifact scores a live batch without changing provenance."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from clash_sos.application.live_model import LiveModelError, score_live_decks
from clash_sos.domain.analytics import PredictionState
from clash_sos.domain.model_artifact import ModelArtifactManifest, ModelOutputFile
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.ml import attention_artifact_io


def test_attention_batch_loads_once_and_retains_training_era(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "manifest.json").write_text('{"manifest_type":"matchup_attention"}')
    seen: list[torch.Tensor] = []

    class Schema:
        identity_vocab = ("one", "two", "three", "four")

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
        [
            (("one",) * 8, ("two",) * 8),
            (("future",) * 8, ("two",) * 8),
            (("three",) * 8, ("four",) * 8),
        ],
    )

    assert loads == [tmp_path]
    assert len(seen) == 1
    assert seen[0].shape == (2, 2, 8)
    assert info.training_era_id == "2026-06"
    assert predictions[0].side_a_win_probability == 0.5
    assert len(predictions) == 3
    assert predictions[1].state is PredictionState.UNAVAILABLE
    assert predictions[1].side_a_win_probability is None
    assert predictions[2].side_a_win_probability == pytest.approx(0.7310586)
    assert predictions[0].provenance == predictions[2].provenance == info.provenance


def test_missing_artifact_fails_with_model_unavailable(tmp_path: Path) -> None:
    with pytest.raises(LiveModelError, match="model_unavailable"):
        score_live_decks(tmp_path, [])


@pytest.mark.parametrize("corrupt_catalog", [False, True])
def test_baseline_live_coverage_uses_verified_frozen_catalog(
    tmp_path: Path, corrupt_catalog: bool
) -> None:
    members = {
        "card_catalog": ("catalog.json", KAGGLE_V6_CARDS.serialize()),
        "evaluation": ("evaluation.json", b"{}"),
        "feature_schema": ("feature-schema.json", b'{"feature_schema_version":"card-log-odds:v1"}'),
        "predictor": ("predictor.json", b'{"effects":{}}'),
    }
    for path, body in members.values():
        (tmp_path / path).write_bytes(body)
    files = tuple(
        sorted(
            (
                ModelOutputFile.model_validate(
                    {
                        "path": path,
                        "kind": kind,
                        "size_bytes": len(body),
                        "sha256": sha256(body).hexdigest(),
                    }
                )
                for kind, (path, body) in members.items()
            ),
            key=lambda file: file.path,
        )
    )
    manifest = ModelArtifactManifest(
        model_version="baseline",
        dataset_version="test",
        catalog_version=KAGGLE_V6_CARDS.version,
        balance_era_id="2026-06",
        files=files,
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    if corrupt_catalog:
        (tmp_path / "catalog.json").write_bytes(b"{}")
        with pytest.raises(LiveModelError, match="model_unavailable"):
            score_live_decks(tmp_path, [])
    else:
        known = tuple(entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries[:8])
        _, predictions = score_live_decks(
            tmp_path, [(known, known), (("future:base", *known[1:]), known)]
        )
        assert predictions[0].side_a_win_probability == 0.5
        assert predictions[1].state is PredictionState.UNAVAILABLE
