from json import loads
from pathlib import Path

import pytest
from tower_dataset_fixture import VERSION, stamp, tower_rows, write_tower_source
from typer.testing import CliRunner

from clash_sos.application.live_model import score_live_decks
from clash_sos.application.model_predict import predict_matchup
from clash_sos.domain.analytics import PredictionState
from clash_sos.domain.attention_cache import AttentionCacheManifest, dump_cache_manifest
from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.card_attributes import CardAttributeTable
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import load_attention_cache
from clash_sos.infrastructure.ml.attention_artifact_io import load_attention_artifact
from clash_sos.interfaces.cli.main import app

runner = CliRunner()


def test_official_snapshot_cli_trains_reloads_and_scores_towers(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_tower_source(source)
    dataset, artifact, cache = (tmp_path / name for name in ("dataset", "artifact", "cache"))
    network = tmp_path / "network.json"
    network.write_text('{"embedding_width":16,"feed_forward_width":32}')
    prepared = runner.invoke(
        app,
        [
            "dataset",
            "prepare-official-attention",
            "--source",
            str(source),
            "--destination",
            str(dataset),
            "--dataset-version",
            VERSION,
            "--balance-era",
            "2026-09",
            "--start",
            stamp(1).isoformat(),
            "--train-end",
            stamp(10).isoformat(),
            "--validation-end",
            stamp(20).isoformat(),
            "--end",
            stamp(27).isoformat(),
            "--network-config",
            str(network),
            "--watch-fraction",
            "0.3",
            "--batch-rows",
            "2",
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    trained = runner.invoke(
        app,
        [
            "model",
            "train-attention",
            "--dataset",
            str(dataset),
            "--destination",
            str(artifact),
            "--protocol",
            str(dataset / "protocol.json"),
            "--feature-schema",
            str(dataset / "feature-schema.json"),
            "--cache-directory",
            str(cache),
            "--output-workspace",
            str(tmp_path / "output"),
            "--model-version",
            "official-tower-test",
            "--device",
            "cpu",
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--batch-rows",
            "2",
        ],
    )
    assert trained.exit_code == 0, trained.output
    manifest, schema, _ = load_attention_artifact(artifact)
    assert schema.input_size == 9 and len(schema.identity_vocab) == 186
    assert CardAttributeTable.from_payload(schema.attribute_snapshot).cards["void"].elixir == 5
    assert manifest.probability_interpretation == schema.probability_interpretation
    evaluation = loads((artifact / "evaluation.json").read_bytes())
    assert all(
        report["probability_interpretation"] == schema.probability_interpretation
        for report in evaluation["reports"]
    )
    row = tower_rows()[0]
    first, second = (
        tuple(f"{card}:{form}" for card, form in zip(ids, forms, strict=True))
        for ids, forms in (
            (row.side_a_card_ids, row.side_a_card_forms),
            (row.side_b_card_ids, row.side_b_card_forms),
        )
    )
    info, scores = score_live_decks(
        artifact,
        [(first, second)] * 2,
        tower_pairs=[(row.side_a_tower, row.side_b_tower), (None, row.side_b_tower)],
    )
    assert info.input_scope == "deck_and_tower"
    assert scores[0].state == PredictionState.AVAILABLE
    assert scores[1].state == PredictionState.UNAVAILABLE
    predicted = predict_matchup(
        artifact,
        first,
        second,
        balance_era_id="2026-09",
        side_a_tower=row.side_a_tower,
        side_b_tower=row.side_b_tower,
    )
    assert predicted.side_a_win_probability == pytest.approx(scores[0].side_a_win_probability)
    swapped = predict_matchup(
        artifact,
        second,
        first,
        balance_era_id="2026-09",
        side_a_tower=row.side_b_tower,
        side_b_tower=row.side_a_tower,
    )
    assert predicted.side_a_win_probability is not None
    assert swapped.side_a_win_probability == pytest.approx(1 - predicted.side_a_win_probability)
    protocol = AttentionProtocol.model_validate_json((dataset / "protocol.json").read_bytes())
    cache_path = cache / "manifest.json"
    cache_manifest = AttentionCacheManifest.model_validate_json(cache_path.read_bytes())
    cache_path.write_bytes(
        dump_cache_manifest(
            cache_manifest.model_copy(update={"cache_version": "attention-input-cache:v1"})
        )
    )
    with pytest.raises(ValueError, match=r"version.*input layout"):
        load_attention_cache(dataset, cache, protocol, schema)


def test_official_preparation_cli_rejects_naive_bounds(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "dataset",
            "prepare-official-attention",
            "--source",
            "missing.jsonl",
            "--destination",
            str(tmp_path / "dataset"),
            "--dataset-version",
            VERSION,
            "--balance-era",
            "2026-09",
            "--start",
            "2026-09-01",
            "--train-end",
            "2026-09-10Z",
            "--validation-end",
            "2026-09-20Z",
            "--end",
            "2026-09-27Z",
        ],
    )
    assert result.exit_code != 0 and "timezone-aware" in result.output
    assert not (tmp_path / "dataset").exists()
