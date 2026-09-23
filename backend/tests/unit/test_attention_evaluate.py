"""Score tiny validated cache rows with aligned prediction sidecars."""

from pathlib import Path

import polars as pl
import pytest
from test_attention_cache_read import write_cache

from clash_sos.application.attention_evaluate import (
    AttentionEvaluationReport,
    evaluate_attention_cache,
)
from clash_sos.application.attention_support import AttentionSupportIndex
from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_schema import AttentionModelConfig, build_attention_schema
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import load_attention_cache
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def test_evaluation_writes_reloadable_report_and_aligned_predictions(tmp_path: Path) -> None:
    dataset, directory, schema, protocol = write_cache(tmp_path)
    cache = load_attention_cache(dataset, directory, protocol, schema, chunk_size=1024)
    support = AttentionSupportIndex.build(cache, tmp_path / "support.db")
    try:
        destination = tmp_path / "evaluation"
        report = evaluate_attention_cache(
            cache,
            AttentionMatchupModel(schema),
            fit_protocol=protocol,
            fit_artifact_id="fixture-attention-v1",
            role="development",
            support=support,
            output_directory=destination,
            batch_size=1,
            comparator=lambda _tokens, _row: 0.5,
        )
        assert report.row_count == protocol.development.row_count == 2
        assert report.protocol.canonical_sha256 == protocol.canonical_sha256
        assert report.evaluation.overall.metrics.row_count == 2
        assert report.evaluation.groups["deck:both_seen"].metrics.row_count == 2
        assert (
            AttentionEvaluationReport.model_validate_json(
                (destination / "report.json").read_bytes()
            )
            == report
        )
        predictions = pl.read_parquet(destination / "predictions-00000.parquet")
        assert predictions["row_ordinal"].to_list() == [0, 1]
        assert predictions["label"].to_list() == [
            int(labels[0]) for _, labels in cache.iter_batches("development", batch_size=1)
        ]
        assert predictions["paired_loss_difference"].null_count() == 0
    finally:
        support.close()


def test_evaluation_rejects_wrong_schema_and_cleans_failed_output(tmp_path: Path) -> None:
    dataset, directory, schema, protocol = write_cache(tmp_path)
    cache = load_attention_cache(dataset, directory, protocol, schema, chunk_size=1024)
    support = AttentionSupportIndex.build(cache, tmp_path / "support.db")
    try:
        incompatible = build_attention_schema(
            KAGGLE_V6_CARDS.serialize(),
            attributes=CARD_ATTRIBUTES,
            network=AttentionModelConfig(embedding_width=128),
        )
        with pytest.raises(ValueError, match="encoding"):
            evaluate_attention_cache(
                cache,
                AttentionMatchupModel(incompatible),
                fit_protocol=protocol,
                fit_artifact_id="fixture",
                role="development",
                support=support,
                output_directory=tmp_path / "bad-model",
            )
        assert not (tmp_path / "bad-model").exists()

        def broken_comparator(_tokens: object, _row: object) -> float:
            """Simulate an evaluation-only scorer failure after output creation."""
            raise RuntimeError("comparator failed")

        with pytest.raises(RuntimeError, match="comparator failed"):
            evaluate_attention_cache(
                cache,
                AttentionMatchupModel(schema),
                fit_protocol=protocol,
                fit_artifact_id="fixture",
                role="development",
                support=support,
                output_directory=tmp_path / "failed-output",
                comparator=broken_comparator,
            )
        assert not (tmp_path / "failed-output").exists()
    finally:
        support.close()
