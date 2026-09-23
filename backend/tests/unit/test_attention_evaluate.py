"""Score tiny validated cache rows with aligned prediction sidecars."""

from math import log
from pathlib import Path

import polars as pl
import pytest
import torch
from test_attention_cache_read import write_cache
from torch import Tensor

from clash_sos.application.attention_evaluate import (
    AttentionEvaluationReport,
    evaluate_attention_cache,
)
from clash_sos.application.attention_fit import AttentionFitConfig, fit_attention_model
from clash_sos.application.attention_support import AttentionSupportIndex
from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_schema import (
    AttentionCardSchema,
    AttentionModelConfig,
    build_attention_schema,
)
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import load_attention_cache
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


class ExtremeLogitModel(AttentionMatchupModel):
    """Emit known wrong-side logits to check evaluator numeric precision."""

    def __init__(self, schema: AttentionCardSchema, logits: list[float]) -> None:
        super().__init__(schema)
        self.logits = iter(logits)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return one prepared float32 logit for each test row."""
        return torch.tensor(
            [next(self.logits) for _ in range(tokens.shape[0])],
            dtype=torch.float32,
            device=tokens.device,
        )


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


def test_extreme_logits_keep_mirrored_losses_and_pair_deltas_equal(tmp_path: Path) -> None:
    dataset, directory, schema, protocol = write_cache(tmp_path)
    cache = load_attention_cache(dataset, directory, protocol, schema, chunk_size=1024)
    labels = [int(label[0]) for _, label in cache.iter_batches("development", batch_size=1)]
    assert sorted(labels) == [0, 1]
    wrong_logits = [17.0 if label == 0 else -17.0 for label in labels]
    support = AttentionSupportIndex.build(cache, tmp_path / "support.db")
    try:
        destination = tmp_path / "extreme-evaluation"
        report = evaluate_attention_cache(
            cache,
            ExtremeLogitModel(schema, wrong_logits),
            fit_protocol=protocol,
            fit_artifact_id="extreme-logits",
            role="development",
            support=support,
            output_directory=destination,
            batch_size=1,
            comparator=lambda _tokens, _row: 0.5,
        )
        predictions = pl.read_parquet(destination / "predictions-00000.parquet")
        probabilities = predictions["probability"].to_list()
        deltas = predictions["paired_loss_difference"].to_list()
        assert all(0.0 < probability < 1.0 for probability in probabilities)
        assert report.evaluation.overall.metrics.log_loss == pytest.approx(17.0, abs=1e-5)
        assert deltas == pytest.approx([17.0 - log(2)] * 2, abs=1e-5)
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


def test_real_cache_fit_refit_and_evaluation_connect(tmp_path: Path) -> None:
    dataset, directory, schema, protocol = write_cache(tmp_path)
    cache = load_attention_cache(dataset, directory, protocol, schema, chunk_size=1024)
    fitted = fit_attention_model(
        schema,
        cache,
        AttentionFitConfig(batch_size=1, max_epochs=2, patience=2, seed=7, device="cpu"),
    )
    support = AttentionSupportIndex.build(cache, tmp_path / "support.db")
    try:
        report = evaluate_attention_cache(
            cache,
            fitted.refit_model,
            fit_protocol=protocol,
            fit_artifact_id="tiny-fitted-model",
            role="development",
            support=support,
            output_directory=tmp_path / "fitted-evaluation",
            batch_size=1,
        )
        assert report.row_count == 2
        assert report.evaluation.overall.metrics.log_loss > 0
        assert fitted.selected_epoch in (1, 2)
    finally:
        support.close()
