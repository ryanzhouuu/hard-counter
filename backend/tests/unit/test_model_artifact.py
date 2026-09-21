from math import log
from typing import Literal

import pytest
from pydantic import ValidationError

from clash_sos.domain.matchup_baseline import DEFAULT_MIRROR_SEED, DEFAULT_SMOOTHING_ALPHA
from clash_sos.domain.model_artifact import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_PAIR_MODEL_VERSION,
    EvaluationReport,
    ModelArtifactManifest,
    ModelOutputFile,
    ProbabilityMetricAccumulator,
    ProbabilityMetrics,
    SplitEvaluation,
    brier_score,
    dump_evaluation_report,
    dump_model_manifest,
    expected_calibration_error,
    log_loss,
)

SHA256 = "a" * 64


def metrics(*, row_count: int = 4) -> ProbabilityMetrics:
    return ProbabilityMetrics(
        log_loss=0.69,
        brier_score=0.25,
        expected_calibration_error=0.01,
        row_count=row_count,
    )


def output_file(
    path: str, kind: Literal["card_catalog", "evaluation", "feature_schema", "predictor"]
) -> ModelOutputFile:
    return ModelOutputFile(path=path, kind=kind, size_bytes=12, sha256=SHA256)


def test_log_loss_for_constant_half_is_negative_log_half() -> None:
    assert log_loss((1, 0, 1), (0.5, 0.5, 0.5)) == pytest.approx(-log(0.5))


def test_brier_score_for_constant_half_is_one_quarter() -> None:
    assert brier_score((1, 0), (0.5, 0.5)) == pytest.approx(0.25)


def test_expected_calibration_error_is_zero_when_perfect() -> None:
    assert expected_calibration_error((1, 0, 1, 0), (1.0, 0.0, 1.0, 0.0), bins=2) == pytest.approx(
        0.0
    )


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        log_loss((1,), (0.5, 0.5))


def test_accumulator_matches_batch_metrics() -> None:
    labels = (1, 0, 1, 0, 1)
    probabilities = (0.0, 1.0, 0.5, 0.25, 0.9)
    accumulator = ProbabilityMetricAccumulator()
    for label, probability in zip(labels, probabilities, strict=True):
        accumulator.update(label, probability)
    metrics = accumulator.finalize()
    assert metrics.log_loss == pytest.approx(log_loss(labels, probabilities))
    assert metrics.brier_score == pytest.approx(brier_score(labels, probabilities))
    assert metrics.expected_calibration_error == pytest.approx(
        expected_calibration_error(labels, probabilities)
    )
    assert metrics.row_count == len(labels)


def test_default_model_literals() -> None:
    assert DEFAULT_MODEL_VERSION == "kaggle-v6-ranked16-card-logodds-v1"
    assert DEFAULT_PAIR_MODEL_VERSION == "kaggle-v6-ranked16-card-pair-v1"
    assert DEFAULT_SMOOTHING_ALPHA == 1.0
    assert DEFAULT_MIRROR_SEED == 0


def test_card_log_odds_report_omits_pair_metrics() -> None:
    dumped = dump_evaluation_report(
        EvaluationReport(
            splits=(
                SplitEvaluation(
                    split="temporal",
                    partition="validation",
                    prior=metrics(),
                    exact_matchup=metrics(),
                    card_log_odds=metrics(),
                ),
            )
        )
    )
    report = EvaluationReport.model_validate_json(dumped)
    assert report.promoted_model == "card_log_odds"
    assert report.splits[0].card_pair is None


def test_card_pair_report_requires_pair_metrics() -> None:
    with pytest.raises(ValidationError, match="card_pair"):
        EvaluationReport(
            promoted_model="card_pair",
            splits=(
                SplitEvaluation(
                    split="temporal",
                    partition="validation",
                    prior=metrics(),
                    exact_matchup=metrics(),
                    card_log_odds=metrics(),
                ),
            ),
        )


def test_card_pair_report_round_trips() -> None:
    report = EvaluationReport(
        promoted_model="card_pair",
        splits=(
            SplitEvaluation(
                split="temporal",
                partition="validation",
                prior=metrics(),
                exact_matchup=metrics(),
                card_log_odds=metrics(),
                card_pair=metrics(),
            ),
        ),
    )
    dumped = dump_evaluation_report(report)
    loaded = EvaluationReport.model_validate_json(dumped)
    assert loaded.promoted_model == "card_pair"
    assert loaded.splits[0].card_pair == metrics()


def test_manifest_requires_inventoried_artifact_files() -> None:
    evaluation = EvaluationReport(
        splits=(
            SplitEvaluation(
                split="temporal",
                partition="validation",
                prior=metrics(),
                exact_matchup=metrics(),
                card_log_odds=metrics(),
            ),
        )
    )
    manifest = ModelArtifactManifest(
        model_version=DEFAULT_MODEL_VERSION,
        dataset_version="kaggle-v6-ranked16-v2",
        catalog_version="kaggle-v6-2026-06",
        balance_era_id="2026-06",
        smoothing_alpha=1.0,
        mirror_seed=0,
        files=(
            output_file("card-catalog.json", "card_catalog"),
            output_file("evaluation.json", "evaluation"),
            output_file("feature-schema.json", "feature_schema"),
            output_file("predictor.json", "predictor"),
        ),
    )
    dumped = dump_model_manifest(manifest)
    assert dumped == dump_model_manifest(ModelArtifactManifest.model_validate_json(dumped))
    dumped_eval = dump_evaluation_report(evaluation)
    assert dumped_eval == dump_evaluation_report(EvaluationReport.model_validate_json(dumped_eval))


def test_manifest_rejects_self_inventory() -> None:
    with pytest.raises(ValidationError, match=r"manifest\.json"):
        ModelArtifactManifest(
            model_version=DEFAULT_MODEL_VERSION,
            dataset_version="kaggle-v6-ranked16-v2",
            catalog_version="catalog",
            balance_era_id="2026-06",
            smoothing_alpha=1.0,
            mirror_seed=0,
            files=(
                output_file("card-catalog.json", "card_catalog"),
                output_file("evaluation.json", "evaluation"),
                output_file("feature-schema.json", "feature_schema"),
                output_file("manifest.json", "predictor"),
            ),
        )
