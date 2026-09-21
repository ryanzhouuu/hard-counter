from json import loads
from math import log
from pathlib import Path

import pytest
from test_model_train import CONFIG, LOSE_IDS, WIN_IDS, train_processed_manifest, write_dataset

from clash_sos.application.model_train import (
    KaggleV6ModelTrainError,
    predict_matchup,
    score_partition,
)
from clash_sos.application.model_train_pair import train_card_pair_model
from clash_sos.domain.analytics import PredictionState
from clash_sos.domain.matchup_baseline import card_identity_key
from clash_sos.domain.matchup_pair import CardPairPredictor
from clash_sos.domain.model_artifact import EvaluationReport, ModelArtifactManifest
from clash_sos.domain.processed_manifest import dump_processed_manifest
from clash_sos.infrastructure.kaggle_v6.train_io import OrientedExample


def read_evaluation(destination: Path) -> EvaluationReport:
    return EvaluationReport.model_validate_json(
        (destination / "evaluation.json").read_text(encoding="utf-8")
    )


def test_score_partition_includes_pair_predictor() -> None:
    example = OrientedExample(
        label=1,
        side_a_keys=("knight:base",),
        side_b_keys=("archers:base",),
        deck_a_hash="win-hash",
        deck_b_hash="lose-hash",
    )
    predictor = CardPairPredictor(
        identities=("knight:base", "archers:base"),
        additive=[0.8, -0.2],
        pair_upper=[0.4],
    )
    scores = score_partition(
        (item for item in (example,)),
        effects={"knight:base": 0.8},
        matchup_counts={("win-hash", "lose-hash"): (1, 1)},
        alpha=1.0,
        pair_predictor=predictor,
    )
    assert scores.prior.row_count == 1
    assert scores.card_pair is not None
    assert scores.card_pair.row_count == 1
    assert scores.card_pair.log_loss < scores.prior.log_loss


def test_train_card_pair_model_writes_reloadable_artifact(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    destination = tmp_path / "model"
    published = train_card_pair_model(
        dataset,
        destination,
        output_workspace=tmp_path / "output",
        temp_directory=tmp_path / "tmp",
        config=CONFIG,
    )
    assert published == destination
    manifest = ModelArtifactManifest.model_validate_json(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.dataset_version == train_processed_manifest().dataset_version
    assert manifest.model_version == "kaggle-v6-ranked16-card-pair-v3"
    schema = loads((destination / "feature-schema.json").read_text(encoding="utf-8"))
    predictor = loads((destination / "predictor.json").read_text(encoding="utf-8"))
    assert schema["epochs"] == 1
    assert schema["learning_rate"] == 0.001
    assert schema["pair_init_scale"] == 8.0
    assert schema["skill_control"] == "past_laplace"
    assert schema["skill_alpha"] == 8.0
    assert schema["skill_coefficient"] > 0
    assert "skill_coefficient" not in predictor
    assert predictor["pair_init_scale"] == 8.0
    knight = tuple(card_identity_key(card, "base") for card in WIN_IDS)
    archers = tuple(card_identity_key(card, "base") for card in LOSE_IDS)
    prediction = predict_matchup(destination, knight, archers)
    swapped = predict_matchup(destination, archers, knight)
    equal = predict_matchup(destination, knight, knight)
    assert prediction.state is PredictionState.AVAILABLE
    assert prediction.side_a_win_probability is not None
    assert prediction.side_a_win_probability > 0.5
    assert swapped.side_a_win_probability == pytest.approx(1 - prediction.side_a_win_probability)
    assert equal.side_a_win_probability == pytest.approx(0.5)


def test_train_card_pair_promotes_pair_metrics(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    destination = tmp_path / "model"
    train_card_pair_model(
        dataset,
        destination,
        output_workspace=tmp_path / "output",
        temp_directory=tmp_path / "tmp",
        config=CONFIG,
    )
    report = read_evaluation(destination)
    assert report.promoted_model == "card_pair"
    validation = next(
        item
        for item in report.splits
        if item.split == "temporal" and item.partition == "validation"
    )
    assert validation.card_pair is not None
    assert validation.card_pair.log_loss <= validation.card_log_odds.log_loss
    assert validation.card_pair.log_loss < validation.prior.log_loss
    assert validation.prior.log_loss == pytest.approx(-log(0.5))


def test_train_card_pair_refuses_join_count_mismatch(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    (dataset / "manifest.json").write_bytes(
        dump_processed_manifest(train_processed_manifest(train_rows=2))
    )
    with pytest.raises(KaggleV6ModelTrainError, match="join count 1 != 2"):
        train_card_pair_model(
            dataset,
            tmp_path / "model",
            output_workspace=tmp_path / "output",
            temp_directory=tmp_path / "tmp",
            config=CONFIG,
        )


def test_train_card_pair_refuses_catalog_version_mismatch(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    manifest = train_processed_manifest().model_copy(update={"catalog_version": "other-catalog"})
    (dataset / "manifest.json").write_bytes(dump_processed_manifest(manifest))
    with pytest.raises(KaggleV6ModelTrainError, match="catalog"):
        train_card_pair_model(
            dataset,
            tmp_path / "model",
            output_workspace=tmp_path / "output",
            temp_directory=tmp_path / "tmp",
            config=CONFIG,
        )


def test_train_card_pair_refuses_existing_destination(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    destination = tmp_path / "model"
    destination.mkdir()
    with pytest.raises(KaggleV6ModelTrainError, match="already exists"):
        train_card_pair_model(
            dataset,
            destination,
            output_workspace=tmp_path / "output",
            temp_directory=tmp_path / "tmp",
            config=CONFIG,
        )
