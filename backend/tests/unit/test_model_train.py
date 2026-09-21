from datetime import UTC, datetime
from math import log
from pathlib import Path

import duckdb
import pytest
from test_processed_manifest import processed_manifest

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.model_train import (
    KaggleV6ModelTrainError,
    predict_matchup,
    train_matchup_baseline,
)
from clash_sos.domain.analytics import PredictionState
from clash_sos.domain.matchup_baseline import card_identity_key
from clash_sos.domain.model_artifact import EvaluationReport, ModelArtifactManifest
from clash_sos.domain.processed_manifest import dump_processed_manifest

STAMP = datetime(2026, 6, 15, tzinfo=UTC)
WIN_IDS = ["knight", "mini-pekka", "musketeer", "valkyrie", "hog", "fireball", "log", "cannon"]
LOSE_IDS = ["archers", "goblins", "bomber", "skeletons", "tombstone", "zap", "arrows", "tesla"]
FORMS = ["base"] * 8
CONFIG = StagingConfig(memory_limit="256MB", threads=1, chunk_size=1024)


def write_dataset(path: Path) -> Path:
    path.mkdir()
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE battles (
                timestamp TIMESTAMPTZ,
                fingerprint VARCHAR,
                archive_member VARCHAR,
                row_number BIGINT,
                partition VARCHAR,
                side_a_card_ids VARCHAR[],
                side_a_card_forms VARCHAR[],
                side_b_card_ids VARCHAR[],
                side_b_card_forms VARCHAR[],
                side_a_deck_hash VARCHAR,
                side_b_deck_hash VARCHAR
            )
            """
        )
        rows = (
            ("fp-train", 0, "train", "win-hash", "lose-hash"),
            ("fp-val", 1, "validation", "val-win-hash", "val-lose-hash"),
            ("fp-test", 2, "test", "test-win-hash", "test-lose-hash"),
        )
        for fingerprint, number, partition, win_hash, lose_hash in rows:
            connection.execute(
                "INSERT INTO battles VALUES (?, ?, 'a.parquet', ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    STAMP,
                    fingerprint,
                    number,
                    partition,
                    WIN_IDS,
                    FORMS,
                    LOSE_IDS,
                    FORMS,
                    win_hash,
                    lose_hash,
                ],
            )
        connection.execute(
            """
            COPY (
                SELECT timestamp, fingerprint, archive_member, row_number,
                       side_a_card_ids, side_a_card_forms, side_b_card_ids,
                       side_b_card_forms, side_a_deck_hash, side_b_deck_hash
                FROM battles
            ) TO ? (FORMAT PARQUET)
            """,
            [str(path / "canonical.parquet")],
        )
        split_sql = """
            COPY (
                SELECT timestamp, fingerprint, archive_member, row_number, partition
                FROM battles
            ) TO ? (FORMAT PARQUET)
            """
        connection.execute(split_sql, [str(path / "splits-temporal.parquet")])
        connection.execute(split_sql, [str(path / "splits-player-disjoint.parquet")])
    finally:
        connection.close()
    (path / "dispositions").mkdir()
    (path / "dispositions" / "member-a.parquet").write_bytes(b"stub")
    (path / "verification-report.json").write_text("{}", encoding="utf-8")
    (path / "manifest.json").write_bytes(dump_processed_manifest(processed_manifest()))
    return path


def read_evaluation(destination: Path) -> EvaluationReport:
    return EvaluationReport.model_validate_json(
        (destination / "evaluation.json").read_text(encoding="utf-8")
    )


def test_train_matchup_baseline_writes_reloadable_artifact(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    destination = tmp_path / "model"
    published = train_matchup_baseline(
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
    assert manifest.dataset_version == processed_manifest().dataset_version
    assert manifest.fit_split == "temporal"
    assert manifest.fit_partition == "train"
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


def test_train_reports_exact_matchup_backoff_and_card_generalization(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    destination = tmp_path / "model"
    train_matchup_baseline(
        dataset,
        destination,
        output_workspace=tmp_path / "output",
        temp_directory=tmp_path / "tmp",
        config=CONFIG,
    )
    report = read_evaluation(destination)
    train = next(
        item for item in report.splits if item.split == "temporal" and item.partition == "train"
    )
    validation = next(
        item
        for item in report.splits
        if item.split == "temporal" and item.partition == "validation"
    )
    assert train.exact_matchup.log_loss < train.prior.log_loss
    assert validation.exact_matchup.log_loss == pytest.approx(validation.prior.log_loss)
    assert validation.card_log_odds.log_loss < validation.prior.log_loss
    assert train.prior.log_loss == pytest.approx(-log(0.5))


def test_train_refuses_existing_destination(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    destination = tmp_path / "model"
    destination.mkdir()
    with pytest.raises(KaggleV6ModelTrainError, match="already exists"):
        train_matchup_baseline(
            dataset,
            destination,
            output_workspace=tmp_path / "output",
            temp_directory=tmp_path / "tmp",
            config=CONFIG,
        )
