from datetime import UTC, datetime
from math import log
from pathlib import Path

import duckdb
import pytest
from test_processed_manifest import (
    TIMESTAMP_MAX,
    TIMESTAMP_MIN,
    accepted_summary,
    disposition_summary,
    player_disjoint_split,
    processed_manifest,
    temporal_split,
)

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.model_train import (
    KaggleV6ModelTrainError,
    predict_matchup,
    score_partition,
    train_matchup_baseline,
)
from clash_sos.domain.analytics import PredictionState
from clash_sos.domain.canonical import RecordState
from clash_sos.domain.matchup_baseline import card_identity_key
from clash_sos.domain.model_artifact import EvaluationReport, ModelArtifactManifest
from clash_sos.domain.processed_manifest import (
    EraCount,
    PlayerDisjointPartitionSummary,
    ProcessedDatasetManifest,
    StateCount,
    TemporalPartitionSummary,
    dump_processed_manifest,
)
from clash_sos.infrastructure.kaggle_v6.train_io import OrientedExample

STAMP = datetime(2026, 6, 15, tzinfo=UTC)
WIN_IDS = ["knight", "mini-pekka", "musketeer", "valkyrie", "hog", "fireball", "log", "cannon"]
LOSE_IDS = ["archers", "goblins", "bomber", "skeletons", "tombstone", "zap", "arrows", "tesla"]
FORMS = ["base"] * 8
CONFIG = StagingConfig(memory_limit="256MB", threads=1, chunk_size=1024)


def train_processed_manifest(
    *,
    train_rows: int = 1,
    validation_rows: int = 1,
    test_rows: int = 1,
    excluded_bridge_rows: int = 0,
) -> ProcessedDatasetManifest:
    accepted = train_rows + validation_rows + test_rows
    valid = accepted
    unsupported = 100 - 10 - 20 - valid
    return processed_manifest().model_validate(
        {
            **processed_manifest().model_dump(mode="python"),
            "dispositions": disposition_summary(
                states=(
                    StateCount(state=RecordState.INSUFFICIENT_DATA, count=0),
                    StateCount(state=RecordState.INVALID, count=10),
                    StateCount(state=RecordState.QUARANTINED, count=20),
                    StateCount(state=RecordState.UNSUPPORTED, count=unsupported),
                    StateCount(state=RecordState.VALID, count=valid),
                )
            ),
            "accepted": accepted_summary(
                row_count=accepted,
                eras=(EraCount(era_id="2026-06", count=accepted),),
            ),
            "temporal_split": temporal_split(
                partitions=(
                    TemporalPartitionSummary(
                        partition="train",
                        row_count=train_rows,
                        timestamp_min=TIMESTAMP_MIN,
                        timestamp_max=datetime(2026, 6, 9, tzinfo=UTC),
                    ),
                    TemporalPartitionSummary(
                        partition="validation",
                        row_count=validation_rows,
                        timestamp_min=datetime(2026, 6, 10, tzinfo=UTC),
                        timestamp_max=datetime(2026, 6, 19, tzinfo=UTC),
                    ),
                    TemporalPartitionSummary(
                        partition="test",
                        row_count=test_rows,
                        timestamp_min=datetime(2026, 6, 20, tzinfo=UTC),
                        timestamp_max=TIMESTAMP_MAX,
                    ),
                )
            ),
            "player_disjoint_split": player_disjoint_split(
                excluded_bridge_rows=excluded_bridge_rows,
                partitions=(
                    PlayerDisjointPartitionSummary(partition="train", row_count=train_rows),
                    PlayerDisjointPartitionSummary(
                        partition="validation", row_count=validation_rows
                    ),
                    PlayerDisjointPartitionSummary(partition="test", row_count=test_rows),
                ),
            ),
        }
    )


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
    (path / "manifest.json").write_bytes(dump_processed_manifest(train_processed_manifest()))
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
    assert manifest.dataset_version == train_processed_manifest().dataset_version
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


def test_score_partition_consumes_a_one_shot_iterator() -> None:
    example = OrientedExample(
        label=1,
        side_a_keys=("knight:base",),
        side_b_keys=("archers:base",),
        deck_a_hash="win-hash",
        deck_b_hash="lose-hash",
    )
    prior, exact, card = score_partition(
        (item for item in (example,)),
        effects={"knight:base": 0.8},
        matchup_counts={("win-hash", "lose-hash"): (1, 1)},
        alpha=1.0,
    )
    assert prior.row_count == 1
    assert exact.row_count == 1
    assert card.row_count == 1
    assert exact.log_loss < prior.log_loss
    assert card.log_loss < prior.log_loss


def test_train_refuses_join_count_mismatch(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    (dataset / "manifest.json").write_bytes(
        dump_processed_manifest(train_processed_manifest(train_rows=2))
    )
    with pytest.raises(KaggleV6ModelTrainError, match="join count 1 != 2"):
        train_matchup_baseline(
            dataset,
            tmp_path / "model",
            output_workspace=tmp_path / "output",
            temp_directory=tmp_path / "tmp",
            config=CONFIG,
        )


def test_train_refuses_catalog_version_mismatch(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    manifest = train_processed_manifest().model_copy(update={"catalog_version": "other-catalog"})
    (dataset / "manifest.json").write_bytes(dump_processed_manifest(manifest))
    with pytest.raises(KaggleV6ModelTrainError, match="catalog"):
        train_matchup_baseline(
            dataset,
            tmp_path / "model",
            output_workspace=tmp_path / "output",
            temp_directory=tmp_path / "tmp",
            config=CONFIG,
        )


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
