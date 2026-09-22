from datetime import UTC, datetime
from json import loads
from math import log
from pathlib import Path

import duckdb
import pytest
from test_model_train import (
    CONFIG,
    FORMS,
    LOSE_IDS,
    WIN_IDS,
    train_processed_manifest,
    write_dataset,
)

from clash_sos.application.model_train import KaggleV6ModelTrainError, predict_matchup
from clash_sos.application.model_train_lgbm import stream_presence_matrix, train_lightgbm_model
from clash_sos.domain.analytics import PredictionState
from clash_sos.domain.matchup_baseline import card_identity_key, should_mirror_sides
from clash_sos.domain.matchup_lgbm import PresenceSchema
from clash_sos.domain.model_artifact import EvaluationReport, ModelArtifactManifest
from clash_sos.domain.player_skill import PlayerSkillTracker
from clash_sos.domain.processed_manifest import dump_processed_manifest
from clash_sos.infrastructure.kaggle_v6.train_io import OrientedExample


def _fingerprints(*, mirrored: bool, count: int) -> tuple[str, ...]:
    found: list[str] = []
    index = 0
    while len(found) < count:
        candidate = f"fp-lgbm-{mirrored}-{index}"
        index += 1
        if should_mirror_sides(candidate) is mirrored:
            found.append(candidate)
    return tuple(found)


def write_contrast_dataset(path: Path) -> Path:
    """Seven temporal-train rows: three wins, three mirrored losses, then a watch win."""
    path.mkdir()
    positive = _fingerprints(mirrored=False, count=4)
    negative = _fingerprints(mirrored=True, count=3)
    train_rows = tuple(
        (datetime(2026, 6, day, tzinfo=UTC), fingerprint, number, "#A", "#B")
        for number, (day, fingerprint) in enumerate(
            (
                (1, positive[0]),
                (2, positive[1]),
                (3, positive[2]),
                (4, negative[0]),
                (5, negative[1]),
                (6, negative[2]),
                (7, positive[3]),
            ),
            start=0,
        )
    )
    holdout = (
        (datetime(2026, 6, 15, tzinfo=UTC), "fp-val", 7, "validation"),
        (datetime(2026, 6, 20, tzinfo=UTC), "fp-test", 8, "test"),
    )
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
                side_b_deck_hash VARCHAR,
                side_a_player_id VARCHAR,
                side_b_player_id VARCHAR
            )
            """
        )
        for timestamp, fingerprint, number, player_a, player_b in train_rows:
            connection.execute(
                """
                INSERT INTO battles
                VALUES (?, ?, 'a.parquet', ?, 'train', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    timestamp,
                    fingerprint,
                    number,
                    WIN_IDS,
                    FORMS,
                    LOSE_IDS,
                    FORMS,
                    "win-hash",
                    "lose-hash",
                    f"{player_a}{number}",
                    f"{player_b}{number}",
                ],
            )
        for timestamp, fingerprint, number, partition in holdout:
            connection.execute(
                "INSERT INTO battles VALUES (?, ?, 'a.parquet', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    timestamp,
                    fingerprint,
                    number,
                    partition,
                    WIN_IDS,
                    FORMS,
                    LOSE_IDS,
                    FORMS,
                    f"{partition}-win",
                    f"{partition}-lose",
                    f"#{partition}-a",
                    f"#{partition}-b",
                ],
            )
        connection.execute(
            """
            COPY (
                SELECT timestamp, fingerprint, archive_member, row_number,
                       side_a_card_ids, side_a_card_forms, side_b_card_ids,
                       side_b_card_forms, side_a_deck_hash, side_b_deck_hash,
                       side_a_player_id, side_b_player_id
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
    (path / "manifest.json").write_bytes(
        dump_processed_manifest(train_processed_manifest(train_rows=7))
    )
    return path


def test_presence_artifact_schema_still_loads() -> None:
    schema = PresenceSchema(("archers:base", "knight:base"))
    row = schema.row(("knight:base",), ("archers:base",), skill_diff=0.0)
    assert schema.skill_column == 4
    assert row.indices[-1] == 4


def test_stream_presence_matrix_excludes_the_current_outcome() -> None:
    schema = PresenceSchema(("archers:base", "knight:base"))
    examples = (
        OrientedExample(1, ("knight:base",), ("archers:base",), "win", "lose", "a", "b"),
        OrientedExample(0, ("knight:base",), ("archers:base",), "win", "lose", "a", "b"),
    )
    matrix, labels = stream_presence_matrix(iter(examples), schema, PlayerSkillTracker(), 2)
    assert labels.tolist() == [1.0, 0.0]
    assert float(matrix[0, schema.skill_column]) == 0.0
    assert float(matrix[1, schema.skill_column]) > 0.0


def test_stream_presence_matrix_rejects_more_than_eight_cards_per_side() -> None:
    identities = tuple(f"card-{index}:base" for index in range(20))
    side = tuple(identities[:9])
    example = OrientedExample(1, side, side, "win", "lose", "a", "b")
    with pytest.raises(KaggleV6ModelTrainError, match="presence budget"):
        stream_presence_matrix(
            iter((example,)),
            PresenceSchema(identities),
            PlayerSkillTracker(),
            1,
        )


def test_train_lightgbm_writes_reloadable_symmetric_artifact(tmp_path: Path) -> None:
    dataset = write_contrast_dataset(tmp_path / "dataset")
    destination = tmp_path / "model"
    published = train_lightgbm_model(
        dataset,
        destination,
        output_workspace=tmp_path / "output",
        temp_directory=tmp_path / "tmp",
        config=CONFIG,
        learning_rate=0.5,
        min_data_in_leaf=1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        lambda_l2=0.0,
        num_threads=1,
        max_rounds=30,
        early_stopping_rounds=5,
    )
    assert published == destination
    assert (destination / "predictor.txt").is_file()
    assert not (destination / "predictor.json").exists()
    manifest = ModelArtifactManifest.model_validate_json(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.model_version == "kaggle-v6-ranked16-lightgbm-v2"
    report = EvaluationReport.model_validate_json(
        (destination / "evaluation.json").read_text(encoding="utf-8")
    )
    assert report.promoted_model == "lightgbm"
    feature_schema = loads((destination / "feature-schema.json").read_text(encoding="utf-8"))
    assert feature_schema["feature_schema_version"] == "lightgbm-summaries:v1"
    assert feature_schema["skill_control"] == "past_laplace"
    assert feature_schema["skill_column"] == 366
    assert feature_schema["attribute_version"] == "card-attributes:2026-06"
    attributes = feature_schema["attributes"]
    assert isinstance(attributes, dict)
    cards = attributes["cards"]
    assert isinstance(cards, dict)
    assert cards["goblin-hut"]["elixir"] == 4
    assert cards["mirror"]["elixir"] is None
    assert feature_schema["best_iteration"] >= 1
    assert feature_schema["watch_fraction"] == 0.1
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
    validation = next(
        item
        for item in report.splits
        if item.split == "temporal" and item.partition == "validation"
    )
    assert validation.lightgbm is not None
    assert validation.card_pair is None
    assert validation.prior.log_loss == pytest.approx(-log(0.5))


def test_train_lightgbm_refuses_existing_destination(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    destination = tmp_path / "model"
    destination.mkdir()
    with pytest.raises(KaggleV6ModelTrainError, match="already exists"):
        train_lightgbm_model(
            dataset,
            destination,
            output_workspace=tmp_path / "output",
            temp_directory=tmp_path / "tmp",
            config=CONFIG,
        )


def test_train_lightgbm_refuses_join_count_mismatch(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    (dataset / "manifest.json").write_bytes(
        dump_processed_manifest(train_processed_manifest(train_rows=2))
    )
    with pytest.raises(KaggleV6ModelTrainError, match="join count 1 != 2"):
        train_lightgbm_model(
            dataset,
            tmp_path / "model",
            output_workspace=tmp_path / "output",
            temp_directory=tmp_path / "tmp",
            config=CONFIG,
        )


def test_train_lightgbm_refuses_catalog_version_mismatch(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    manifest = train_processed_manifest().model_copy(update={"catalog_version": "other-catalog"})
    (dataset / "manifest.json").write_bytes(dump_processed_manifest(manifest))
    with pytest.raises(KaggleV6ModelTrainError, match="catalog"):
        train_lightgbm_model(
            dataset,
            tmp_path / "model",
            output_workspace=tmp_path / "output",
            temp_directory=tmp_path / "tmp",
            config=CONFIG,
        )
