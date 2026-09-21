from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from clash_sos.domain.matchup_baseline import card_identity_key, should_mirror_sides
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb
from clash_sos.infrastructure.kaggle_v6.train_io import (
    KaggleV6TrainError,
    aggregate_card_counts,
    aggregate_matchup_counts,
    iter_oriented_examples,
    require_partition_rows,
)

KEEP_FP = next(
    f"fp-{index}" for index in range(256) if not should_mirror_sides(f"fp-{index}", seed=0)
)
SWAP_FP = next(f"fp-{index}" for index in range(256) if should_mirror_sides(f"fp-{index}", seed=0))
EARLY_FP = next(
    f"early-{index}" for index in range(256) if not should_mirror_sides(f"early-{index}", seed=0)
)
LATE_FP = next(
    f"late-{index}" for index in range(256) if not should_mirror_sides(f"late-{index}", seed=0)
)
STAMP = datetime(2026, 6, 15, tzinfo=UTC)
WIN_IDS = ["knight", "mini-pekka", "musketeer", "valkyrie", "hog", "fireball", "log", "cannon"]
LOSE_IDS = ["archers", "goblins", "bomber", "skeletons", "tombstone", "zap", "arrows", "tesla"]
FORMS = ["base"] * 8


def write_battle_parquet(
    path: Path,
    rows: tuple[tuple[str, str, int, str, list[str], list[str], str, str], ...],
    *,
    stamps: tuple[datetime, ...] | None = None,
) -> None:
    if stamps is not None and len(stamps) != len(rows):
        raise ValueError("stamps length must match rows")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        for index, (
            fingerprint,
            member,
            number,
            partition,
            a_ids,
            b_ids,
            a_hash,
            b_hash,
        ) in enumerate(rows):
            connection.execute(
                """
                INSERT INTO battles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    STAMP if stamps is None else stamps[index],
                    fingerprint,
                    member,
                    number,
                    partition,
                    a_ids,
                    FORMS,
                    b_ids,
                    FORMS,
                    a_hash,
                    b_hash,
                    "#WINNER",
                    "#LOSER",
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
            [str(path.with_name("canonical.parquet"))],
        )
        connection.execute(
            """
            COPY (
                SELECT timestamp, fingerprint, archive_member, row_number, partition
                FROM battles
            ) TO ? (FORMAT PARQUET)
            """,
            [str(path.with_name("splits-temporal.parquet"))],
        )
    finally:
        connection.close()


def connect(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    return connect_staging_duckdb(
        memory_limit="256MB", threads=1, temp_directory=tmp_path / "duckdb"
    )


def test_iter_oriented_examples_mirrors_winner_first_rows(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    split = tmp_path / "splits-temporal.parquet"
    write_battle_parquet(
        canonical,
        (
            (KEEP_FP, "a.parquet", 0, "train", WIN_IDS, LOSE_IDS, "win-hash", "lose-hash"),
            (SWAP_FP, "a.parquet", 1, "train", WIN_IDS, LOSE_IDS, "win-hash", "lose-hash"),
            ("fp-val", "a.parquet", 2, "validation", WIN_IDS, LOSE_IDS, "win-hash", "lose-hash"),
        ),
    )
    connection = connect(tmp_path)
    try:
        examples = tuple(
            iter_oriented_examples(
                connection,
                canonical_path=canonical,
                split_path=split,
                partition="train",
                seed=0,
            )
        )
    finally:
        connection.close()

    assert len(examples) == 2
    keep = next(example for example in examples if example.label == 1)
    swap = next(example for example in examples if example.label == 0)
    assert keep.side_a_keys[0] == card_identity_key("knight", "base")
    assert keep.deck_a_hash == "win-hash"
    assert swap.side_a_keys[0] == card_identity_key("archers", "base")
    assert swap.deck_a_hash == "lose-hash"


def test_aggregate_card_counts_credits_original_winners(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    split = tmp_path / "splits-temporal.parquet"
    write_battle_parquet(
        canonical,
        (("fp-keep", "a.parquet", 0, "train", WIN_IDS, LOSE_IDS, "win-hash", "lose-hash"),),
    )
    connection = connect(tmp_path)
    try:
        counts = aggregate_card_counts(
            connection,
            canonical_path=canonical,
            split_path=split,
            partition="train",
            seed=0,
        )
    finally:
        connection.close()

    assert counts[card_identity_key("knight", "base")] == (1, 1)
    assert counts[card_identity_key("archers", "base")] == (0, 1)


def test_aggregate_matchup_counts_uses_oriented_deck_pairs(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    split = tmp_path / "splits-temporal.parquet"
    write_battle_parquet(
        canonical,
        (("fp-keep", "a.parquet", 0, "train", WIN_IDS, LOSE_IDS, "win-hash", "lose-hash"),),
    )
    connection = connect(tmp_path)
    try:
        counts = aggregate_matchup_counts(
            connection,
            canonical_path=canonical,
            split_path=split,
            partition="train",
            seed=0,
        )
        example = next(
            iter_oriented_examples(
                connection,
                canonical_path=canonical,
                split_path=split,
                partition="train",
                seed=0,
            )
        )
    finally:
        connection.close()

    assert counts[(example.deck_a_hash, example.deck_b_hash)] == (example.label, 1)


def test_train_io_rejects_empty_partition(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    split = tmp_path / "splits-temporal.parquet"
    write_battle_parquet(
        canonical,
        (("fp-keep", "a.parquet", 0, "validation", WIN_IDS, LOSE_IDS, "win-hash", "lose-hash"),),
    )
    connection = connect(tmp_path)
    try:
        with pytest.raises(KaggleV6TrainError, match="empty"):
            require_partition_rows(
                connection,
                canonical_path=canonical,
                split_path=split,
                split="temporal",
                partition="train",
                seed=0,
                expected_rows=1,
            )
    finally:
        connection.close()


def test_train_io_rejects_join_count_mismatch(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    split = tmp_path / "splits-temporal.parquet"
    write_battle_parquet(
        canonical,
        (("fp-keep", "a.parquet", 0, "train", WIN_IDS, LOSE_IDS, "win-hash", "lose-hash"),),
    )
    connection = connect(tmp_path)
    try:
        with pytest.raises(KaggleV6TrainError, match="join count 1 != 2"):
            require_partition_rows(
                connection,
                canonical_path=canonical,
                split_path=split,
                split="temporal",
                partition="train",
                seed=0,
                expected_rows=2,
            )
    finally:
        connection.close()


def test_iter_oriented_examples_swaps_player_ids_when_mirrored(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    split = tmp_path / "splits-temporal.parquet"
    write_battle_parquet(
        canonical,
        (
            (KEEP_FP, "a.parquet", 0, "train", WIN_IDS, LOSE_IDS, "win-hash", "lose-hash"),
            (SWAP_FP, "a.parquet", 1, "train", WIN_IDS, LOSE_IDS, "win-hash", "lose-hash"),
        ),
    )
    connection = connect(tmp_path)
    try:
        examples = tuple(
            iter_oriented_examples(
                connection,
                canonical_path=canonical,
                split_path=split,
                partition="train",
                seed=0,
            )
        )
    finally:
        connection.close()
    keep = next(example for example in examples if example.label == 1)
    swap = next(example for example in examples if example.label == 0)
    assert keep.side_a_player_id == "#WINNER"
    assert keep.side_b_player_id == "#LOSER"
    assert swap.side_a_player_id == "#LOSER"
    assert swap.side_b_player_id == "#WINNER"


def test_iter_oriented_examples_can_stream_in_timestamp_order(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    split = tmp_path / "splits-temporal.parquet"
    later = datetime(2026, 6, 20, tzinfo=UTC)
    earlier = datetime(2026, 6, 10, tzinfo=UTC)
    write_battle_parquet(
        canonical,
        (
            (LATE_FP, "a.parquet", 0, "train", WIN_IDS, LOSE_IDS, "late-hash", "lose-hash"),
            (EARLY_FP, "b.parquet", 1, "train", WIN_IDS, LOSE_IDS, "early-hash", "lose-hash"),
        ),
        stamps=(later, earlier),
    )
    connection = connect(tmp_path)
    try:
        examples = tuple(
            iter_oriented_examples(
                connection,
                canonical_path=canonical,
                split_path=split,
                partition="train",
                seed=0,
                order_by_time=True,
            )
        )
    finally:
        connection.close()
    assert [example.deck_a_hash for example in examples] == ["early-hash", "late-hash"]
