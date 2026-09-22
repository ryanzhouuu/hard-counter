"""Five-row processed dataset for cache I/O and provenance tests."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import duckdb
from test_processed_manifest import processed_manifest

from clash_sos.application.attention_protocol_resolve import resolve_attention_slice
from clash_sos.domain.attention_protocol import AttentionProtocol, Partition
from clash_sos.domain.attention_schema import (
    AttentionCardSchema,
    AttentionModelConfig,
    build_attention_schema,
)
from clash_sos.domain.canonical_dataset import deck_content_hash
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.domain.matchup_baseline import should_mirror_sides
from clash_sos.domain.processed_manifest import (
    ProcessedDatasetManifest,
    ProcessedOutputFile,
    dump_processed_manifest,
)
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS

WIN = ("knight", "mini-pekka", "musketeer", "valkyrie", "hog", "fireball", "log", "cannon")
LOSE = ("archers", "goblins", "bomber", "skeletons", "tombstone", "zap", "arrows", "tesla")
FORMS = ("base",) * 8
LEVELS = (16,) * 8


def stamp(day: int) -> datetime:
    """Keep fixture timestamps inside the processed manifest's published bounds."""
    return datetime(2026, 6, day, tzinfo=UTC)


def fingerprint(*, mirrored: bool) -> str:
    """Find a stable source fingerprint for either legacy mirror outcome."""
    return next(
        value
        for index in range(256)
        if should_mirror_sides(value := f"cache-fp-{index}") is mirrored
    )


def write_cache_dataset(path: Path) -> tuple[AttentionCardSchema, AttentionProtocol]:
    """Publish minimal joined Parquet with true source hashes and resolved slices."""
    path.mkdir()
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    connection.execute(
        """
        CREATE TABLE battles (
            timestamp TIMESTAMPTZ, fingerprint VARCHAR, archive_member VARCHAR,
            row_number BIGINT, partition VARCHAR, side_a_card_ids VARCHAR[],
            side_a_card_forms VARCHAR[], side_a_card_levels UTINYINT[],
            side_b_card_ids VARCHAR[], side_b_card_forms VARCHAR[],
            side_b_card_levels UTINYINT[], side_a_deck_hash VARCHAR,
            side_b_deck_hash VARCHAR, side_a_player_id VARCHAR,
            side_b_player_id VARCHAR
        )
        """
    )
    rows = (
        (2, "train", fingerprint(mirrored=False)),
        (9, "train", fingerprint(mirrored=True)),
        (10, "validation", "val-keep"),
        (19, "validation", "val-swap"),
        (25, "test", "test-keep"),
    )
    for index, (day, partition, fp) in enumerate(rows):
        connection.execute(
            "INSERT INTO battles VALUES (?, ?, 'a.parquet', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                stamp(day),
                fp,
                index,
                partition,
                WIN,
                FORMS,
                LEVELS,
                LOSE,
                FORMS,
                LEVELS,
                deck_content_hash(WIN, FORMS, LEVELS),
                deck_content_hash(LOSE, FORMS, LEVELS),
                f"#WIN{index}",
                f"#LOSE{index}",
            ],
        )
    connection.execute(
        f"COPY (SELECT * EXCLUDE partition FROM battles) TO '{path / 'canonical.parquet'}' "
        "(FORMAT PARQUET)"
    )
    for name in ("splits-temporal.parquet", "splits-player-disjoint.parquet"):
        filter_clause = "WHERE row_number <> 3" if name == "splits-player-disjoint.parquet" else ""
        connection.execute(
            f"COPY (SELECT timestamp, fingerprint, archive_member, row_number, partition "
            f"FROM battles {filter_clause}) TO '{path / name}' (FORMAT PARQUET)"
        )
    base = processed_manifest()
    files: list[ProcessedOutputFile] = []
    for item in base.files:
        if item.path in {
            "canonical.parquet",
            "splits-temporal.parquet",
            "splits-player-disjoint.parquet",
        }:
            size, digest = hash_file(path / item.path, 1024)
            files.append(item.model_copy(update={"size_bytes": size, "sha256": digest}))
        else:
            files.append(item)
    manifest = ProcessedDatasetManifest.model_validate(
        {**base.model_dump(mode="python"), "files": tuple(files)}
    )
    manifest_path = path / "manifest.json"
    manifest_path.write_bytes(dump_processed_manifest(manifest))
    schema = build_attention_schema(
        KAGGLE_V6_CARDS.serialize(), attributes=CARD_ATTRIBUTES, network=AttentionModelConfig()
    )
    canonical = path / "canonical.parquet"
    split = path / "splits-temporal.parquet"

    def resolved(partition: Partition, first: int, last: int):
        """Use the real join to populate the protocol's count and row digest."""
        return resolve_attention_slice(
            connection,
            canonical_path=canonical,
            split_path=split,
            partition=partition,
            start=stamp(first),
            end=stamp(last),
        )

    protocol = AttentionProtocol(
        family="temporal",
        dataset_version=manifest.dataset_version,
        balance_era_id="2026-06",
        processed_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        canonical_sha256=next(x.sha256 for x in files if x.kind == "canonical"),
        split_file="splits-temporal.parquet",
        split_sha256=next(x.sha256 for x in files if x.kind == "temporal_split"),
        encoding_sha256=schema.fingerprint(),
        mirror_seed=0,
        selection_fit=resolved("train", 2, 9),
        watch=resolved("train", 9, 10),
        refit=resolved("train", 2, 10),
        development=resolved("validation", 10, 20),
        reporting=resolved("test", 20, 26),
    )
    connection.close()
    return schema, protocol
