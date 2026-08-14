from datetime import UTC, datetime

import pytest
from pydantic import AnyHttpUrl, ValidationError

from clash_sos.domain.manifests import (
    ArchiveManifest,
    DatasetFileManifest,
    DatasetManifest,
    DatasetObservationsManifest,
    DatasetSchemaManifest,
    DatasetValidationManifest,
    LicenseManifest,
    ModeCountManifest,
    RetrievalManifest,
    SchemaColumnManifest,
    SourceManifest,
)

SHA256 = "a" * 64


def source_manifest() -> SourceManifest:
    return SourceManifest(
        source_id="kaggle:jackmangione/clash-royale-matchups-june2026:v6",
        provider="kaggle",
        dataset_handle="jackmangione/clash-royale-matchups-june2026",
        dataset_version=6,
        dataset_url=AnyHttpUrl(
            "https://www.kaggle.com/datasets/jackmangione/clash-royale-matchups-june2026"
        ),
        license=LicenseManifest(
            identifier="CC-BY-NC-SA-4.0",
            name="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
            url=AnyHttpUrl("https://creativecommons.org/licenses/by-nc-sa/4.0/"),
            restrictions=("noncommercial_use",),
        ),
        known_limitations=("No battle identifier is supplied by the source.",),
        retrieval=RetrievalManifest(method="manual_download"),
    )


def dataset_manifest() -> DatasetManifest:
    return DatasetManifest(
        source_id=source_manifest().source_id,
        archive=ArchiveManifest(path="source.zip", size_bytes=10, sha256=SHA256),
        files=(
            DatasetFileManifest(
                path="battles.parquet",
                kind="parquet",
                size_bytes=5,
                sha256=SHA256,
            ),
        ),
        schema=DatasetSchemaManifest(
            format="parquet",
            columns=(
                SchemaColumnManifest(
                    name="timestamp",
                    physical_type="timestamp[us]",
                    nullable=False,
                ),
            ),
            fingerprint=SHA256,
        ),
        observations=DatasetObservationsManifest(
            row_count=1,
            timestamp_column="timestamp",
            timestamp_min=datetime(2026, 6, 21, tzinfo=UTC),
            timestamp_max=datetime(2026, 6, 21, tzinfo=UTC),
            modes=("Ranked1v1_NewArena",),
            mode_counts=(ModeCountManifest(mode="Ranked1v1_NewArena", row_count=1),),
            card_id_min=0,
            card_id_max=175,
        ),
        validation=DatasetValidationManifest(status="not_run"),
    )


def test_source_manifest_identifies_pinned_kaggle_source() -> None:
    manifest = source_manifest()

    assert manifest.manifest_type == "source"
    assert manifest.dataset_version == 6
    assert manifest.source_id.endswith(":v6")


def test_dataset_manifest_round_trips_through_json() -> None:
    manifest = dataset_manifest()

    serialized = manifest.model_dump_json(by_alias=True)
    restored = DatasetManifest.model_validate_json(serialized)

    assert restored == manifest
    assert '"schema"' in serialized


@pytest.mark.parametrize(
    "path",
    ["/absolute.parquet", "../escape.parquet", "nested/../file.parquet"],
)
def test_dataset_file_rejects_non_canonical_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="file path must"):
        DatasetFileManifest(path=path, kind="parquet", size_bytes=1, sha256=SHA256)


def test_manifest_rejects_invalid_checksum() -> None:
    with pytest.raises(ValidationError, match="String should match pattern"):
        ArchiveManifest(path="source.zip", size_bytes=10, sha256="not-a-checksum")


def test_observations_reject_reversed_ranges() -> None:
    with pytest.raises(ValidationError, match="timestamp_min"):
        DatasetObservationsManifest(
            timestamp_min=datetime(2026, 6, 22, tzinfo=UTC),
            timestamp_max=datetime(2026, 6, 21, tzinfo=UTC),
        )


def test_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_field"):
        payload = source_manifest().model_dump()
        payload["extra_field"] = "unexpected"
        SourceManifest.model_validate(payload)


def test_schema_physical_type_semantics_are_documented() -> None:
    description = SchemaColumnManifest.model_json_schema()["properties"]["physical_type"]

    assert description["description"] == (
        "DuckDB DESCRIBE SQL type for dataset manifest version 1."
    )


def test_manifest_requires_sorted_unique_file_inventory() -> None:
    manifest = dataset_manifest()
    duplicate = manifest.files[0].model_copy(update={"row_count": 1})

    with pytest.raises(ValidationError, match="unique, sorted"):
        DatasetManifest.model_validate(
            {
                **manifest.model_dump(by_alias=True),
                "files": [duplicate.model_dump(), duplicate.model_dump()],
            }
        )


def test_manifest_reconciles_file_and_observation_rows() -> None:
    manifest = dataset_manifest()
    counted_file = manifest.files[0].model_copy(update={"row_count": 2})

    with pytest.raises(ValidationError, match="reconcile"):
        manifest.model_copy(update={"files": (counted_file,)}).model_validate(
            {**manifest.model_dump(by_alias=True), "files": [counted_file.model_dump()]}
        )


def test_modes_are_sorted_unique_and_reconcile_with_rows() -> None:
    with pytest.raises(ValidationError, match="unique, sorted"):
        DatasetObservationsManifest(
            row_count=2,
            modes=("Ranked", "Ladder"),
            mode_counts=(
                ModeCountManifest(mode="Ranked", row_count=1),
                ModeCountManifest(mode="Ladder", row_count=1),
            ),
        )

    with pytest.raises(ValidationError, match="sum to row_count"):
        DatasetObservationsManifest(
            row_count=2,
            modes=("Ranked",),
            mode_counts=(ModeCountManifest(mode="Ranked", row_count=1),),
        )


def test_validation_counts_are_all_or_none_and_reconcile() -> None:
    with pytest.raises(ValidationError, match="all present or all absent"):
        DatasetValidationManifest(status="passed", accepted_rows=1)

    manifest = dataset_manifest()
    with pytest.raises(ValidationError, match="reconcile"):
        DatasetManifest.model_validate(
            {
                **manifest.model_dump(by_alias=True),
                "validation": {
                    "status": "passed",
                    "accepted_rows": 0,
                    "rejected_rows": 0,
                    "quarantined_rows": 0,
                },
            }
        )


def test_manifest_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        DatasetObservationsManifest(timestamp_min=datetime(2026, 6, 21))

    with pytest.raises(ValidationError, match="timezone-aware"):
        RetrievalManifest(method="manual_download", retrieved_at=datetime(2026, 6, 21))
