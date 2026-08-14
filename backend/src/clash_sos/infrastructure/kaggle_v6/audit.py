"""Bounded-memory inventory and profiling for the Kaggle version 6 archive."""

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZipFile

import duckdb

from clash_sos.domain.manifests import (
    ArchiveManifest,
    DatasetFileManifest,
    DatasetManifest,
    DatasetSchemaManifest,
    DatasetValidationManifest,
    SchemaColumnManifest,
)
from clash_sos.infrastructure.kaggle_v6.audit_io import (
    KaggleV6AuditError,
    extracted_member,
    hash_file,
    validate_archive_members,
    validate_card_mapping,
)
from clash_sos.infrastructure.kaggle_v6.audit_profile import (
    ParquetProfile,
    combine_profiles,
    profile_parquet,
    read_parquet_schema,
)
from clash_sos.infrastructure.kaggle_v6.schema import (
    KAGGLE_V6_SCHEMA,
    IncompatibleKaggleSchemaError,
    validate_kaggle_v6_schema,
)
from clash_sos.infrastructure.kaggle_v6.source import (
    KAGGLE_V6_ARCHIVE_SHA256,
    KAGGLE_V6_ARCHIVE_SIZE,
    KAGGLE_V6_CARD_MAPPING_NAME,
    KAGGLE_V6_SOURCE_ID,
)


def audit_kaggle_v6_archive(
    archive_path: Path,
    *,
    temp_directory: Path,
    memory_limit: str = "1GB",
    threads: int = 2,
    chunk_size: int = 8 * 1024 * 1024,
    expected_archive_size: int | None = KAGGLE_V6_ARCHIVE_SIZE,
    expected_archive_sha256: str | None = KAGGLE_V6_ARCHIVE_SHA256,
) -> DatasetManifest:
    if threads < 1 or chunk_size < 1:
        raise ValueError("threads and chunk_size must be positive")
    temp_directory.mkdir(parents=True, exist_ok=True)
    archive_size, archive_hash = hash_file(archive_path, chunk_size)
    if expected_archive_size is not None and archive_size != expected_archive_size:
        raise KaggleV6AuditError("archive size does not match the pinned Kaggle v6 source")
    if expected_archive_sha256 is not None and archive_hash != expected_archive_sha256:
        raise KaggleV6AuditError("archive checksum does not match the pinned Kaggle v6 source")
    files: list[DatasetFileManifest] = []
    profiles: list[ParquetProfile] = []
    schema_fingerprint: str | None = None

    with (
        ZipFile(archive_path) as archive,
        TemporaryDirectory(prefix="clash-sos-audit-", dir=temp_directory) as work_directory,
    ):
        members = validate_archive_members(archive)
        parquet_members = [member for member in members if member.filename.endswith(".parquet")]
        if len(parquet_members) != 23:
            raise KaggleV6AuditError("Kaggle v6 archive must contain exactly 23 Parquet files")

        connection = duckdb.connect(
            config={
                "memory_limit": memory_limit,
                "threads": str(threads),
                "temp_directory": work_directory,
            }
        )
        try:
            connection.execute("SET TimeZone='UTC'")
            for member in members:
                with extracted_member(
                    archive, member, Path(work_directory), chunk_size
                ) as extraction:
                    extracted, member_hash = extraction
                    if member.filename == KAGGLE_V6_CARD_MAPPING_NAME:
                        validate_card_mapping(extracted)
                        kind = "card_mapping"
                        row_count = None
                    else:
                        kind = "parquet"
                        observed_schema = read_parquet_schema(connection, extracted)
                        try:
                            observed_fingerprint = validate_kaggle_v6_schema(observed_schema)
                        except IncompatibleKaggleSchemaError as error:
                            raise KaggleV6AuditError(
                                f"incompatible schema in {member.filename}"
                            ) from error
                        if schema_fingerprint not in {None, observed_fingerprint}:
                            raise KaggleV6AuditError("Parquet members have inconsistent schemas")
                        schema_fingerprint = observed_fingerprint
                        profile = profile_parquet(connection, extracted)
                        profiles.append(profile)
                        row_count = profile.row_count
                    files.append(
                        DatasetFileManifest(
                            path=member.filename,
                            kind=kind,
                            size_bytes=member.file_size,
                            sha256=member_hash,
                            row_count=row_count,
                        )
                    )
        finally:
            connection.close()

    if schema_fingerprint is None:
        raise KaggleV6AuditError("Kaggle v6 archive has no Parquet schema")
    observations = combine_profiles(profiles)
    return DatasetManifest(
        source_id=KAGGLE_V6_SOURCE_ID,
        archive=ArchiveManifest(
            path=archive_path.name,
            size_bytes=archive_size,
            sha256=archive_hash,
        ),
        files=tuple(files),
        schema=DatasetSchemaManifest(
            format="parquet",
            columns=tuple(
                SchemaColumnManifest(
                    name=column.name,
                    physical_type=column.physical_type,
                    nullable=column.nullable,
                )
                for column in KAGGLE_V6_SCHEMA
            ),
            fingerprint=schema_fingerprint,
        ),
        observations=observations,
        validation=DatasetValidationManifest(
            status="passed",
            notes=(
                "Archive, member checksums, card mapping, and Parquet schemas passed.",
                "R1 does not count row dispositions; unsupported rows are not rejected rows.",
            ),
        ),
    )


def write_dataset_manifest(manifest: DatasetManifest, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{manifest.model_dump_json(by_alias=True, indent=2)}\n".encode()
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=output_path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
