"""Validated provenance contracts for external datasets and local snapshots."""

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RelativePath = Annotated[str, StringConstraints(min_length=1)]


class ManifestModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class LicenseManifest(ManifestModel):
    identifier: str = Field(min_length=1)
    name: str = Field(min_length=1)
    url: AnyHttpUrl
    restrictions: tuple[str, ...] = ()


class RetrievalManifest(ManifestModel):
    method: Literal["manual_download", "kagglehub", "kaggle_cli", "api"]
    retrieved_at: datetime | None = None
    notes: str | None = None

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("retrieved_at must be timezone-aware")
        return value


class SourceManifest(ManifestModel):
    manifest_type: Literal["source"] = "source"
    manifest_version: Literal[1] = 1
    source_id: str = Field(min_length=1)
    provider: Literal["kaggle"]
    dataset_handle: str = Field(min_length=1)
    dataset_version: int = Field(gt=0)
    dataset_url: AnyHttpUrl
    license: LicenseManifest
    known_limitations: tuple[str, ...] = ()
    retrieval: RetrievalManifest


class DatasetFileManifest(ManifestModel):
    path: RelativePath
    kind: Literal["parquet", "card_mapping", "other"]
    size_bytes: int = Field(ge=0)
    sha256: Sha256
    row_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        _validate_relative_path(self.path)
        return self


class SchemaColumnManifest(ManifestModel):
    name: str = Field(min_length=1)
    physical_type: str = Field(
        min_length=1,
        description="DuckDB DESCRIBE SQL type for dataset manifest version 1.",
    )
    nullable: bool


class DatasetSchemaManifest(ManifestModel):
    format: Literal["parquet"]
    columns: tuple[SchemaColumnManifest, ...]
    fingerprint: Sha256

    @model_validator(mode="after")
    def validate_columns(self) -> Self:
        names = [column.name for column in self.columns]
        if not names or len(names) != len(set(names)):
            raise ValueError("schema columns must be non-empty and unique")
        return self


class ModeCountManifest(ManifestModel):
    mode: str = Field(min_length=1)
    row_count: int = Field(ge=0)


class DatasetObservationsManifest(ManifestModel):
    row_count: int | None = Field(default=None, ge=0)
    timestamp_column: str | None = None
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None
    modes: tuple[str, ...] = ()
    mode_counts: tuple[ModeCountManifest, ...] = ()
    card_id_min: int | None = None
    card_id_max: int | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        for timestamp in (self.timestamp_min, self.timestamp_max):
            if timestamp is not None and (
                timestamp.tzinfo is None or timestamp.utcoffset() is None
            ):
                raise ValueError("observation timestamps must be timezone-aware")
        if self.timestamp_min and self.timestamp_max and self.timestamp_min > self.timestamp_max:
            raise ValueError("timestamp_min must not be later than timestamp_max")
        if (
            self.card_id_min is not None
            and self.card_id_max is not None
            and self.card_id_min > self.card_id_max
        ):
            raise ValueError("card_id_min must not be greater than card_id_max")
        expected_modes = tuple(count.mode for count in self.mode_counts)
        if expected_modes != tuple(sorted(set(expected_modes))):
            raise ValueError("mode counts must have unique, sorted modes")
        if self.modes != expected_modes:
            raise ValueError("modes must exactly match mode_counts")
        if self.row_count is not None and sum(count.row_count for count in self.mode_counts) != (
            self.row_count
        ):
            raise ValueError("mode counts must sum to row_count")
        return self


class DatasetValidationManifest(ManifestModel):
    status: Literal["not_run", "passed", "failed"]
    accepted_rows: int | None = Field(default=None, ge=0)
    rejected_rows: int | None = Field(default=None, ge=0)
    quarantined_rows: int | None = Field(default=None, ge=0)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        counts = (self.accepted_rows, self.rejected_rows, self.quarantined_rows)
        if self.status == "not_run" and any(count is not None for count in counts):
            raise ValueError("not_run validation cannot include row counts")
        if any(count is None for count in counts) and any(count is not None for count in counts):
            raise ValueError("validation row counts must be all present or all absent")
        return self


class ArchiveManifest(ManifestModel):
    path: RelativePath
    size_bytes: int = Field(ge=0)
    sha256: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        _validate_relative_path(self.path)
        return self


class DatasetManifest(ManifestModel):
    manifest_type: Literal["dataset"] = "dataset"
    manifest_version: Literal[1] = 1
    source_id: str = Field(min_length=1)
    archive: ArchiveManifest
    files: tuple[DatasetFileManifest, ...]
    dataset_schema: DatasetSchemaManifest | None = Field(default=None, alias="schema")
    observations: DatasetObservationsManifest = DatasetObservationsManifest()
    validation: DatasetValidationManifest

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        paths = tuple(file.path for file in self.files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ValueError("dataset files must have unique, sorted paths")
        parquet_counts = [file.row_count for file in self.files if file.kind == "parquet"]
        if (
            parquet_counts
            and all(count is not None for count in parquet_counts)
            and self.observations.row_count != sum(count or 0 for count in parquet_counts)
        ):
            raise ValueError("Parquet row counts must reconcile with observations")
        validation_counts = (
            self.validation.accepted_rows,
            self.validation.rejected_rows,
            self.validation.quarantined_rows,
        )
        if all(count is not None for count in validation_counts) and (
            self.observations.row_count
            != sum(count for count in validation_counts if count is not None)
        ):
            raise ValueError("validation row counts must reconcile with observations")
        return self


def validate_relative_path(path: str) -> None:
    if path.startswith("/") or "\\" in path:
        raise ValueError("file path must be a relative POSIX path")
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("file path must not contain empty, dot, or parent components")


_validate_relative_path = validate_relative_path
