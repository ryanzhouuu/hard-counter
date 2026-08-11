"""Validated provenance contracts for external datasets and local snapshots."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, StringConstraints, model_validator

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
    def validate_path(self) -> "DatasetFileManifest":
        if self.path.startswith("/") or "\\" in self.path:
            raise ValueError("file path must be a relative POSIX path")
        if any(part in {"", ".", ".."} for part in self.path.split("/")):
            raise ValueError("file path must not contain empty, dot, or parent components")
        return self


class SchemaColumnManifest(ManifestModel):
    name: str = Field(min_length=1)
    physical_type: str = Field(min_length=1)
    nullable: bool


class DatasetSchemaManifest(ManifestModel):
    format: Literal["parquet"]
    columns: tuple[SchemaColumnManifest, ...]
    fingerprint: Sha256


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
    def validate_time_range(self) -> "DatasetObservationsManifest":
        if self.timestamp_min and self.timestamp_max and self.timestamp_min > self.timestamp_max:
            raise ValueError("timestamp_min must not be later than timestamp_max")
        if (
            self.card_id_min is not None
            and self.card_id_max is not None
            and self.card_id_min > self.card_id_max
        ):
            raise ValueError("card_id_min must not be greater than card_id_max")
        return self


class DatasetValidationManifest(ManifestModel):
    status: Literal["not_run", "passed", "failed"]
    accepted_rows: int | None = Field(default=None, ge=0)
    rejected_rows: int | None = Field(default=None, ge=0)
    quarantined_rows: int | None = Field(default=None, ge=0)
    notes: tuple[str, ...] = ()


class ArchiveManifest(ManifestModel):
    path: RelativePath
    size_bytes: int = Field(ge=0)
    sha256: Sha256


class DatasetManifest(ManifestModel):
    manifest_type: Literal["dataset"] = "dataset"
    manifest_version: Literal[1] = 1
    source_id: str = Field(min_length=1)
    archive: ArchiveManifest
    files: tuple[DatasetFileManifest, ...]
    dataset_schema: DatasetSchemaManifest | None = Field(default=None, alias="schema")
    observations: DatasetObservationsManifest = DatasetObservationsManifest()
    validation: DatasetValidationManifest
