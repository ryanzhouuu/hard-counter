"""Normalized official battles and immutable tower-aware snapshot inventories."""

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from clash_sos.domain.attention_cache import CacheFile
from clash_sos.domain.attention_protocol import Partition
from clash_sos.domain.canonical_dataset import CanonicalBattleRow
from clash_sos.domain.manifests import ManifestModel


class TowerBattleRow(CanonicalBattleRow):
    side_a_tower: str = Field(pattern=r"^[a-z0-9-]+:tower$")
    side_b_tower: str = Field(pattern=r"^[a-z0-9-]+:tower$")
    side_a_tower_level: Literal[16]
    side_b_tower_level: Literal[16]

    @model_validator(mode="after")
    def require_official_source(self) -> Self:
        if self.source_id != "official-api":
            raise ValueError("tower battles require the official-api source")
        return self


class AttentionDatasetManifest(ManifestModel):
    manifest_type: Literal["attention_dataset"] = "attention_dataset"
    canonical_schema_version: Literal["official-ranked16-schema:v1"] = "official-ranked16-schema:v1"
    dataset_version: str = Field(min_length=1)
    balance_era_id: str = Field(min_length=1)
    catalog_version: str = Field(min_length=1)
    tower_catalog_version: str = Field(min_length=1)
    row_count: int = Field(gt=0)
    partition_counts: dict[Partition, int]
    start: datetime
    train_end: datetime
    validation_end: datetime
    end: datetime
    files: tuple[CacheFile, ...]

    @field_validator("start", "train_end", "validation_end", "end")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot bounds must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def check_inventory(self) -> Self:
        if not self.start < self.train_end < self.validation_end < self.end:
            raise ValueError("snapshot bounds must increase strictly")
        if set(self.partition_counts) != {"train", "validation", "test"} or any(
            count < 1 for count in self.partition_counts.values()
        ):
            raise ValueError("snapshot partitions must all be nonempty")
        if sum(self.partition_counts.values()) != self.row_count:
            raise ValueError("snapshot partition counts must sum to row count")
        if len(self.files) != 2 or {item.path for item in self.files} != {
            "canonical.parquet",
            "splits-temporal.parquet",
        }:
            raise ValueError("snapshot inventory requires canonical and temporal split files")
        return self
