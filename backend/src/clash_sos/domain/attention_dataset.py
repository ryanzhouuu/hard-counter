"""Normalized official battles and immutable tower-aware snapshot inventories."""

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from clash_sos.domain.attention_cache import CacheFile
from clash_sos.domain.attention_protocol import Partition
from clash_sos.domain.canonical_dataset import CanonicalBattleRow
from clash_sos.domain.manifests import ManifestModel

OfficialSchemaVersion = Literal[
    "official-ranked16-schema:v1",
    "official-ranked16-schema:v2",
    "official-ranked16-schema:v3",
]


def official_ranked_modes(version: str) -> tuple[str, ...]:
    """Keep frozen single-mode contracts distinct from the mixed ranked contract."""
    modes = {
        "official-ranked16-schema:v1": ("Ranked1v1_NewArena",),
        "official-ranked16-schema:v2": ("Ranked1v1_NewArena2",),
        "official-ranked16-schema:v3": ("Ranked1v1_NewArena", "Ranked1v1_NewArena2"),
    }
    if version not in modes:
        raise ValueError("snapshot requires an official schema version")
    return modes[version]


class TowerBattleRow(CanonicalBattleRow):
    """Preserve the original official ranked mode in published v1 snapshots."""

    side_a_tower: str = Field(pattern=r"^[a-z0-9-]+:tower$")
    side_b_tower: str = Field(pattern=r"^[a-z0-9-]+:tower$")
    side_a_tower_level: Literal[16]
    side_b_tower_level: Literal[16]

    @model_validator(mode="after")
    def require_official_source(self) -> Self:
        """Reject non-official rows and current-era ranked modes in v1."""
        if self.source_id != "official-api" or self.mode != "Ranked1v1_NewArena":
            raise ValueError("v1 tower battles require the original official ranked mode")
        return self


class TowerBattleRowV2(CanonicalBattleRow):
    """Keep the current official ranked mode rather than relabeling it as June ranked."""

    side_a_tower: str = Field(pattern=r"^[a-z0-9-]+:tower$")
    side_b_tower: str = Field(pattern=r"^[a-z0-9-]+:tower$")
    side_a_tower_level: Literal[16]
    side_b_tower_level: Literal[16]

    @model_validator(mode="after")
    def require_current_official_source(self) -> Self:
        """Reject relabeled June modes in the new official snapshot contract."""
        if self.source_id != "official-api" or self.mode != "Ranked1v1_NewArena2":
            raise ValueError("v2 tower battles require the current official ranked mode")
        return self


class TowerBattleRowV3(CanonicalBattleRow):
    """Preserve either ranked API name without using it to infer the balance era."""

    side_a_tower: str = Field(pattern=r"^[a-z0-9-]+:tower$")
    side_b_tower: str = Field(pattern=r"^[a-z0-9-]+:tower$")
    side_a_tower_level: Literal[16]
    side_b_tower_level: Literal[16]

    @model_validator(mode="after")
    def require_official_ranked_source(self) -> Self:
        if self.source_id != "official-api" or self.mode not in official_ranked_modes(
            "official-ranked16-schema:v3"
        ):
            raise ValueError("v3 tower battles require an official ranked mode")
        return self


def official_row_type(
    version: str,
) -> type[TowerBattleRow] | type[TowerBattleRowV2] | type[TowerBattleRowV3]:
    official_ranked_modes(version)
    return {
        "official-ranked16-schema:v1": TowerBattleRow,
        "official-ranked16-schema:v2": TowerBattleRowV2,
        "official-ranked16-schema:v3": TowerBattleRowV3,
    }[version]


class AttentionDatasetManifest(ManifestModel):
    manifest_type: Literal["attention_dataset"] = "attention_dataset"
    canonical_schema_version: OfficialSchemaVersion = "official-ranked16-schema:v1"
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
