"""Tower identities and API level offsets, independent of deployable card attributes."""

from functools import cached_property
from typing import Literal, Self

from pydantic import Field, model_validator

from clash_sos.domain.canonical import CardId
from clash_sos.domain.manifests import ManifestModel


class TowerEntry(ManifestModel):
    tower_id: CardId
    name: str = Field(min_length=1)
    api_id: int = Field(gt=0, strict=True)
    max_api_level: Literal[8, 11, 16]

    @property
    def identity(self) -> str:
        return f"{self.tower_id.value}:tower"

    def normalize_level(self, level: int) -> int:
        """Convert rarity-relative API levels to the common 1-16 scale."""
        if type(level) is not int or not 1 <= level <= self.max_api_level:
            raise ValueError("invalid tower level")
        return level + 16 - self.max_api_level


class TowerCatalog(ManifestModel):
    catalog_version: str = Field(min_length=1)
    entries: tuple[TowerEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        """Require unambiguous API IDs, normalized names, and stable identities."""
        for values in (
            [entry.api_id for entry in self.entries],
            [" ".join(entry.name.casefold().split()) for entry in self.entries],
            [entry.identity for entry in self.entries],
        ):
            if len(set(values)) != len(values) or "" in values:
                raise ValueError("tower catalog entries must be unique and nonempty")
        return self

    @cached_property
    def by_api_id(self) -> dict[int, TowerEntry]:
        return {entry.api_id: entry for entry in self.entries}

    @cached_property
    def by_name(self) -> dict[str, TowerEntry]:
        return {" ".join(entry.name.casefold().split()): entry for entry in self.entries}
