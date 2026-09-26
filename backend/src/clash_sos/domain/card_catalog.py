"""Validate catalog snapshots independently of their source and model token layout."""

from dataclasses import dataclass
from functools import cached_property
from json import dumps
from typing import cast

from clash_sos.domain.canonical import CardForm, CardId, CardRef


@dataclass(frozen=True)
class CardCatalogEntry:
    source_id: int
    source_name: str
    card: CardRef


@dataclass(frozen=True)
class CardCatalog:
    version: str
    entries: tuple[CardCatalogEntry, ...]

    def __post_init__(self) -> None:
        if not self.version.strip() or not self.entries:
            raise ValueError("catalog version and entries are required")
        if any(type(entry.source_id) is not int for entry in self.entries) or [
            entry.source_id for entry in self.entries
        ] != list(range(len(self.entries))):
            raise ValueError("catalog source IDs must be ordered from zero")
        names = [entry.source_name for entry in self.entries]
        identities = [entry.card.identity_key for entry in self.entries]
        if (
            any(not name.strip() for name in names)
            or len(set(names)) != len(names)
            or len(set(identities)) != len(identities)
        ):
            raise ValueError("catalog names and card identities must be unique and nonempty")
        roots = {
            entry.card.card_id.value
            for entry in self.entries
            if entry.card.form in {CardForm.BASE, CardForm.CHAMPION}
        }
        if any(entry.card.card_id.value not in roots for entry in self.entries):
            raise ValueError("catalog forms must reference a cataloged base or champion card")

    @cached_property
    def by_source_id(self) -> dict[int, CardCatalogEntry]:
        return {entry.source_id: entry for entry in self.entries}

    def find(self, source_id: int) -> CardCatalogEntry | None:
        return self.by_source_id.get(source_id)

    def to_payload(self) -> dict[str, object]:
        return {
            "catalog_version": self.version,
            "entries": [
                [
                    entry.source_id,
                    entry.source_name,
                    entry.card.card_id.value,
                    entry.card.form.value,
                ]
                for entry in self.entries
            ],
        }

    def serialize(self) -> bytes:
        return dumps(self.to_payload(), ensure_ascii=True, separators=(",", ":")).encode()

    @classmethod
    def from_payload(cls, payload: object) -> "CardCatalog":
        if not isinstance(payload, dict):
            raise ValueError("card catalog snapshot must be an object")
        body = cast(dict[str, object], payload)
        version, raw_entries = body.get("catalog_version"), body.get("entries")
        if not isinstance(version, str) or not isinstance(raw_entries, list):
            raise ValueError("catalog version and entries are required")
        entries: list[CardCatalogEntry] = []
        for raw in cast(list[object], raw_entries):
            if not isinstance(raw, list):
                raise ValueError("invalid card catalog entry")
            values = cast(list[object], raw)
            if len(values) != 4:
                raise ValueError("invalid card catalog entry")
            source_id, name, card_id, form = values
            if (
                type(source_id) is not int
                or not isinstance(name, str)
                or not isinstance(card_id, str)
                or not isinstance(form, str)
            ):
                raise ValueError("invalid card catalog entry")
            entries.append(
                CardCatalogEntry(
                    source_id, name, CardRef(card_id=CardId(card_id), form=CardForm(form))
                )
            )
        return cls(version, tuple(entries))
