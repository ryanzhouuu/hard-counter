"""Versioned card catalog for the Kaggle version 6 source."""

from dataclasses import dataclass
from functools import cached_property
from importlib.resources import files
from json import dumps, loads
from typing import Any

from clash_sos.domain.canonical import CardForm, CardId, CardRef


@dataclass(frozen=True)
class KaggleCardCatalogEntry:
    source_id: int
    source_name: str
    card: CardRef


@dataclass(frozen=True)
class KaggleCardCatalog:
    version: str
    entries: tuple[KaggleCardCatalogEntry, ...]

    @cached_property
    def by_source_id(self) -> dict[int, KaggleCardCatalogEntry]:
        return {entry.source_id: entry for entry in self.entries}

    def find(self, source_id: int) -> KaggleCardCatalogEntry | None:
        return self.by_source_id.get(source_id)

    def serialize(self) -> bytes:
        payload = {
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
        return dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()


def _load_catalog() -> KaggleCardCatalog:
    resource = files(__package__).joinpath("cards.json")
    payload: dict[str, Any] = loads(resource.read_text(encoding="utf-8"))
    entries = tuple(
        KaggleCardCatalogEntry(
            source_id=source_id,
            source_name=source_name,
            card=CardRef(card_id=CardId(card_id), form=CardForm(form)),
        )
        for source_id, source_name, card_id, form in payload["entries"]
    )
    catalog = KaggleCardCatalog(version=payload["catalog_version"], entries=entries)
    source_ids = [entry.source_id for entry in entries]
    if source_ids != list(range(176)) or len({entry.source_name for entry in entries}) != 176:
        raise ValueError("Kaggle v6 card catalog must cover unique source IDs 0 through 175")
    base_cards = {
        entry.card.card_id.value
        for entry in entries
        if entry.card.form in {CardForm.BASE, CardForm.CHAMPION}
    }
    if any(
        entry.card.form in {CardForm.EVOLUTION, CardForm.HERO}
        and entry.card.card_id.value not in base_cards
        for entry in entries
    ):
        raise ValueError("Kaggle v6 forms must reference a cataloged base card")
    return catalog


KAGGLE_V6_CARDS = _load_catalog()
