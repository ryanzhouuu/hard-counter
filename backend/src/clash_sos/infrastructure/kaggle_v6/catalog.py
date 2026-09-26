"""Versioned card catalog for the Kaggle version 6 source."""

from importlib.resources import files
from json import loads

from clash_sos.domain.card_catalog import CardCatalog as KaggleCardCatalog
from clash_sos.domain.card_catalog import CardCatalogEntry as KaggleCardCatalogEntry

__all__ = ["KAGGLE_V6_CARDS", "KaggleCardCatalog", "KaggleCardCatalogEntry"]


def _load_catalog() -> KaggleCardCatalog:
    resource = files(__package__).joinpath("cards.json")
    catalog = KaggleCardCatalog.from_payload(loads(resource.read_text(encoding="utf-8")))
    if catalog.version != "kaggle-v6-2026-06" or len(catalog.entries) != 176:
        raise ValueError("Kaggle v6 card catalog must cover unique source IDs 0 through 175")
    return catalog


KAGGLE_V6_CARDS = _load_catalog()
