"""Load selected card inputs while retaining the packaged June defaults."""

from json import loads
from pathlib import Path

from clash_sos.domain.card_attributes import CARD_ATTRIBUTES, CardAttributeTable
from clash_sos.domain.card_catalog import CardCatalog
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_CARD_CATALOG
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS

DEFAULT_CARD_CATALOG = KAGGLE_V6_CARDS


class CardInputError(ValueError):
    """A selected catalog or attribute file cannot be used."""


def _read_payload(path: Path) -> object:
    try:
        return loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise CardInputError(f"cannot read card input: {path}") from error


def load_card_catalog(path: Path | None = None) -> CardCatalog:
    if path is None:
        return DEFAULT_CARD_CATALOG
    try:
        return CardCatalog.from_payload(_read_payload(path))
    except ValueError as error:
        raise CardInputError(f"invalid card catalog: {path}") from error


def load_live_card_catalog(path: Path | None = None) -> CardCatalog:
    """Use released mappings for live lookup while training retains June defaults."""
    return CURRENT_CARD_CATALOG if path is None else load_card_catalog(path)


def load_card_attributes(path: Path | None = None) -> CardAttributeTable:
    if path is None:
        return CARD_ATTRIBUTES
    try:
        return CardAttributeTable.from_payload(_read_payload(path))
    except ValueError as error:
        raise CardInputError(f"invalid card attributes: {path}") from error
