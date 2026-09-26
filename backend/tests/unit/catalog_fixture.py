"""Synthetic catalog additions exercise extensibility without claiming API mappings."""

from clash_sos.domain.card_attributes import CARD_ATTRIBUTES, CardAttribute, CardAttributeTable
from clash_sos.domain.card_catalog import CardCatalog
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def expanded_catalog(additions: int = 90) -> tuple[CardCatalog, CardAttributeTable]:
    entries = [
        [entry.source_id, entry.source_name, entry.card.card_id.value, entry.card.form.value]
        for entry in KAGGLE_V6_CARDS.entries
    ]
    cards = dict(CARD_ATTRIBUTES.cards)
    for index in range(additions):
        card_id = f"z-future-{index:03}"
        entries.append([len(entries), f"future {index}", card_id, "base"])
        cards[card_id] = CardAttribute(3, frozenset())
    catalog = CardCatalog.from_payload({"catalog_version": "catalog:expanded", "entries": entries})
    return catalog, CardAttributeTable("attributes:expanded", cards)
