"""General catalogs retain the snapshot layout without June-specific cardinalities."""

from copy import deepcopy

import pytest

from clash_sos.domain.card_catalog import CardCatalog
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def test_catalog_accepts_additions_and_preserves_source_lookup() -> None:
    payload = KAGGLE_V6_CARDS.to_payload()
    entries = [
        [entry.source_id, entry.source_name, entry.card.card_id.value, entry.card.form.value]
        for entry in KAGGLE_V6_CARDS.entries
    ]
    entries.extend(
        [
            [176, "future card", "future-card", "base"],
            [177, "evo future card", "future-card", "evolution"],
        ]
    )
    payload.update(catalog_version="cards:expanded", entries=entries)
    catalog = CardCatalog.from_payload(payload)
    assert len(catalog.entries) == 178
    assert catalog.find(177) == catalog.entries[-1]
    assert catalog.find(178) is None
    assert CardCatalog.from_payload(catalog.to_payload()) == catalog


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ([177, "future card", "future-card", "base"], "source IDs"),
        ([176, "knight", "future-card", "base"], "unique"),
        ([176, "future card", "knight", "base"], "unique"),
        ([176, "future card", "future-card", "hero"], "cataloged base"),
        ([True, "future card", "future-card", "base"], "invalid card catalog entry"),
        ([176, "future card"], "invalid card catalog entry"),
    ],
)
def test_catalog_rejects_ambiguous_or_dangling_mappings(entry: list[object], message: str) -> None:
    payload = deepcopy(KAGGLE_V6_CARDS.to_payload())
    entries = [
        [item.source_id, item.source_name, item.card.card_id.value, item.card.form.value]
        for item in KAGGLE_V6_CARDS.entries
    ]
    payload["entries"] = [*entries, entry]
    with pytest.raises(ValueError, match=message):
        CardCatalog.from_payload(payload)


@pytest.mark.parametrize("payload", [{}, {"catalog_version": "", "entries": []}, []])
def test_catalog_requires_a_version_and_entries(payload: object) -> None:
    with pytest.raises(ValueError):
        CardCatalog.from_payload(payload)
