"""Pin root-card API IDs and rarity levels independently of model tokens."""

import pytest

from clash_sos.infrastructure.clash_royale.card_api_catalog import (
    CURRENT_CARD_API_CATALOG,
    CardAPIEntry,
)
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_CARD_CATALOG


def test_api_mapping_covers_current_root_cards() -> None:
    roots = {
        entry.card.card_id.value
        for entry in CURRENT_CARD_CATALOG.entries
        if entry.card.form.value in {"base", "champion"}
    }
    assert set(CURRENT_CARD_API_CATALOG) == roots
    assert len({entry.api_id for entry in CURRENT_CARD_API_CATALOG.values()}) == len(roots)
    assert CURRENT_CARD_API_CATALOG["minion-giant"].api_id == 26000107


def test_rarity_levels_normalize_to_common_level_16() -> None:
    rare = CURRENT_CARD_API_CATALOG["giant"]
    assert rare.max_api_level == 14
    assert rare.normalize_level(14, 14) == 16
    assert rare.normalize_level(13, 14) == 15


@pytest.mark.parametrize(("level", "maximum"), [(0, 14), (15, 14), (14, 16), (True, 14)])
def test_rarity_levels_reject_inconsistent_api_fields(level: object, maximum: object) -> None:
    with pytest.raises(ValueError, match="invalid_card_level"):
        CardAPIEntry(26000003, 14).normalize_level(level, maximum)
