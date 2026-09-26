"""Current mappings add released identities without rewriting the June catalog."""

from clash_sos.infrastructure.clash_royale.catalog import CURRENT_CARD_CATALOG
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def test_current_catalog_preserves_june_and_adds_released_forms() -> None:
    old = {entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries}
    current = {entry.card.identity_key for entry in CURRENT_CARD_CATALOG.entries}
    assert current - old == {
        "ronin:base",
        "minion-giant:base",
        "elite-barbarians:evolution",
        "valkyrie:hero",
        "berserker:hero",
        "ice-wizard:hero",
    }
    assert old <= current
    assert len(KAGGLE_V6_CARDS.entries) == 176
    assert len(CURRENT_CARD_CATALOG.entries) == 182
