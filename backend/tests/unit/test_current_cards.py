"""Current mappings add released identities without rewriting the June catalog."""

from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.clash_royale.catalog import (
    CURRENT_CARD_ATTRIBUTES,
    CURRENT_CARD_CATALOG,
)
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


def test_current_attributes_cover_roots_and_isolate_balance_changes() -> None:
    roots = {entry.card.card_id.value for entry in CURRENT_CARD_CATALOG.entries}
    assert set(CURRENT_CARD_ATTRIBUTES.cards) == roots
    assert CURRENT_CARD_ATTRIBUTES.for_identity("void:base").elixir == 5
    assert CARD_ATTRIBUTES.for_identity("void:base").elixir == 3
    assert CURRENT_CARD_ATTRIBUTES.for_identity("ronin:base").elixir == 5
    giant = CURRENT_CARD_ATTRIBUTES.for_identity("minion-giant:base")
    assert giant.elixir == 4 and giant.roles == frozenset({"win_condition"})
    assert (
        CURRENT_CARD_ATTRIBUTES.for_identity("valkyrie:hero") == CARD_ATTRIBUTES.cards["valkyrie"]
    )
    assert CURRENT_CARD_ATTRIBUTES.for_identity("spirit-empress:base").elixir == 3


def test_current_air_response_labels_preserve_historical_attributes() -> None:
    assert CURRENT_CARD_ATTRIBUTES.version == "card-attributes:2026-10-01-r2"
    expected = {
        "furnace": {"air_defense"},
        "goblin-hut": {"building", "air_defense"},
        "zappies": {"air_defense"},
        "mother-witch": {"air_defense"},
        "royal-delivery": {"spell", "air_defense"},
        "goblin-curse": {"spell", "air_defense"},
    }
    for root, roles in expected.items():
        current = CURRENT_CARD_ATTRIBUTES.cards[root]
        assert current.roles == frozenset(roles)
        assert current.elixir == CARD_ATTRIBUTES.cards[root].elixir
        assert "air_defense" not in CARD_ATTRIBUTES.cards[root].roles
    assert CARD_ATTRIBUTES.version == "card-attributes:2026-06"
    assert CARD_ATTRIBUTES.cards["furnace"].roles == frozenset({"building"})


def test_current_rage_has_an_air_damage_role_without_rewriting_june() -> None:
    assert CURRENT_CARD_ATTRIBUTES.cards["rage"].roles == frozenset({"spell", "air_defense"})
    assert CURRENT_CARD_ATTRIBUTES.cards["rage"].elixir == 2
    assert CARD_ATTRIBUTES.cards["rage"].roles == frozenset({"spell"})
