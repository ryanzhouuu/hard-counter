import pytest

from clash_sos.domain.card_attributes import CARD_ATTRIBUTES, SUMMARY_COLUMNS, CardAttributeTable
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def test_attribute_table_covers_every_catalog_card() -> None:
    catalog_ids = {
        entry.card.card_id.value
        for entry in KAGGLE_V6_CARDS.entries
        if entry.card.form.value in {"base", "champion"}
    }
    assert CARD_ATTRIBUTES.version == "card-attributes:2026-06"
    assert CARD_ATTRIBUTES.cards.keys() == catalog_ids
    assert SUMMARY_COLUMNS == (
        "mean_elixir",
        "win_condition_count",
        "building_count",
        "spell_count",
        "cycle_count",
        "air_defense_count",
        "bait_count",
    )


def test_known_cards_keep_june_2026_elixir_and_roles() -> None:
    knight = CARD_ATTRIBUTES.for_identity("knight:evolution")
    xbow = CARD_ATTRIBUTES.for_identity("x-bow:base")
    barrel = CARD_ATTRIBUTES.for_identity("goblin-barrel:evolution")
    assert knight.elixir == 3 and knight.roles == frozenset()
    assert xbow.elixir == 6 and xbow.roles == frozenset({"win_condition", "building"})
    assert barrel.roles == frozenset({"win_condition", "spell", "bait"})
    assert CARD_ATTRIBUTES.for_identity("goblin-hut:base").elixir == 4
    assert CARD_ATTRIBUTES.for_identity("void:base").elixir == 3
    assert CARD_ATTRIBUTES.for_identity("spirit-empress:base").elixir == 3
    assert CARD_ATTRIBUTES.for_identity("mirror:base").elixir is None
    assert "cycle" in CARD_ATTRIBUTES.for_identity("skeletons:base").roles
    assert CARD_ATTRIBUTES.for_identity("musketeer:base").roles == frozenset({"air_defense"})


def test_summaries_average_elixir_and_count_every_role() -> None:
    summaries = CARD_ATTRIBUTES.summaries(("x-bow:base", "skeletons:base", "mirror:base"))
    assert summaries == pytest.approx((3.8333333333333335, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
    assert CARD_ATTRIBUTES.summaries(()) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_missing_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing card attributes"):
        CARD_ATTRIBUTES.for_identity("not-a-card:base")


def test_attribute_versions_are_data_inputs() -> None:
    payload = CARD_ATTRIBUTES.to_payload()
    payload["attribute_version"] = "card-attributes:next"
    table = CardAttributeTable.from_payload(payload)
    assert table.version == "card-attributes:next"
    assert table.cards == CARD_ATTRIBUTES.cards
    payload["attribute_version"] = " "
    with pytest.raises(ValueError, match="version is required"):
        CardAttributeTable.from_payload(payload)
