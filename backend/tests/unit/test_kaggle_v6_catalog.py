from hashlib import sha256

from clash_sos.domain.canonical import CardForm, CardId
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def test_catalog_has_complete_unique_source_coverage() -> None:
    entries = KAGGLE_V6_CARDS.entries

    assert [entry.source_id for entry in entries] == list(range(176))
    assert len({entry.source_name for entry in entries}) == 176


def test_catalog_maps_forms_to_explicit_base_card_ids() -> None:
    pekka = KAGGLE_V6_CARDS.find(4)
    evolved_pekka = KAGGLE_V6_CARDS.find(123)
    hero_mini_pekka = KAGGLE_V6_CARDS.find(168)
    assert pekka is not None
    assert evolved_pekka is not None
    assert hero_mini_pekka is not None
    assert pekka.card.card_id == CardId("pekka")
    assert evolved_pekka.card.model_dump() == {
        "card_id": "pekka",
        "form": CardForm.EVOLUTION,
    }
    assert hero_mini_pekka.card.model_dump() == {
        "card_id": "mini-pekka",
        "form": CardForm.HERO,
    }


def test_catalog_uses_reviewed_champion_allowlist() -> None:
    champion_source_ids = {
        entry.source_id for entry in KAGGLE_V6_CARDS.entries if entry.card.form is CardForm.CHAMPION
    }

    assert champion_source_ids == {65, 68, 69, 70, 71, 77, 81, 84}
    spirit_empress = KAGGLE_V6_CARDS.find(119)
    assert spirit_empress is not None
    assert spirit_empress.card.form is CardForm.BASE


def test_catalog_serialization_is_stable() -> None:
    assert KAGGLE_V6_CARDS.version == "kaggle-v6-2026-06"
    assert sha256(KAGGLE_V6_CARDS.serialize()).hexdigest() == (
        "a4a67519172d43ef737f51547553efbd34cc0302de51d4a7643f34f38c13e4a1"
    )
