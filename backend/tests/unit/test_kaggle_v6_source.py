from clash_sos.infrastructure.kaggle_v6.source import (
    KAGGLE_V6_ARCHIVE_NAME,
    KAGGLE_V6_ARCHIVE_SHA256,
    KAGGLE_V6_ARCHIVE_SIZE,
    KAGGLE_V6_CARD_MAPPING_NAME,
    KAGGLE_V6_SOURCE,
    KAGGLE_V6_SOURCE_ID,
)


def test_kaggle_v6_source_is_pinned() -> None:
    assert KAGGLE_V6_SOURCE.source_id == KAGGLE_V6_SOURCE_ID
    assert KAGGLE_V6_SOURCE.dataset_version == 6
    assert KAGGLE_V6_SOURCE.dataset_handle == ("jackmangione/clash-royale-matchups-june2026")
    assert KAGGLE_V6_ARCHIVE_NAME == "clash-royale-matchups-june2026-v6.zip"
    assert KAGGLE_V6_ARCHIVE_SIZE == 6_148_754_593
    assert KAGGLE_V6_ARCHIVE_SHA256 == (
        "280ab43d916b34dd178856ada449a69cf8130f5200737572ecd994eadaa5a89b"
    )
    assert KAGGLE_V6_CARD_MAPPING_NAME == "cardToID.json"


def test_kaggle_v6_source_records_license_and_limitations() -> None:
    assert KAGGLE_V6_SOURCE.license.identifier == "CC-BY-NC-SA-4.0"
    assert KAGGLE_V6_SOURCE.license.restrictions == (
        "noncommercial_use",
        "attribution",
        "share_alike",
    )
    limitations = " ".join(KAGGLE_V6_SOURCE.known_limitations)
    assert "winner-first" in limitations
    assert "no battle identifier" in limitations
    assert "America/New_York" in limitations
    assert "Ranked1v1_NewArena" in limitations
