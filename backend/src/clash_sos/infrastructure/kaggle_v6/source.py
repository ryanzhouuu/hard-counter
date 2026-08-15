"""Pinned source registration for the Kaggle version 6 snapshot."""

from datetime import UTC, datetime, timedelta

from pydantic import AnyHttpUrl

from clash_sos.domain.manifests import (
    LicenseManifest,
    RetrievalManifest,
    SourceManifest,
)

KAGGLE_V6_SOURCE_ID = "kaggle:jackmangione/clash-royale-matchups-june2026:v6"
KAGGLE_V6_ARCHIVE_NAME = "clash-royale-matchups-june2026-v6.zip"
KAGGLE_V6_ARCHIVE_SIZE = 6_148_754_593
KAGGLE_V6_ARCHIVE_SHA256 = "280ab43d916b34dd178856ada449a69cf8130f5200737572ecd994eadaa5a89b"
KAGGLE_V6_CARD_MAPPING_NAME = "cardToID.json"
KAGGLE_V6_OBSERVED_TIMESTAMP_MIN_UTC = datetime(2026, 5, 22, 20, 16, 15, tzinfo=UTC)
KAGGLE_V6_OBSERVED_TIMESTAMP_MAX_UTC = datetime(2026, 6, 26, 18, 32, 2, tzinfo=UTC)
KAGGLE_V6_OBSERVED_TIMESTAMP_EXCLUSIVE_END_UTC = KAGGLE_V6_OBSERVED_TIMESTAMP_MAX_UTC + timedelta(
    microseconds=1
)

KAGGLE_V6_SOURCE = SourceManifest(
    source_id=KAGGLE_V6_SOURCE_ID,
    provider="kaggle",
    dataset_handle="jackmangione/clash-royale-matchups-june2026",
    dataset_version=6,
    dataset_url=AnyHttpUrl(
        "https://www.kaggle.com/datasets/jackmangione/clash-royale-matchups-june2026"
    ),
    license=LicenseManifest(
        identifier="CC-BY-NC-SA-4.0",
        name="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
        url=AnyHttpUrl("https://creativecommons.org/licenses/by-nc-sa/4.0/"),
        restrictions=("noncommercial_use", "attribution", "share_alike"),
    ),
    known_limitations=(
        "Rows are winner-first and do not represent draws.",
        "The source supplies no battle identifier.",
        "Source timestamps use America/New_York and require UTC normalization.",
        "Only Ranked1v1_NewArena at observed card level 16 is supported by v1.",
    ),
    retrieval=RetrievalManifest(
        method="manual_download",
        notes="Pinned Kaggle dataset version 6; raw archive remains immutable.",
    ),
)
