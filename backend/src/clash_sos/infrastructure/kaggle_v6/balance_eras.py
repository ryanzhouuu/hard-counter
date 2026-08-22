"""Evidence-backed balance-era registry for Kaggle version 6 preparation runs."""

from datetime import UTC, datetime

from pydantic import AnyHttpUrl

from clash_sos.domain.canonical import BalanceEra, BalanceEraRegistry, EraBoundaryEvidence
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS

KAGGLE_V6_ERA_REGISTRY_VERSION = "kaggle-v6-balance-eras:v1"

_MAY_2026_UPDATE_URL = AnyHttpUrl(
    "https://supercell.com/en/games/clashroyale/blog/release-notes/may-balance-changes-2026/"
)
_JUNE_2026_UPDATE_URL = AnyHttpUrl(
    "https://supercell.com/en/games/clashroyale/blog/release-notes/june-balance-changes-2026/"
)
_JULY_2026_UPDATE_URL = AnyHttpUrl(
    "https://supercell.com/en/games/clashroyale/blog/release-notes/july-balance-changes-2026/"
)

_MAY_2026_START_EVIDENCE = EraBoundaryEvidence(
    summary="Supercell documented the May balance update for May 4, 2026.",
    reference=_MAY_2026_UPDATE_URL,
    stated_date=datetime(2026, 5, 4, tzinfo=UTC),
    precision="day",
    boundary_policy=(
        "The exact rollout instant is not published; the era starts at midnight UTC "
        "the day after the stated update date."
    ),
)
_MAY_2026_END_EVIDENCE = EraBoundaryEvidence(
    summary="Supercell documented the June balance update for June 1, 2026.",
    reference=_JUNE_2026_UPDATE_URL,
    stated_date=datetime(2026, 6, 1, tzinfo=UTC),
    precision="day",
    boundary_policy=(
        "The transition day is conservatively left unsupported; this era ends exclusive "
        "at the start of the stated update date."
    ),
)
_JUNE_2026_START_EVIDENCE = EraBoundaryEvidence(
    summary="Supercell documented the June balance update for June 1, 2026.",
    reference=_JUNE_2026_UPDATE_URL,
    stated_date=datetime(2026, 6, 1, tzinfo=UTC),
    precision="day",
    boundary_policy=(
        "The transition day is conservatively left unsupported; this era starts at midnight UTC "
        "the day after the stated update date."
    ),
)
_JUNE_2026_END_EVIDENCE = EraBoundaryEvidence(
    summary="Supercell documents the next balance update for July 6, 2026.",
    reference=_JULY_2026_UPDATE_URL,
    stated_date=datetime(2026, 7, 6, tzinfo=UTC),
    precision="day",
    boundary_policy="This era ends exclusive at the start of the next documented update date.",
)

KAGGLE_V6_ERA_REGISTRY = BalanceEraRegistry(
    registry_version=KAGGLE_V6_ERA_REGISTRY_VERSION,
    eras=(
        BalanceEra(
            era_id="2026-05",
            valid_from=datetime(2026, 5, 5, tzinfo=UTC),
            valid_to=datetime(2026, 6, 1, tzinfo=UTC),
            card_catalog_version=KAGGLE_V6_CARDS.version,
            start_evidence=_MAY_2026_START_EVIDENCE,
            end_evidence=_MAY_2026_END_EVIDENCE,
        ),
        BalanceEra(
            era_id="2026-06",
            valid_from=datetime(2026, 6, 2, tzinfo=UTC),
            valid_to=datetime(2026, 7, 6, tzinfo=UTC),
            card_catalog_version=KAGGLE_V6_CARDS.version,
            start_evidence=_JUNE_2026_START_EVIDENCE,
            end_evidence=_JUNE_2026_END_EVIDENCE,
        ),
    ),
)
