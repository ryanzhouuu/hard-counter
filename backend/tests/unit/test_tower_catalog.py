"""Tower identities stay distinct from deck cards and normalize API levels."""

import pytest
from pydantic import ValidationError

from clash_sos.domain.tower_catalog import TowerCatalog
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_TOWER_CATALOG


def test_released_towers_keep_api_ids_and_normalized_levels() -> None:
    catalog = CURRENT_TOWER_CATALOG
    assert {entry.identity for entry in catalog.entries} == {
        "tower-princess:tower",
        "cannoneer:tower",
        "dagger-duchess:tower",
        "royal-chef:tower",
    }
    for entry in catalog.entries:
        assert catalog.by_api_id[entry.api_id] is entry
        assert catalog.by_name[entry.name.casefold()] is entry
        assert entry.normalize_level(entry.max_api_level) == 16
        assert entry.normalize_level(entry.max_api_level - 1) == 15
        for invalid in (0, True, entry.max_api_level + 1):
            with pytest.raises(ValueError, match="level"):
                entry.normalize_level(invalid)


def test_tower_catalog_rejects_ambiguous_entries() -> None:
    payload = CURRENT_TOWER_CATALOG.model_dump(mode="python")
    payload["entries"] = (payload["entries"][0], payload["entries"][0])
    with pytest.raises(ValidationError, match="unique"):
        TowerCatalog.model_validate(payload)
