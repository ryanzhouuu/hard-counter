"""Read supportCards without inventing towers or folding them into deck slots."""

from dataclasses import dataclass
from typing import cast

from clash_sos.domain.tower_catalog import TowerCatalog
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_TOWER_CATALOG


@dataclass(frozen=True)
class LiveTower:
    name: str
    identity: str | None
    level: int | None
    icon_url: str | None
    issue: str | None


def adapt_tower(
    side: dict[str, object],
    catalog: TowerCatalog = CURRENT_TOWER_CATALOG,
) -> LiveTower:
    """Resolve one tower by agreeing ID/name, retaining unknowns and absent levels."""
    values = side.get("supportCards")
    if values is None or values == []:
        return LiveTower("Unknown tower", None, None, None, "missing_tower")
    if not isinstance(values, list):
        return LiveTower("Unknown tower", None, None, None, "invalid_tower")
    rows = cast(list[object], values)
    if len(rows) != 1 or not isinstance(rows[0], dict):
        return LiveTower("Unknown tower", None, None, None, "invalid_tower")
    raw = cast(dict[str, object], rows[0])
    name, api_id = raw.get("name"), raw.get("id")
    name = name if isinstance(name, str) else "Unknown tower"
    entry = catalog.by_api_id.get(api_id) if type(api_id) is int else None
    if entry is None:
        return LiveTower(name, None, None, None, "unknown_tower")
    if name != "Unknown tower" and " ".join(name.casefold().split()) != entry.name.casefold():
        return LiveTower(name, None, None, None, "invalid_tower")
    if "maxLevel" in raw and raw["maxLevel"] != entry.max_api_level:
        return LiveTower(name, None, None, None, "invalid_tower")
    level = raw.get("level")
    normalized = None
    if level is not None:
        if type(level) is not int:
            return LiveTower(name, None, None, None, "invalid_tower")
        try:
            normalized = entry.normalize_level(level)
        except ValueError:
            return LiveTower(name, None, None, None, "invalid_tower")
    icons = raw.get("iconUrls")
    url = cast(dict[str, object], icons).get("medium") if isinstance(icons, dict) else None
    icon = (
        url
        if isinstance(url, str) and url.startswith("https://api-assets.clashroyale.com/")
        else None
    )
    return LiveTower(entry.name, entry.identity, normalized, icon, None)
