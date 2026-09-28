"""Pin official root-card IDs and rarity levels for strict collection parsing."""

from dataclasses import dataclass
from importlib.resources import files
from json import loads
from typing import cast

from clash_sos.domain.card_catalog import CardCatalog
from clash_sos.infrastructure.clash_royale.catalog import CURRENT_CARD_CATALOG


@dataclass(frozen=True)
class CardAPIEntry:
    """Identify a deployable root card independently of model token numbers."""

    api_id: int
    max_api_level: int

    def normalize_level(self, level: object, max_level: object) -> int:
        """Reject mismatched API metadata before converting to the 1-16 scale."""
        if (
            type(level) is not int
            or type(max_level) is not int
            or max_level != self.max_api_level
            or not 1 <= level <= self.max_api_level
        ):
            raise ValueError("invalid_card_level")
        return level + 16 - self.max_api_level


def load_card_api_catalog(catalog: CardCatalog = CURRENT_CARD_CATALOG) -> dict[str, CardAPIEntry]:
    """Require one API mapping for every root in the selected live catalog."""
    payload = loads(files(__package__).joinpath("card_api_ids.json").read_text(encoding="utf-8"))
    rows = cast(dict[str, object], payload["cards"])
    roots = {
        entry.card.card_id.value
        for entry in catalog.entries
        if entry.card.form.value in {"base", "champion"}
    }
    if set(rows) != roots:
        raise ValueError("API card mappings do not cover the live catalog")
    result: dict[str, CardAPIEntry] = {}
    for card_id, raw in rows.items():
        if not isinstance(raw, dict):
            raise ValueError("invalid API card mapping")
        entry = cast(dict[str, object], raw)
        api_id, max_level = entry.get("api_id"), entry.get("max_api_level")
        if type(api_id) is not int or type(max_level) is not int or not 1 <= max_level <= 16:
            raise ValueError("invalid API card mapping")
        result[card_id] = CardAPIEntry(api_id, max_level)
    if len({entry.api_id for entry in result.values()}) != len(result):
        raise ValueError("API card IDs must be unique")
    return result


CURRENT_CARD_API_CATALOG = load_card_api_catalog()
