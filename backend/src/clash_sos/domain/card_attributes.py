"""Versioned elixir and role attributes with a packaged June 2026 default.

Evolutions and heroes inherit the base card. Mirror has no fixed deploy cost.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from json import loads
from typing import cast

SUMMARY_COLUMNS: tuple[str, ...] = (
    "mean_elixir",
    "win_condition_count",
    "building_count",
    "spell_count",
    "cycle_count",
    "air_defense_count",
    "bait_count",
)
ROLE_NAMES: frozenset[str] = frozenset(
    {"win_condition", "building", "spell", "cycle", "air_defense", "bait"}
)
ATTRIBUTE_VERSION = "card-attributes:2026-06"


@dataclass(frozen=True)
class CardAttribute:
    """Deploy cost and roles for one card id. Elixir is absent for Mirror."""

    elixir: int | None
    roles: frozenset[str]


class CardAttributeTable:
    """Lookup and deck summaries for one attribute version."""

    def __init__(self, version: str, cards: Mapping[str, CardAttribute]) -> None:
        if not version.strip():
            raise ValueError("attribute version is required")
        if not cards:
            raise ValueError("card attributes are required")
        self.version = version
        self.cards = dict(cards)

    def for_identity(self, identity: str) -> CardAttribute:
        """Return attributes for `card-id:form`. The form is ignored."""
        card_id = identity.rsplit(":", 1)[0]
        found = self.cards.get(card_id)
        if found is None:
            raise ValueError(f"missing card attributes for {identity}")
        return found

    def summaries(self, identities: Sequence[str]) -> tuple[float, ...]:
        """Return the seven summary values for the included identities.

        Mirror costs one more than the mean deploy cost of the other included
        cards. A side that is only Mirror uses cost 1. An empty side is zeros.
        """
        if not identities:
            return tuple(0.0 for _column in SUMMARY_COLUMNS)
        fixed: list[int] = []
        mirror_count = 0
        counts = {role: 0 for role in ROLE_NAMES}
        for identity in identities:
            attribute = self.for_identity(identity)
            if attribute.elixir is None:
                mirror_count += 1
            else:
                fixed.append(attribute.elixir)
            for role in attribute.roles:
                counts[role] += 1
        if mirror_count:
            base = (sum(fixed) / len(fixed)) if fixed else 0.0
            total = float(sum(fixed)) + (base + 1.0) * mirror_count
        else:
            total = float(sum(fixed))
        mean = total / len(identities)
        return (
            mean,
            float(counts["win_condition"]),
            float(counts["building"]),
            float(counts["spell"]),
            float(counts["cycle"]),
            float(counts["air_defense"]),
            float(counts["bait"]),
        )

    def to_payload(self) -> dict[str, object]:
        """Return the snapshot stored on a published feature schema."""
        return {
            "attribute_version": self.version,
            "summary_columns": list(SUMMARY_COLUMNS),
            "cards": {
                card_id: {
                    "elixir": attribute.elixir,
                    "roles": sorted(attribute.roles),
                }
                for card_id, attribute in sorted(self.cards.items())
            },
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CardAttributeTable":
        """Parse a package file or an artifact snapshot."""
        if not isinstance(payload, dict):
            raise ValueError("card attributes must be an object")
        body = cast(dict[str, object], payload)
        columns = body.get("summary_columns")
        if list(SUMMARY_COLUMNS) != columns:
            raise ValueError("summary columns do not match the attribute table")
        raw_cards = body.get("cards")
        if not isinstance(raw_cards, dict):
            raise ValueError("card attributes require a cards object")
        cards: dict[str, CardAttribute] = {}
        for card_id, raw in cast(dict[object, object], raw_cards).items():
            if not isinstance(card_id, str) or not isinstance(raw, dict):
                raise ValueError("each card attribute must be an object")
            entry = cast(dict[str, object], raw)
            elixir = entry.get("elixir")
            roles = entry.get("roles")
            if elixir is not None and (not isinstance(elixir, int) or isinstance(elixir, bool)):
                raise ValueError(f"invalid elixir for {card_id}")
            if isinstance(elixir, int) and not 1 <= elixir <= 9:
                raise ValueError(f"invalid elixir for {card_id}")
            if not isinstance(roles, list):
                raise ValueError(f"invalid roles for {card_id}")
            role_values = cast(list[object], roles)
            role_names = [role for role in role_values if isinstance(role, str)]
            if len(role_names) != len(role_values) or len(role_names) != len(set(role_names)):
                raise ValueError(f"invalid roles for {card_id}")
            if any(role not in ROLE_NAMES for role in role_names):
                raise ValueError(f"invalid roles for {card_id}")
            cards[card_id] = CardAttribute(elixir=elixir, roles=frozenset(role_names))
        if sum(attribute.elixir is None for attribute in cards.values()) != 1:
            raise ValueError("mirror must be the only card without elixir")
        if cards.get("mirror") is None or cards["mirror"].elixir is not None:
            raise ValueError("mirror must be the only card without elixir")
        version = body.get("attribute_version")
        if not isinstance(version, str):
            raise ValueError("attribute version is required")
        return cls(version, cards)


def _load_card_attributes() -> CardAttributeTable:
    resource = files(__package__).joinpath("card_attributes.json")
    return CardAttributeTable.from_payload(loads(resource.read_text(encoding="utf-8")))


CARD_ATTRIBUTES = _load_card_attributes()
