"""Convert recent official battles to deck-only inputs without level or mode filters."""

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from json import dumps
from typing import cast

from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS, KaggleCardCatalog


@dataclass(frozen=True)
class LiveCard:
    """Preserve the upstream label when the model has no matching identity."""

    name: str
    identity: str | None


@dataclass(frozen=True)
class LiveBattle:
    """A recent battle retained for display even when it cannot be scored."""

    timestamp: datetime | None
    fingerprint: str
    mode: str
    opponent_name: str
    outcome: str
    player_cards: tuple[LiveCard, ...]
    opponent_cards: tuple[LiveCard, ...]
    skip_reason: str | None

    @property
    def player_deck(self) -> tuple[str, ...]:
        """Return identities only when the whole player deck is known."""
        return tuple(card.identity for card in self.player_cards if card.identity is not None)

    @property
    def opponent_deck(self) -> tuple[str, ...]:
        """Return identities only when the whole opponent deck is known."""
        return tuple(card.identity for card in self.opponent_cards if card.identity is not None)


def _card_names(catalog: KaggleCardCatalog) -> dict[str, str]:
    return {entry.source_name: entry.card.identity_key for entry in catalog.entries}


def _text(value: object, fallback: str = "") -> str:
    return value if isinstance(value, str) and value.strip() else fallback


def _objects(value: object) -> list[dict[str, object]]:
    """Reject a partially malformed list instead of trusting some members."""
    if not isinstance(value, list):
        return []
    rows = cast(list[object], value)
    if any(not isinstance(row, dict) for row in rows):
        return []
    return cast(list[dict[str, object]], rows)


def _timestamp(value: object) -> datetime | None:
    """Parse the official compact UTC battle timestamp, including fractions."""
    if not isinstance(value, str):
        return None
    for pattern in ("%Y%m%dT%H%M%S.%fZ", "%Y%m%dT%H%M%SZ"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _cards(side: dict[str, object], names: dict[str, str]) -> tuple[LiveCard, ...]:
    """Use explicit form markers; unknown forms never fall back to base cards."""
    cards: list[LiveCard] = []
    for raw in _objects(side.get("cards")):
        name = _text(raw.get("name"), "Unknown card")
        lookup = " ".join(name.casefold().split())
        evolution_level = raw.get("evolutionLevel")
        hero_level = raw.get("heroLevel")
        if type(evolution_level) is int and evolution_level > 0 and not lookup.startswith("evo "):
            lookup = f"evo {lookup}"
        if type(hero_level) is int and hero_level > 0 and not lookup.startswith("hero "):
            lookup = f"hero {lookup}"
        cards.append(LiveCard(name=name, identity=names.get(lookup)))
    return tuple(cards)


def _valid_deck(cards: tuple[LiveCard, ...]) -> str | None:
    """Require eight distinct identities known to the artifact catalog."""
    if len(cards) != 8:
        return "incomplete_deck"
    identities = [card.identity for card in cards]
    if any(identity is None for identity in identities):
        return "unknown_card"
    if len(set(identities)) != 8:
        return "repeated_card"
    return None


def adapt_battle(
    raw: dict[str, object], tag: str, *, catalog: KaggleCardCatalog = KAGGLE_V6_CARDS
) -> LiveBattle:
    """Keep unsupported battles visible while admitting only decisive known 1v1 decks."""
    teams = _objects(raw.get("team"))
    opponents = _objects(raw.get("opponent"))
    player_side = next((side for side in teams if side.get("tag") == tag), None)
    other_sides = opponents
    if player_side is None:
        player_side = next((side for side in opponents if side.get("tag") == tag), None)
        other_sides = teams
    other_side = other_sides[0] if len(other_sides) == 1 else {}
    game_mode = raw.get("gameMode")
    mode = (
        _text(cast(dict[str, object], game_mode).get("name")) if isinstance(game_mode, dict) else ""
    ) or _text(raw.get("type"), "Unknown mode")
    timestamp = _timestamp(raw.get("battleTime"))
    names = _card_names(catalog)
    player_cards = _cards(player_side or {}, names)
    opponent_cards = _cards(other_side, names)
    player_crowns = (player_side or {}).get("crowns")
    opponent_crowns = other_side.get("crowns")
    outcome = "unknown"
    if type(player_crowns) is int and type(opponent_crowns) is int:
        outcome = "win" if player_crowns > opponent_crowns else "loss"
        if player_crowns == opponent_crowns:
            outcome = "draw"
    reason = None
    if player_side is None:
        reason = "player_not_in_battle"
    elif len(teams) != 1 or len(opponents) != 1:
        reason = "team_battle"
    elif timestamp is None or outcome == "unknown":
        reason = "incomplete_battle"
    elif outcome == "draw":
        reason = "draw"
    else:
        reason = _valid_deck(player_cards) or _valid_deck(opponent_cards)
    fingerprint = sha256(dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return LiveBattle(
        timestamp=timestamp,
        fingerprint=fingerprint,
        mode=mode,
        opponent_name=_text(other_side.get("name"), "Unknown opponent"),
        outcome=outcome,
        player_cards=player_cards,
        opponent_cards=opponent_cards,
        skip_reason=reason,
    )
