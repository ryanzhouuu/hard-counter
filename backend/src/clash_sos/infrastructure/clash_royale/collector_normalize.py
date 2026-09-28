"""Turn official battle-log objects into stable, side-independent collection records."""

import re
from datetime import UTC, datetime
from hashlib import sha256
from typing import Self, cast

from pydantic import field_validator, model_validator

from clash_sos.domain.canonical import DomainModel
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.infrastructure.clash_royale.adapter import LiveCard, adapt_battle
from clash_sos.infrastructure.clash_royale.card_api_catalog import CURRENT_CARD_API_CATALOG
from clash_sos.infrastructure.clash_royale.tower_adapter import adapt_tower

RANKED_MODES = frozenset({"Ranked1v1_NewArena", "Ranked1v1_NewArena2"})
TAG_PATTERN = re.compile(r"#[A-Z0-9]{3,15}")


class BattleRejected(ValueError):
    """A stable reason for omitting an unnormalizable API observation."""


class CollectedSide(DomainModel):
    """A side's identities and rarity-normalized levels, without display fields."""

    tag: str
    cards: tuple[tuple[str, int], ...]
    tower: str
    tower_level: int

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        """Guard stored variants against duplicated or reordered deck slots."""
        if not TAG_PATTERN.fullmatch(self.tag):
            raise ValueError("invalid player tag")
        if len(self.cards) != 8 or len({identity for identity, _ in self.cards}) != 8:
            raise ValueError("collected side requires eight distinct identities")
        if self.cards != tuple(sorted(self.cards)):
            raise ValueError("collected cards must be sorted")
        if any(not 1 <= level <= 16 for _, level in self.cards):
            raise ValueError("collected card level is out of range")
        if not self.tower.endswith(":tower") or not 1 <= self.tower_level <= 16:
            raise ValueError("collected tower is invalid")
        return self


class CollectedBattle(DomainModel):
    """One comparable version of a match, regardless of the observed player."""

    timestamp: datetime
    battle_type: str
    mode: str
    sides: tuple[CollectedSide, CollectedSide]
    winner_tag: str | None

    @field_validator("timestamp")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Use one timestamp spelling for both players' match keys."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("battle time must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def check_sides(self) -> Self:
        """Require one canonical tag order and a winner drawn from those sides."""
        tags = tuple(side.tag for side in self.sides)
        if tags != tuple(sorted(set(tags))) or (self.winner_tag not in (*tags, None)):
            raise ValueError("collected battle sides or winner are inconsistent")
        return self

    @property
    def event_key(self) -> str:
        """Identify a candidate match without assuming decks or outcome agree."""
        payload = {
            "source": "official-api",
            "timestamp": self.timestamp,
            "players": [side.tag for side in self.sides],
        }
        return sha256(canonical_json_bytes(payload)).hexdigest()

    @property
    def variant_hash(self) -> str:
        """Compare all training-relevant values under canonical side order."""
        return sha256(canonical_json_bytes(self.model_dump(mode="python"))).hexdigest()

    @property
    def exclusion_reason(self) -> str | None:
        """Preserve coherent ineligible variants for conflict detection."""
        if self.battle_type != "pathOfLegend" or self.mode not in RANKED_MODES:
            return "unsupported_mode"
        if self.winner_tag is None:
            return "draw"
        if any(
            side.tower_level != 16 or any(level != 16 for _, level in side.cards)
            for side in self.sides
        ):
            return "non_max_level"
        return None


def _tag(value: object) -> str:
    """Reject malformed tags before using them as persistent identities."""
    if not isinstance(value, str):
        raise BattleRejected("invalid_player_tag")
    tag = value.strip().upper()
    if not TAG_PATTERN.fullmatch(tag):
        raise BattleRejected("invalid_player_tag")
    return tag


def _side(raw: dict[str, object], resolved_cards: tuple[LiveCard, ...]) -> CollectedSide:
    """Require a complete deck and known tower with verified API level offsets."""
    tag = _tag(raw.get("tag"))
    cards_value = raw.get("cards")
    if not isinstance(cards_value, list) or len(cast(list[object], cards_value)) != 8:
        raise BattleRejected("incomplete_deck")
    cards: list[tuple[str, int]] = []
    if len(resolved_cards) != 8:
        raise BattleRejected("incomplete_deck")
    for value, resolved in zip(cast(list[object], cards_value), resolved_cards, strict=True):
        if not isinstance(value, dict):
            raise BattleRejected("invalid_card")
        card = cast(dict[str, object], value)
        identity = resolved.identity
        if identity is None:
            raise BattleRejected("unknown_card")
        root = identity.rsplit(":", 1)[0]
        mapping = CURRENT_CARD_API_CATALOG[root]
        if type(card.get("id")) is not int or card["id"] != mapping.api_id:
            raise BattleRejected("invalid_card_id")
        try:
            level = mapping.normalize_level(card.get("level"), card.get("maxLevel"))
        except ValueError as error:
            raise BattleRejected("invalid_card_level") from error
        cards.append((identity, level))
    if len({identity for identity, _ in cards}) != 8:
        raise BattleRejected("repeated_card")
    tower = adapt_tower(raw)
    if tower.identity is None or tower.level is None:
        raise BattleRejected(tower.issue or "invalid_tower")
    return CollectedSide(
        tag=tag, cards=tuple(sorted(cards)), tower=tower.identity, tower_level=tower.level
    )


def normalize_battle(raw: dict[str, object], observed_tag: str) -> CollectedBattle:
    """Normalize a one-versus-one log entry or raise a countable rejection."""
    teams, opponents = raw.get("team"), raw.get("opponent")
    if not isinstance(teams, list) or not isinstance(opponents, list):
        raise BattleRejected("team_battle")
    team_rows, opponent_rows = cast(list[object], teams), cast(list[object], opponents)
    if (
        len(team_rows) != 1
        or len(opponent_rows) != 1
        or not isinstance(team_rows[0], dict)
        or not isinstance(opponent_rows[0], dict)
    ):
        raise BattleRejected("team_battle")
    team = cast(dict[str, object], team_rows[0])
    opponent = cast(dict[str, object], opponent_rows[0])
    team_tag = _tag(team.get("tag"))
    live = adapt_battle(raw, team_tag)
    sides = tuple(
        sorted(
            (_side(team, live.player_cards), _side(opponent, live.opponent_cards)),
            key=lambda side: side.tag,
        )
    )
    if _tag(observed_tag) not in {side.tag for side in sides}:
        raise BattleRejected("player_not_in_battle")
    timestamp = live.timestamp
    if timestamp is None:
        raise BattleRejected("invalid_battle_time")
    team_crowns, opponent_crowns = team.get("crowns"), opponent.get("crowns")
    if (
        type(team_crowns) is not int
        or type(opponent_crowns) is not int
        or team_crowns < 0
        or opponent_crowns < 0
    ):
        raise BattleRejected("invalid_outcome")
    winner_tag = None
    if team_crowns != opponent_crowns:
        winner_tag = team_tag if team_crowns > opponent_crowns else _tag(opponent.get("tag"))
    game_mode = raw.get("gameMode")
    mode = cast(dict[str, object], game_mode).get("name") if isinstance(game_mode, dict) else None
    battle_type = raw.get("type")
    if not isinstance(mode, str) or not isinstance(battle_type, str):
        raise BattleRejected("invalid_mode")
    return CollectedBattle(
        timestamp=timestamp,
        battle_type=battle_type,
        mode=mode,
        sides=cast(tuple[CollectedSide, CollectedSide], sides),
        winner_tag=winner_tag,
    )
