"""Validate user-built decks and expose the selected artifact's input choices."""

from dataclasses import asdict
from json import loads
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from clash_sos.application.live_analysis import LiveModelReport
from clash_sos.application.live_model import LiveModelError, score_live_decks
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.card_catalog import CardCatalog
from clash_sos.domain.model_artifact import ModelArtifactManifest

CardIdentity = Annotated[
    str,
    Field(
        strict=True,
        max_length=96,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*:(base|evolution|hero|champion)$",
    ),
]
TowerIdentity = Annotated[
    str, Field(strict=True, max_length=96, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*:tower$")
]


class MatchupDeck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cards: tuple[CardIdentity, ...] = Field(min_length=8, max_length=8)
    tower: TowerIdentity | None = None

    @model_validator(mode="after")
    def unique_cards(self) -> Self:
        if len({identity.rsplit(":", 1)[0] for identity in self.cards}) != 8:
            raise ValueError("Select eight different cards; forms of the same card share a slot.")
        return self


class MatchupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    side_a: MatchupDeck
    side_b: MatchupDeck


class CardChoice(BaseModel):
    identity: str
    name: str
    form: str


class TowerChoice(BaseModel):
    identity: str
    name: str


class MatchupOptions(BaseModel):
    model: LiveModelReport
    cards: tuple[CardChoice, ...]
    towers: tuple[TowerChoice, ...]


class MatchupResult(BaseModel):
    model: LiveModelReport
    side_a_win_probability: float = Field(ge=0, le=1)
    side_b_win_probability: float = Field(ge=0, le=1)


class MatchupInputError(ValueError):
    """A validly shaped deck falls outside the selected model's input contract."""


def matchup_options(artifact: Path) -> MatchupOptions:
    """Offer only frozen card forms and towers after verifying the serving artifact."""
    info, _ = score_live_decks(artifact, [])
    try:
        raw_manifest = (artifact / "manifest.json").read_bytes()
        towers: tuple[TowerChoice, ...] = ()
        if loads(raw_manifest)["manifest_type"] == "matchup_attention":
            schema = AttentionCardSchema.model_validate_json(
                (artifact / "feature-schema.json").read_bytes()
            )
            catalog = CardCatalog.from_payload(schema.catalog_snapshot)
            if schema.tower_catalog is not None:
                towers = tuple(
                    TowerChoice(identity=entry.identity, name=entry.name)
                    for entry in schema.tower_catalog.entries
                )
        else:
            manifest = ModelArtifactManifest.model_validate_json(raw_manifest)
            member = next(file for file in manifest.files if file.kind == "card_catalog")
            catalog = CardCatalog.from_payload(loads((artifact / member.path).read_bytes()))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        raise LiveModelError("model_unavailable") from error
    return MatchupOptions(
        model=LiveModelReport(**asdict(info)),
        cards=tuple(
            sorted(
                (
                    CardChoice(
                        identity=entry.card.identity_key,
                        name=entry.source_name.title(),
                        form=entry.card.form.value,
                    )
                    for entry in catalog.entries
                ),
                key=lambda card: card.name,
            )
        ),
        towers=towers,
    )


def predict_deck_matchup(artifact: Path, request: MatchupRequest) -> MatchupResult:
    info, predictions = score_live_decks(
        artifact,
        [(request.side_a.cards, request.side_b.cards)],
        tower_pairs=[(request.side_a.tower, request.side_b.tower)],
    )
    if info.input_scope == "deck_and_tower" and (
        request.side_a.tower is None or request.side_b.tower is None
    ):
        raise MatchupInputError("Select a tower troop for both decks.")
    if info.input_scope == "deck_only" and (
        request.side_a.tower is not None or request.side_b.tower is not None
    ):
        raise MatchupInputError("The selected model uses cards only; omit tower troops.")
    probability = predictions[0].side_a_win_probability
    if probability is None:
        raise MatchupInputError(
            "The selected model does not cover these card forms or tower troops."
        )
    return MatchupResult(
        model=LiveModelReport(**asdict(info)),
        side_a_win_probability=probability,
        side_b_win_probability=1 - probability,
    )
