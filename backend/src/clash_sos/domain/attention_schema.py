"""Frozen card-token layout for the June 2026 deck-only attention model.

The schema maps compact identity tokens to base/form indices and static
attributes. It can be built and validated without importing PyTorch.
"""

from collections.abc import Sequence
from functools import cached_property
from hashlib import sha256
from json import loads
from typing import Literal, Self, cast

from pydantic import Field, model_validator

from clash_sos.domain.canonical import CardForm, CardId
from clash_sos.domain.canonical_dataset import (
    ACCEPTED_CARD_LEVEL,
    CANONICAL_SCHEMA_VERSION,
    DECK_SIZE,
    canonical_json_bytes,
)
from clash_sos.domain.card_attributes import (
    ATTRIBUTE_VERSION,
    CardAttributeTable,
)
from clash_sos.domain.manifests import ManifestModel, Sha256

SCHEMA_VERSION = "attention-card-schema:v1"
CATALOG_VERSION = "kaggle-v6-2026-06"
BALANCE_ERA_ID = "2026-06"
ELIXIR_DIVISOR = 9
ROLE_ORDER = (
    "win_condition",
    "building",
    "spell",
    "cycle",
    "air_defense",
    "bait",
)
ATTRIBUTE_COLUMNS = ("scaled_elixir", "elixir_missing", *ROLE_ORDER)
PROBABILITY_INTERPRETATION = "deck-only matchup estimate under an equal-skill assumption"
AttributeVector = tuple[float, float, float, float, float, float, float, float]


class AttentionModelConfig(ManifestModel):
    """Starting network dimensions and ablation switches stored with each schema."""

    embedding_width: int = Field(default=64, gt=0)
    attention_heads: int = Field(default=4, gt=0)
    within_deck_blocks: int = Field(default=2, ge=0)
    cross_deck_blocks: int = Field(default=1, ge=0)
    feed_forward_width: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    explicit_interactions: bool = True
    neural_component: bool = True
    within_deck_attention: bool = True
    cross_deck_attention: bool = True

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Enabled attention paths need blocks with divisible head dimensions."""
        if not self.neural_component and not self.explicit_interactions:
            raise ValueError("at least one matchup component must be enabled")
        if self.neural_component and self.embedding_width % self.attention_heads:
            raise ValueError("embedding width must be divisible by attention heads")
        if self.neural_component and self.within_deck_attention and not self.within_deck_blocks:
            raise ValueError("enabled within-deck attention requires a block")
        if self.neural_component and self.cross_deck_attention and not self.cross_deck_blocks:
            raise ValueError("enabled cross-deck attention requires a block")
        return self


def _attribute_vector(identity: str, table: CardAttributeTable) -> AttributeVector:
    """Keep Mirror missingness separate from its zero numeric placeholder."""
    attribute = table.for_identity(identity)
    values = (
        float(attribute.elixir or 0) / ELIXIR_DIVISOR,
        float(attribute.elixir is None),
        *(float(role in attribute.roles) for role in ROLE_ORDER),
    )
    return cast(AttributeVector, values)


def _catalog_identities(snapshot: dict[str, object]) -> tuple[str, ...]:
    """Read every source identity from the stored, source-ordered v6 catalog."""
    entries = snapshot.get("entries")
    if snapshot.get("catalog_version") != CATALOG_VERSION or not isinstance(entries, list):
        raise ValueError("incompatible card catalog snapshot")
    identities: list[str] = []
    names: list[str] = []
    for index, raw in enumerate(cast(list[object], entries)):
        if not isinstance(raw, list):
            raise ValueError("invalid card catalog entry")
        entry = cast(list[object], raw)
        if len(entry) != 4:
            raise ValueError("invalid card catalog entry")
        source_id, source_name, card_id, form = entry
        if source_id != index or not isinstance(source_name, str) or not source_name:
            raise ValueError("card catalog source IDs and names must be ordered")
        if not isinstance(card_id, str) or not isinstance(form, str):
            raise ValueError("invalid card catalog identity")
        names.append(source_name)
        identities.append(f"{CardId(card_id).value}:{CardForm(form).value}")
    if len(identities) != 176 or len(set(identities)) != 176 or len(set(names)) != 176:
        raise ValueError("card catalog must contain 176 unique identities")
    return tuple(identities)


class AttentionCardSchema(ManifestModel):
    """Artifact-ready identity vocabulary and checked static token lookup."""

    schema_version: Literal["attention-card-schema:v1"] = SCHEMA_VERSION
    catalog_version: Literal["kaggle-v6-2026-06"] = CATALOG_VERSION
    catalog_sha256: Sha256
    catalog_snapshot: dict[str, object]
    canonical_schema_version: Literal["kaggle-v6-ranked16-schema:v1"] = CANONICAL_SCHEMA_VERSION
    balance_era_id: Literal["2026-06"] = BALANCE_ERA_ID
    accepted_level: Literal[16] = ACCEPTED_CARD_LEVEL
    attribute_version: Literal["card-attributes:2026-06"] = ATTRIBUTE_VERSION
    elixir_divisor: Literal[9] = ELIXIR_DIVISOR
    attribute_columns: tuple[str, ...] = ATTRIBUTE_COLUMNS
    base_vocab: tuple[str, ...]
    form_vocab: tuple[str, ...]
    identity_vocab: tuple[str, ...]
    token_base_indices: tuple[int, ...]
    token_form_indices: tuple[int, ...]
    token_attributes: tuple[AttributeVector, ...]
    attribute_snapshot: dict[str, object]
    network: AttentionModelConfig
    probability_interpretation: Literal[
        "deck-only matchup estimate under an equal-skill assumption"
    ] = PROBABILITY_INTERPRETATION

    @model_validator(mode="after")
    def validate_lookup(self) -> Self:
        """Reject reordered, incomplete, or internally inconsistent snapshots."""
        if sha256(canonical_json_bytes(self.catalog_snapshot)).hexdigest() != self.catalog_sha256:
            raise ValueError("card catalog hash does not match snapshot")
        catalog_identities = _catalog_identities(self.catalog_snapshot)
        if tuple(sorted(catalog_identities)) != self.identity_vocab:
            raise ValueError("identity vocabulary does not match catalog snapshot")
        if self.attribute_columns != ATTRIBUTE_COLUMNS:
            raise ValueError("attribute columns do not match schema version")
        if len(self.base_vocab) != 121 or tuple(sorted(set(self.base_vocab))) != self.base_vocab:
            raise ValueError("base vocabulary must contain 121 unique sorted cards")
        if self.form_vocab != tuple(sorted(form.value for form in CardForm)):
            raise ValueError("form vocabulary must contain every supported form")
        if len(self.identity_vocab) != 176 or tuple(sorted(set(self.identity_vocab))) != (
            self.identity_vocab
        ):
            raise ValueError("identity vocabulary must contain 176 unique sorted keys")
        if any(
            len(values) != len(self.identity_vocab)
            for values in (self.token_base_indices, self.token_form_indices, self.token_attributes)
        ):
            raise ValueError("token lookup arrays must align with identity vocabulary")
        table = CardAttributeTable.from_payload(self.attribute_snapshot)
        if table.version != self.attribute_version or set(table.cards) != set(self.base_vocab):
            raise ValueError("attribute snapshot does not cover the base vocabulary")
        for index, identity in enumerate(self.identity_vocab):
            parts = identity.rsplit(":", 1)
            if (
                len(parts) != 2
                or parts[0] not in self.base_vocab
                or parts[1] not in self.form_vocab
            ):
                raise ValueError(f"unsupported card identity: {identity}")
            base_index, form_index = self.token_base_indices[index], self.token_form_indices[index]
            if not (
                0 <= base_index < len(self.base_vocab) and 0 <= form_index < len(self.form_vocab)
            ):
                raise ValueError(f"token index is out of range for {identity}")
            if (
                self.base_vocab[base_index] != parts[0]
                or self.form_vocab[form_index] != parts[1]
                or self.token_attributes[index] != _attribute_vector(identity, table)
            ):
                raise ValueError(f"token lookup does not match {identity}")
        return self

    @cached_property
    def identity_index(self) -> dict[str, int]:
        """Cache lookup for the multi-million-row input materializer."""
        return {identity: index for index, identity in enumerate(self.identity_vocab)}

    def encode_deck(
        self, identities: Sequence[str], *, levels: Sequence[int] | None = None
    ) -> tuple[int, ...]:
        """Return sorted tokens; reject unsupported decks and non-16 input levels."""
        if len(identities) != DECK_SIZE or (levels is not None and len(levels) != DECK_SIZE):
            raise ValueError("deck must contain exactly eight aligned cards")
        if len(set(identities)) != DECK_SIZE:
            raise ValueError("deck must contain distinct card-form identities")
        if levels is not None and any(level != self.accepted_level for level in levels):
            raise ValueError("attention schema requires card level 16")
        try:
            return tuple(sorted(self.identity_index[identity] for identity in identities))
        except KeyError as error:
            raise ValueError(f"unsupported card identity: {error.args[0]}") from error

    def fingerprint(self) -> str:
        """Hash the complete validated schema for cache and artifact provenance."""
        return sha256(canonical_json_bytes(self.model_dump(mode="python"))).hexdigest()


def build_attention_schema(
    catalog_bytes: bytes,
    *,
    attributes: CardAttributeTable,
    network: AttentionModelConfig,
) -> AttentionCardSchema:
    """Snapshot catalog identities and attributes into a validated v1 schema."""
    raw_snapshot: object = loads(catalog_bytes)
    if not isinstance(raw_snapshot, dict):
        raise ValueError("card catalog snapshot must be an object")
    catalog_snapshot = cast(dict[str, object], raw_snapshot)
    ordered = tuple(sorted(_catalog_identities(catalog_snapshot)))
    base_vocab = tuple(sorted({identity.rsplit(":", 1)[0] for identity in ordered}))
    form_vocab = tuple(sorted(form.value for form in CardForm))
    base_index = {card_id: index for index, card_id in enumerate(base_vocab)}
    form_index = {form: index for index, form in enumerate(form_vocab)}
    return AttentionCardSchema(
        catalog_sha256=sha256(catalog_bytes).hexdigest(),
        catalog_snapshot=catalog_snapshot,
        base_vocab=base_vocab,
        form_vocab=form_vocab,
        identity_vocab=ordered,
        token_base_indices=tuple(base_index[identity.rsplit(":", 1)[0]] for identity in ordered),
        token_form_indices=tuple(form_index[identity.rsplit(":", 1)[1]] for identity in ordered),
        token_attributes=tuple(_attribute_vector(identity, attributes) for identity in ordered),
        attribute_snapshot=attributes.to_payload(),
        network=network,
    )
