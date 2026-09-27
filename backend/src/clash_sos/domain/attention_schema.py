"""Frozen card/tower token layout with June 2026 defaults.

The schema maps compact identity tokens to base/form indices and static
attributes. It can be built and validated without importing PyTorch.
"""

from collections.abc import Sequence
from functools import cached_property
from hashlib import sha256
from json import loads
from typing import Literal, Self, cast

from pydantic import Field, model_validator

from clash_sos.domain.canonical import CardForm
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
from clash_sos.domain.card_catalog import CardCatalog
from clash_sos.domain.manifests import ManifestModel, Sha256
from clash_sos.domain.tower_catalog import TowerCatalog

SCHEMA_VERSION = "attention-card-schema:v1"
MAX_CARD_IDENTITIES = 2**16
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
TOWER_INTERPRETATION = "deck-and-tower matchup estimate under an equal-skill assumption"
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
    if identity.endswith(":tower"):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    attribute = table.for_identity(identity)
    values = (
        float(attribute.elixir or 0) / ELIXIR_DIVISOR,
        float(attribute.elixir is None),
        *(float(role in attribute.roles) for role in ROLE_ORDER),
    )
    return cast(AttributeVector, values)


def _catalog_identities(snapshot: dict[str, object]) -> tuple[str, ...]:
    catalog = CardCatalog.from_payload(snapshot)
    return tuple(entry.card.identity_key for entry in catalog.entries)


class AttentionCardSchema(ManifestModel):
    """Artifact-ready identity vocabulary and checked static token lookup."""

    schema_version: Literal["attention-card-schema:v1", "attention-card-schema:v2"] = SCHEMA_VERSION
    catalog_version: str = Field(default=CATALOG_VERSION, min_length=1)
    catalog_sha256: Sha256
    catalog_snapshot: dict[str, object]
    canonical_schema_version: Literal[
        "kaggle-v6-ranked16-schema:v1", "official-ranked16-schema:v1"
    ] = CANONICAL_SCHEMA_VERSION
    tower_catalog: TowerCatalog | None = None
    balance_era_id: str = Field(default=BALANCE_ERA_ID, min_length=1)
    accepted_level: Literal[16] = ACCEPTED_CARD_LEVEL
    attribute_version: str = Field(default=ATTRIBUTE_VERSION, min_length=1)
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
        "deck-only matchup estimate under an equal-skill assumption",
        "deck-and-tower matchup estimate under an equal-skill assumption",
    ] = PROBABILITY_INTERPRETATION

    @model_validator(mode="after")
    def validate_lookup(self) -> Self:
        """Reject reordered, incomplete, or internally inconsistent snapshots."""
        if sha256(canonical_json_bytes(self.catalog_snapshot)).hexdigest() != self.catalog_sha256:
            raise ValueError("card catalog hash does not match snapshot")
        deck_identities = _catalog_identities(self.catalog_snapshot)
        tower_identities = (
            tuple(entry.identity for entry in self.tower_catalog.entries)
            if self.tower_catalog is not None
            else ()
        )
        catalog_identities = (*deck_identities, *tower_identities)
        if (self.schema_version == "attention-card-schema:v2") != bool(tower_identities):
            raise ValueError("schema version must match tower input layout")
        expected_canonical = (
            "official-ranked16-schema:v1" if tower_identities else CANONICAL_SCHEMA_VERSION
        )
        expected_interpretation = (
            TOWER_INTERPRETATION if tower_identities else PROBABILITY_INTERPRETATION
        )
        if (
            self.canonical_schema_version != expected_canonical
            or self.probability_interpretation != expected_interpretation
        ):
            raise ValueError("schema population and interpretation do not match input layout")
        if self.catalog_snapshot["catalog_version"] != self.catalog_version:
            raise ValueError("catalog version does not match snapshot")
        if tuple(sorted(catalog_identities)) != self.identity_vocab:
            raise ValueError("identity vocabulary does not match catalog snapshot")
        if self.attribute_columns != ATTRIBUTE_COLUMNS:
            raise ValueError("attribute columns do not match schema version")
        expected_bases = tuple(
            sorted({identity.rsplit(":", 1)[0] for identity in catalog_identities})
        )
        if self.base_vocab != expected_bases:
            raise ValueError("base vocabulary does not match catalog snapshot")
        forms = tuple(
            sorted([form.value for form in CardForm] + (["tower"] if tower_identities else []))
        )
        if self.form_vocab != forms:
            raise ValueError("form vocabulary must contain every supported form")
        if len(self.identity_vocab) > MAX_CARD_IDENTITIES:
            raise ValueError("identity vocabulary exceeds uint16 capacity")
        if any(
            len(values) != len(self.identity_vocab)
            for values in (self.token_base_indices, self.token_form_indices, self.token_attributes)
        ):
            raise ValueError("token lookup arrays must align with identity vocabulary")
        table = CardAttributeTable.from_payload(self.attribute_snapshot)
        deck_bases = {identity.rsplit(":", 1)[0] for identity in deck_identities}
        tower_bases = {identity.rsplit(":", 1)[0] for identity in tower_identities}
        if deck_bases & tower_bases:
            raise ValueError("tower and deployable card IDs must be distinct")
        if table.version != self.attribute_version or set(table.cards) != deck_bases:
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

    @property
    def input_size(self) -> int:
        return DECK_SIZE + int(self.tower_catalog is not None)

    @cached_property
    def tower_indices(self) -> frozenset[int]:
        return frozenset(
            index for identity, index in self.identity_index.items() if identity.endswith(":tower")
        )

    def encode_deck(
        self, identities: Sequence[str], *, levels: Sequence[int] | None = None
    ) -> tuple[int, ...]:
        """Return sorted tokens; reject unsupported decks and non-16 input levels."""
        if len(identities) != DECK_SIZE or (levels is not None and len(levels) != DECK_SIZE):
            raise ValueError("deck must contain exactly eight aligned cards")
        if len(set(identities)) != DECK_SIZE:
            raise ValueError("deck must contain distinct card-form identities")
        if any(identity.endswith(":tower") for identity in identities):
            raise ValueError("tower troops cannot occupy deployable deck slots")
        if levels is not None and any(level != self.accepted_level for level in levels):
            raise ValueError("attention schema requires card level 16")
        try:
            return tuple(sorted(self.identity_index[identity] for identity in identities))
        except KeyError as error:
            raise ValueError(f"unsupported card identity: {error.args[0]}") from error

    def encode_side(
        self,
        identities: Sequence[str],
        *,
        tower: str | None = None,
        levels: Sequence[int] | None = None,
        tower_level: int | None = None,
    ) -> tuple[int, ...]:
        """Encode eight cards plus a required known tower for v2 inputs."""
        cards = self.encode_deck(identities, levels=levels)
        if self.tower_catalog is None:
            return cards
        token = self.identity_index.get(tower or "")
        if token not in self.tower_indices:
            raise ValueError("tower-aware inputs require a known tower identity")
        if tower_level is not None and tower_level != self.accepted_level:
            raise ValueError("attention schema requires tower level 16")
        assert token is not None
        return (*cards, token)

    def fingerprint(self) -> str:
        """Keep the v1 default hash stable for already-published input caches."""
        payload = self.model_dump(mode="python")
        if self.tower_catalog is None:
            payload.pop("tower_catalog")
        if self.network.neural_component:
            payload["network"].pop("neural_component")
        return sha256(canonical_json_bytes(payload)).hexdigest()


def build_attention_schema(
    catalog_bytes: bytes,
    *,
    attributes: CardAttributeTable,
    network: AttentionModelConfig,
    balance_era_id: str = BALANCE_ERA_ID,
    tower_catalog: TowerCatalog | None = None,
) -> AttentionCardSchema:
    """Freeze catalog identities and attributes into the selected input layout."""
    raw_snapshot: object = loads(catalog_bytes)
    if not isinstance(raw_snapshot, dict):
        raise ValueError("card catalog snapshot must be an object")
    catalog_snapshot = cast(dict[str, object], raw_snapshot)
    tower_identities = (
        tuple(entry.identity for entry in tower_catalog.entries) if tower_catalog else ()
    )
    ordered = tuple(sorted((*_catalog_identities(catalog_snapshot), *tower_identities)))
    base_vocab = tuple(sorted({identity.rsplit(":", 1)[0] for identity in ordered}))
    form_vocab = tuple(
        sorted([form.value for form in CardForm] + (["tower"] if tower_catalog else []))
    )
    base_index = {card_id: index for index, card_id in enumerate(base_vocab)}
    form_index = {form: index for index, form in enumerate(form_vocab)}
    return AttentionCardSchema(
        schema_version="attention-card-schema:v2" if tower_catalog else SCHEMA_VERSION,
        canonical_schema_version="official-ranked16-schema:v1"
        if tower_catalog
        else CANONICAL_SCHEMA_VERSION,
        tower_catalog=tower_catalog,
        probability_interpretation=TOWER_INTERPRETATION
        if tower_catalog
        else PROBABILITY_INTERPRETATION,
        catalog_version=cast(str, catalog_snapshot["catalog_version"]),
        catalog_sha256=sha256(canonical_json_bytes(catalog_snapshot)).hexdigest(),
        catalog_snapshot=catalog_snapshot,
        attribute_version=attributes.version,
        balance_era_id=balance_era_id,
        base_vocab=base_vocab,
        form_vocab=form_vocab,
        identity_vocab=ordered,
        token_base_indices=tuple(base_index[identity.rsplit(":", 1)[0]] for identity in ordered),
        token_form_indices=tuple(form_index[identity.rsplit(":", 1)[1]] for identity in ordered),
        token_attributes=tuple(_attribute_vector(identity, attributes) for identity in ordered),
        attribute_snapshot=attributes.to_payload(),
        network=network,
    )
