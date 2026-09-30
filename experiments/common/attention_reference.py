"""Architecture-specific native attention references on verified common populations."""

from hashlib import sha256
from math import isfinite

from torch import Tensor, nn

from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.cache import oriented_digest
from experiments.common.data_access import ResearchRow
from experiments.matchup_features.model import ResearchModel


def input_encoding_digest(schema: AttentionCardSchema) -> str:
    """Compare encoding independently; native cache/protocol hashes still include network."""
    return sha256(canonical_json_bytes(schema.model_dump(exclude={"network"}))).hexdigest()


def require_identical_reference_population(
    explicit_schema: AttentionCardSchema,
    explicit_rows: tuple[ResearchRow, ...],
    attention_schema: AttentionCardSchema,
    attention_rows: tuple[ResearchRow, ...],
) -> str:
    if input_encoding_digest(explicit_schema) != input_encoding_digest(attention_schema):
        raise ValueError("reference schemas have different input encodings")
    expected = oriented_digest(explicit_rows)
    if not explicit_rows or expected != oriented_digest(attention_rows):
        raise ValueError("references require identical oriented token/label/player populations")
    return expected


class AttentionReference(ResearchModel):
    def __init__(self, schema: AttentionCardSchema, *, expected_schema_sha256: str) -> None:
        nn.Module.__init__(self)
        if schema.fingerprint() != expected_schema_sha256:
            raise ValueError("attention reference requires its network-specific schema hash")
        self.schema = schema
        self.reference = AttentionMatchupModel(schema)

    def forward(self, tokens: Tensor, features: Tensor) -> Tensor:
        if features.shape != (len(tokens), 0):
            raise ValueError("attention reference cannot consume engineered or player features")
        return self.reference(tokens)

    def penalty(self, strength: float) -> Tensor:
        if not isfinite(strength) or strength < 0:
            raise ValueError("penalty must be finite and nonnegative")
        return next(self.reference.parameters()).new_zeros(())

    def parameter_groups(self, weight_decay: float) -> list[dict[str, object]]:
        return [{"params": list(self.reference.parameters()), "weight_decay": weight_decay}]
