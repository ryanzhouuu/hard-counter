"""Manifest contract for immutable deck-only attention artifacts.

The protocol identifies the training population; all published members are
inventoried separately from the manifest so reload can verify their bytes.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.attention_schema import PROBABILITY_INTERPRETATION
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel, RelativePath, Sha256, validate_relative_path

DEFAULT_ATTENTION_MODEL_VERSION = "kaggle-v6-ranked16-attention-v1"
REQUIRED_ATTENTION_FILES = {
    "card-catalog.json": "card_catalog",
    "feature-schema.json": "feature_schema",
    "evaluation.json": "evaluation",
    "weights.pt": "weights",
}


class AttentionArtifactFile(ManifestModel):
    """A checked published member; prediction files may be numerous."""

    path: RelativePath
    kind: Literal["card_catalog", "feature_schema", "evaluation", "weights", "predictions"]
    size_bytes: int = Field(ge=0)
    sha256: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Confine every member to a relative artifact path."""
        validate_relative_path(self.path)
        return self


class AttentionArtifactRuntime(ManifestModel):
    """Record the selected accelerator and interpreter used for fitting."""

    device: Literal["cpu", "mps"]
    python_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)


class AttentionArtifactManifest(ManifestModel):
    """Bind model weights to one card schema, fit population, and evaluation."""

    manifest_type: Literal["matchup_attention"] = "matchup_attention"
    manifest_version: Literal[1] = 1
    model_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    catalog_version: str = Field(min_length=1)
    balance_era_id: str = Field(min_length=1)
    encoding_sha256: Sha256
    fit_protocol: AttentionProtocol
    fit_seed: int = Field(ge=0, le=2**32 - 1)
    runtime: AttentionArtifactRuntime
    probability_interpretation: Literal[
        "deck-only matchup estimate under an equal-skill assumption"
    ] = PROBABILITY_INTERPRETATION
    files: tuple[AttentionArtifactFile, ...]

    @model_validator(mode="after")
    def validate_identity_and_files(self) -> Self:
        """Reject mixed populations, duplicate paths, and incomplete inventories."""
        protocol = self.fit_protocol
        if (
            self.dataset_version != protocol.dataset_version
            or self.balance_era_id != protocol.balance_era_id
            or self.encoding_sha256 != protocol.encoding_sha256
        ):
            raise ValueError("artifact and fit protocol identities disagree")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("artifact file paths must be unique and sorted")
        declared = {item.path: item.kind for item in self.files}
        if any(declared.get(path) != kind for path, kind in REQUIRED_ATTENTION_FILES.items()):
            raise ValueError("attention artifact is missing a required file kind")
        if any(
            path not in REQUIRED_ATTENTION_FILES
            and (
                kind != "predictions"
                or not path.startswith("evaluations/")
                or not path.endswith(".parquet")
            )
            for path, kind in declared.items()
        ):
            raise ValueError("attention artifact contains an unexpected member")
        return self


def dump_attention_manifest(manifest: AttentionArtifactManifest) -> bytes:
    """Serialize the checked inventory in canonical form with a final newline."""
    return canonical_json_bytes(manifest.model_dump(mode="python")) + b"\n"
