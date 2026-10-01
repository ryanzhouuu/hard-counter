from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Self, cast

import numpy as np
import torch
from pydantic import Field, model_validator
from torch import Tensor

from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel
from experiments.common.artifacts import load_run
from experiments.common.calibration import sigmoid
from experiments.common.components import components
from experiments.common.contracts import PopulationIdentity, StudyConfig, Variant
from experiments.common.data_access import ResearchRow, RoleAccess
from experiments.common.fit import FitResult, predict_logits
from experiments.common.predictions import Prediction
from experiments.higher_order.features import extract as pattern_features
from experiments.matchup_features.model import ResearchModel
from experiments.mechanics.contracts import MechanicsCatalog
from experiments.mechanics.load import from_payload, to_payload
from experiments.player_adjustment.history import FrozenHistory
from experiments.player_adjustment.joint import FrozenPlayerEffects, JointPlayerEffects
from experiments.player_adjustment.model import PlayerModel


def predictions(
    rows: tuple[ResearchRow, ...], logits: np.ndarray, temperature: float
) -> tuple[Prediction, ...]:
    return tuple(
        Prediction(
            r.key,
            r.event_key,
            r.key[0],
            r.player_a,
            r.player_b,
            r.label,
            float(z),
            sigmoid(float(z), temperature),
            sigmoid(float(z)),
        )
        for r, z in zip(rows, logits, strict=True)
    )


def save_checkpoint(
    path: Path,
    config: StudyConfig,
    variant: Variant,
    schema: AttentionCardSchema,
    catalog: MechanicsCatalog,
    population: PopulationIdentity,
    names: tuple[str, ...],
    formulas: tuple[str, ...],
    fitted: FitResult,
    transform: dict[str, object],
    temperatures: tuple[float, ...],
) -> None:
    metadata = {
        "kind": "research-checkpoint:v1",
        "variant": variant.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "schema": schema.model_dump(mode="json"),
        "mechanics": to_payload(catalog),
        "feature_names": names,
        "formulas": formulas,
        "scales": tuple(float(s) for s in fitted.scales),
        "selected_epoch": fitted.selected_epoch,
        "transform": transform,
        "population": population.model_dump(mode="json"),
        "temperatures": temperatures,
    }
    torch.save(
        {
            "metadata": canonical_json_bytes(metadata).decode(),
            "state_dict": fitted.model.state_dict(),
        },
        path,
    )


class CheckpointMetadata(ManifestModel):
    kind: str
    variant: Variant
    config: StudyConfig
    input_schema: AttentionCardSchema = Field(alias="schema")
    mechanics: dict[str, object]
    feature_names: tuple[str, ...]
    formulas: tuple[str, ...]
    scales: tuple[float, ...]
    selected_epoch: int = Field(gt=0)
    transform: dict[str, object]
    population: PopulationIdentity
    temperatures: tuple[float, float]

    @model_validator(mode="after")
    def verify_transforms(self) -> Self:
        if len(self.feature_names) != len(self.formulas) or len(self.scales) != len(
            self.feature_names
        ):
            raise ValueError("checkpoint feature registry/scales mismatch")
        if len(set(self.feature_names)) != len(self.feature_names) or any(
            not isfinite(s) or s <= 0 for s in (*self.scales, *self.temperatures)
        ):
            raise ValueError("checkpoint transforms must be unique finite and positive")
        return self


@dataclass(frozen=True)
class Checkpoint:
    model: ResearchModel
    metadata: CheckpointMetadata
    catalog: MechanicsCatalog

    def logits(self, rows: tuple[ResearchRow, ...], *, actual: bool = False) -> np.ndarray:
        meta = self.metadata
        branch = meta.variant.nuisance
        if branch == "history":
            values = (
                FrozenHistory.model_validate(meta.transform["history"]).gaps(rows).reshape(-1, 1)
            )
        elif branch == "joint":
            assert isinstance(self.model, PlayerModel) and self.model.player_effects is not None
            values = self.model.player_effects.vocabulary.encode(rows)
        elif meta.config.study_id == "higher-order" and meta.variant.feature_groups:
            pattern = cast(dict[str, object], meta.transform["patterns"])
            values = np.asarray(
                [pattern_features(r.tokens, self.catalog).values for r in rows]
            ) * np.asarray(pattern["active"])
        else:
            recipe = components(meta.config, meta.variant, self.catalog, meta.input_schema, rows[0])
            values = recipe.builder(rows, rows)
        if isinstance(self.model, PlayerModel) and not actual:
            values = (
                np.zeros_like(values)
                if branch == "history"
                else np.full_like(values, -1)
                if branch == "joint"
                else values
            )
        return predict_logits(self.model, rows, values / np.asarray(meta.scales))


def load_checkpoint(path: Path) -> Checkpoint:
    manifest = load_run(path.parent)
    if manifest.status != "complete" or path.name != "checkpoint.pt":
        raise ValueError("checkpoint must belong to a completed research run")
    loaded = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
    metadata = CheckpointMetadata.model_validate_json(cast(str, loaded["metadata"]))
    if (
        metadata.kind != "research-checkpoint:v1"
        or metadata.config != manifest.config
        or metadata.population != manifest.population
    ):
        raise ValueError("checkpoint research contract does not match run")
    catalog = from_payload(metadata.mechanics)
    branch = metadata.variant.nuisance
    if branch == "joint":
        effects = JointPlayerEffects.from_frozen(
            FrozenPlayerEffects.model_validate(metadata.transform["joint"])
        )
        model: ResearchModel = PlayerModel(
            len(metadata.input_schema.identity_vocab), branch="joint", player_effects=effects
        )
    elif branch == "history":
        model = PlayerModel(len(metadata.input_schema.identity_vocab), branch="history")
    elif metadata.variant.architecture == "attention":
        from experiments.common.attention_reference import AttentionReference

        model = AttentionReference(
            metadata.input_schema, expected_schema_sha256=metadata.input_schema.fingerprint()
        )
    else:
        model = ResearchModel(
            len(metadata.input_schema.identity_vocab),
            len(metadata.feature_names),
            input_size=metadata.input_schema.input_size,
        )
    model.load_state_dict(cast(dict[str, Tensor], loaded["state_dict"]))
    model.eval()
    return Checkpoint(model, metadata, catalog)


def swap_error(model: ResearchModel, tokens: Tensor, feature: np.ndarray, nuisance: str) -> float:
    reversed_feature = feature[:, ::-1].copy() if nuisance == "joint" else -feature
    with torch.inference_mode():
        x = torch.tensor(feature, dtype=torch.float32, device=tokens.device)
        sx = torch.tensor(reversed_feature, dtype=torch.float32, device=tokens.device)
        return float((model(tokens, x) + model(tokens.flip(1), sx)).abs().max())


def write_support(directory: Path, access: RoleAccess, rows: tuple[ResearchRow, ...]) -> None:
    def covariates(population: tuple[ResearchRow, ...]) -> list[dict[str, object]]:
        return [
            {
                "key": r.key,
                "event_key": r.event_key,
                "player_a": r.player_a,
                "player_b": r.player_b,
                "tokens": r.tokens,
            }
            for r in population
        ]

    payload = {
        "protocol": access.protocol.model_dump(mode="json"),
        "refit": covariates(rows),
        "development": covariates(access.read("development", "compare")),
    }
    (directory / "support.json").write_bytes(canonical_json_bytes(payload))
