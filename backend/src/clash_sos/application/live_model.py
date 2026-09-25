"""Score a recent battle batch with one configured immutable matchup artifact."""

from collections.abc import Sequence
from dataclasses import dataclass
from json import loads
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from clash_sos.application.model_predict import predict_matchup
from clash_sos.domain.analytics import MatchupPrediction, PredictionProvenance, PredictionState
from clash_sos.domain.model_artifact import ModelArtifactManifest

DeckPair = tuple[tuple[str, ...], tuple[str, ...]]


class LiveModelError(ValueError):
    """The selected artifact cannot produce live deck estimates."""


@dataclass(frozen=True)
class LiveModelInfo:
    model_version: str
    dataset_version: str
    catalog_version: str
    training_era_id: str

    @property
    def provenance(self) -> PredictionProvenance:
        return PredictionProvenance(
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            card_catalog_version=self.catalog_version,
            balance_era_id=self.training_era_id,
        )


def _manifest_kind(artifact: Path) -> str:
    """Reject missing or non-object manifests before selecting a loader."""
    try:
        raw: object = loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LiveModelError("model_unavailable") from error
    if not isinstance(raw, dict):
        raise LiveModelError("model_unavailable")
    body = cast(dict[str, object], raw)
    kind = body.get("manifest_type")
    if not isinstance(kind, str):
        raise LiveModelError("model_unavailable")
    return kind


def score_live_decks(
    artifact: Path, pairs: Sequence[DeckPair]
) -> tuple[LiveModelInfo, tuple[MatchupPrediction, ...]]:
    """Keep June provenance while permitting explicit use on newer deck pairs."""
    kind = _manifest_kind(artifact)
    if kind == "matchup_attention":
        try:
            import torch

            from clash_sos.infrastructure.ml.attention_artifact_io import load_attention_artifact
        except ImportError as error:
            raise LiveModelError("missing_ml_runtime") from error
        try:
            manifest, schema, model = load_attention_artifact(artifact)
            tokens = [(schema.encode_deck(a), schema.encode_deck(b)) for a, b in pairs]
            probabilities: list[float] = []
            if tokens:
                with torch.inference_mode():
                    logits = model(torch.tensor(tokens, dtype=torch.long))
                    probabilities = [
                        float(value.item())
                        for value in torch.sigmoid(logits.to(dtype=torch.float64))
                    ]
        except (OSError, ValueError, RuntimeError) as error:
            raise LiveModelError("model_unavailable") from error
        info = LiveModelInfo(
            model_version=manifest.model_version,
            dataset_version=manifest.dataset_version,
            catalog_version=manifest.catalog_version,
            training_era_id=manifest.balance_era_id,
        )
        predictions = tuple(
            MatchupPrediction(
                state=PredictionState.AVAILABLE,
                side_a_win_probability=probability,
                provenance=info.provenance,
            )
            for probability in probabilities
        )
        return info, predictions
    try:
        manifest = ModelArtifactManifest.model_validate_json(
            (artifact / "manifest.json").read_bytes()
        )
        info = LiveModelInfo(
            model_version=manifest.model_version,
            dataset_version=manifest.dataset_version,
            catalog_version=manifest.catalog_version,
            training_era_id=manifest.balance_era_id,
        )
        predictions = tuple(predict_matchup(artifact, a, b) for a, b in pairs)
    except (OSError, ValueError, ValidationError) as error:
        raise LiveModelError("model_unavailable") from error
    return info, predictions
