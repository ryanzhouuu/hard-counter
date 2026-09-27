"""Score a recent battle batch with one configured immutable matchup artifact."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from json import loads
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from clash_sos.application.model_predict import predict_matchup
from clash_sos.domain.analytics import MatchupPrediction, PredictionProvenance, PredictionState
from clash_sos.domain.canonical import RecordIssue
from clash_sos.domain.card_catalog import CardCatalog
from clash_sos.domain.model_artifact import ModelArtifactManifest

DeckPair = tuple[tuple[str, ...], tuple[str, ...]]
TowerPair = tuple[str | None, str | None]


class LiveModelError(ValueError):
    """The selected artifact cannot produce live deck estimates."""


@dataclass(frozen=True)
class LiveModelInfo:
    model_version: str
    dataset_version: str
    catalog_version: str
    training_era_id: str
    input_scope: str = "deck_only"

    @property
    def provenance(self) -> PredictionProvenance:
        return PredictionProvenance(
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            card_catalog_version=self.catalog_version,
            balance_era_id=self.training_era_id,
        )


def _unavailable_prediction() -> MatchupPrediction:
    return MatchupPrediction(
        state=PredictionState.UNAVAILABLE, issue=RecordIssue.UNAVAILABLE_MODEL_COVERAGE
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
    artifact: Path,
    pairs: Sequence[DeckPair],
    *,
    tower_pairs: Sequence[TowerPair] | None = None,
) -> tuple[LiveModelInfo, tuple[MatchupPrediction, ...]]:
    """Retain training provenance and exclude pairs outside the frozen vocabulary."""
    if tower_pairs is not None and len(tower_pairs) != len(pairs):
        raise ValueError("tower pairs must align with deck pairs")
    kind = _manifest_kind(artifact)
    if kind == "matchup_attention":
        try:
            import torch

            from clash_sos.infrastructure.ml.attention_artifact_io import load_attention_artifact
        except ImportError as error:
            raise LiveModelError("missing_ml_runtime") from error
        try:
            manifest, schema, model = load_attention_artifact(artifact)
            vocabulary = set(schema.identity_vocab)
            with_towers = schema.tower_catalog is not None
            tower_vocabulary: set[str] = (
                {entry.identity for entry in schema.tower_catalog.entries}
                if schema.tower_catalog is not None
                else set()
            )
            towers = tower_pairs if tower_pairs is not None else [(None, None)] * len(pairs)
            supported = [
                index
                for index, (a, b) in enumerate(pairs)
                if set(a).issubset(vocabulary)
                and set(b).issubset(vocabulary)
                and (not with_towers or all(tower in tower_vocabulary for tower in towers[index]))
            ]
            tokens = [
                (
                    schema.encode_side(pairs[index][0], tower=towers[index][0]),
                    schema.encode_side(pairs[index][1], tower=towers[index][1]),
                )
                for index in supported
            ]
            probabilities: list[float | None] = [None] * len(pairs)
            if tokens:
                with torch.inference_mode():
                    logits = model(torch.tensor(tokens, dtype=torch.long))
                    for index, value in zip(
                        supported, torch.sigmoid(logits.to(dtype=torch.float64)), strict=True
                    ):
                        probabilities[index] = float(value.item())
        except (OSError, ValueError, RuntimeError) as error:
            raise LiveModelError("model_unavailable") from error
        info = LiveModelInfo(
            model_version=manifest.model_version,
            dataset_version=manifest.dataset_version,
            catalog_version=manifest.catalog_version,
            training_era_id=manifest.balance_era_id,
            input_scope="deck_and_tower" if with_towers else "deck_only",
        )
        predictions = tuple(
            MatchupPrediction(
                state=PredictionState.AVAILABLE,
                side_a_win_probability=probability,
                provenance=info.provenance,
            )
            if probability is not None
            else _unavailable_prediction()
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
        catalog_file = next(file for file in manifest.files if file.kind == "card_catalog")
        catalog_bytes = (artifact / catalog_file.path).read_bytes()
        if (
            len(catalog_bytes) != catalog_file.size_bytes
            or sha256(catalog_bytes).hexdigest() != catalog_file.sha256
        ):
            raise LiveModelError("model_unavailable")
        catalog = CardCatalog.from_payload(loads(catalog_bytes))
        if catalog.version != manifest.catalog_version:
            raise LiveModelError("model_unavailable")
        vocabulary = {entry.card.identity_key for entry in catalog.entries}
        predictions = tuple(
            predict_matchup(artifact, a, b)
            if set(a).issubset(vocabulary) and set(b).issubset(vocabulary)
            else _unavailable_prediction()
            for a, b in pairs
        )
    except (OSError, ValueError, ValidationError) as error:
        raise LiveModelError("model_unavailable") from error
    return info, predictions
