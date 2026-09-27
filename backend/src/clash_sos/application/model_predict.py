"""Load published matchup artifacts and dispatch prediction by schema version.

Training implementations stay behind lazy imports so legacy and future model
families can share this entry point without creating application import cycles.
"""

from collections.abc import Sequence
from json import loads
from pathlib import Path
from typing import cast

from clash_sos.application.model_errors import KaggleV6ModelTrainError
from clash_sos.domain.analytics import MatchupPrediction, PredictionProvenance, PredictionState
from clash_sos.domain.matchup_baseline import predict_card_log_odds
from clash_sos.domain.matchup_lgbm import LIGHTGBM_SCHEMA_VERSIONS
from clash_sos.domain.matchup_pair import PAIR_FEATURE_SCHEMA_VERSION, CardPairPredictor
from clash_sos.domain.model_artifact import ModelArtifactManifest


def _json_object(path: Path) -> dict[str, object]:
    """Load a predictor JSON object or reject the malformed artifact."""
    payload = loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise KaggleV6ModelTrainError("model artifact is invalid")
    return cast(dict[str, object], payload)


def predict_matchup(
    artifact: Path,
    side_a: Sequence[str],
    side_b: Sequence[str],
    *,
    balance_era_id: str | None = None,
    side_a_tower: str | None = None,
    side_b_tower: str | None = None,
) -> MatchupPrediction:
    """Predict P(side A wins); attention calls must attest the matchup era."""
    manifest_path = artifact / "manifest.json"
    schema_path = artifact / "feature-schema.json"
    if not manifest_path.is_file() or not schema_path.is_file():
        raise KaggleV6ModelTrainError("model artifact is incomplete")
    manifest_body = _json_object(manifest_path)
    if manifest_body.get("manifest_type") == "matchup_attention":
        if balance_era_id is None:
            raise KaggleV6ModelTrainError("attention predictions require a balance era ID")
        try:
            import torch

            from clash_sos.infrastructure.ml.attention_artifact_io import load_attention_artifact
        except ImportError as error:
            from clash_sos.infrastructure.ml.attention_runtime import ML_INSTALL_COMMAND

            raise KaggleV6ModelTrainError(
                f"attention models require the 'ml' extra; run `{ML_INSTALL_COMMAND}`"
            ) from error
        try:
            manifest, attention_schema, model = load_attention_artifact(artifact)
        except ValueError as error:
            raise KaggleV6ModelTrainError(str(error)) from error
        if balance_era_id != manifest.balance_era_id:
            raise KaggleV6ModelTrainError("attention artifact does not cover the supplied era")
        try:
            tokens = (
                attention_schema.encode_side(side_a, tower=side_a_tower),
                attention_schema.encode_side(side_b, tower=side_b_tower),
            )
        except ValueError as error:
            raise KaggleV6ModelTrainError(str(error)) from error
        with torch.inference_mode():
            logits = model(torch.tensor([tokens], dtype=torch.long))
            probability = float(torch.sigmoid(logits.to(dtype=torch.float64))[0].item())
        return MatchupPrediction(
            state=PredictionState.AVAILABLE,
            side_a_win_probability=probability,
            provenance=PredictionProvenance(
                model_version=manifest.model_version,
                dataset_version=manifest.dataset_version,
                card_catalog_version=manifest.catalog_version,
                balance_era_id=manifest.balance_era_id,
            ),
        )
    manifest = ModelArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    schema_payload = loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema_payload, dict):
        raise KaggleV6ModelTrainError("model artifact is invalid")
    schema = cast(dict[str, object], schema_payload)
    predictor_path = artifact / next(
        file.path for file in manifest.files if file.kind == "predictor"
    )
    if not predictor_path.is_file():
        raise KaggleV6ModelTrainError("model artifact is incomplete")
    if schema.get("feature_schema_version") in LIGHTGBM_SCHEMA_VERSIONS:
        from clash_sos.application.model_train_lgbm import predict_lightgbm_artifact

        try:
            probability = predict_lightgbm_artifact(predictor_path, schema, side_a, side_b)
        except ValueError as error:
            raise KaggleV6ModelTrainError(str(error)) from error
    elif schema.get("feature_schema_version") == PAIR_FEATURE_SCHEMA_VERSION:
        predictor_body = _json_object(predictor_path)
        try:
            pair_model = CardPairPredictor.from_payload(predictor_body)
        except ValueError as error:
            raise KaggleV6ModelTrainError(str(error)) from error
        probability = pair_model.predict(side_a, side_b)
    else:
        predictor_body = _json_object(predictor_path)
        raw_effects = predictor_body.get("effects")
        if not isinstance(raw_effects, dict):
            raise KaggleV6ModelTrainError("predictor effects must be an object")
        effects: dict[str, float] = {}
        for key, value in cast(dict[object, object], raw_effects).items():
            if not isinstance(key, str) or isinstance(value, bool):
                raise KaggleV6ModelTrainError("predictor effects must map identities to numbers")
            if not isinstance(value, int | float):
                raise KaggleV6ModelTrainError("predictor effects must map identities to numbers")
            effects[key] = float(value)
        probability = predict_card_log_odds(side_a, side_b, effects)
    return MatchupPrediction(
        state=PredictionState.AVAILABLE,
        side_a_win_probability=probability,
        provenance=PredictionProvenance(
            model_version=manifest.model_version,
            dataset_version=manifest.dataset_version,
            card_catalog_version=manifest.catalog_version,
            balance_era_id=manifest.balance_era_id,
        ),
    )
