"""Train, evaluate, and publish a card-log-odds matchup baseline artifact.

Fits only the temporal train partition. Evaluation scores temporal and
player-disjoint partitions. Destination versions are never overwritten.
"""

from collections.abc import Mapping, Sequence
from json import loads
from pathlib import Path
from shutil import rmtree
from typing import Literal, cast

from pydantic import ValidationError

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.analytics import MatchupPrediction, PredictionProvenance, PredictionState
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.matchup_baseline import (
    DEFAULT_MIRROR_SEED,
    DEFAULT_SMOOTHING_ALPHA,
    card_log_odds,
    exact_matchup_probability,
    predict_card_log_odds,
)
from clash_sos.domain.model_artifact import (
    DEFAULT_MODEL_VERSION,
    EVALUATION_PARTITIONS,
    EvaluationReport,
    ModelArtifactManifest,
    ModelOutputFile,
    ProbabilityMetrics,
    SplitEvaluation,
    brier_score,
    dump_evaluation_report,
    dump_model_manifest,
    expected_calibration_error,
    log_loss,
)
from clash_sos.domain.processed_manifest import ProcessedDatasetManifest
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.publish_io import (
    KaggleV6PublishError,
    publish_processed_version,
    require_same_filesystem,
)
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb
from clash_sos.infrastructure.kaggle_v6.train_io import (
    KaggleV6TrainError,
    OrientedExample,
    aggregate_card_counts,
    aggregate_matchup_counts,
    iter_oriented_examples,
)


class KaggleV6ModelTrainError(ValueError):
    pass


FileKind = Literal["card_catalog", "evaluation", "feature_schema", "predictor"]
SplitName = Literal["temporal", "player_disjoint"]


def _kind_path(dataset: Path, manifest: ProcessedDatasetManifest, kind: str) -> Path:
    """Resolve an inventoried processed-dataset file under the published directory."""
    match = next(file for file in manifest.files if file.kind == kind)
    return dataset / match.path


def _probability_metrics(
    labels: Sequence[int], probabilities: Sequence[float]
) -> ProbabilityMetrics:
    return ProbabilityMetrics(
        log_loss=log_loss(labels, probabilities),
        brier_score=brier_score(labels, probabilities),
        expected_calibration_error=expected_calibration_error(labels, probabilities),
        row_count=len(labels),
    )


def _score_partition(
    examples: Sequence[OrientedExample],
    *,
    effects: Mapping[str, float],
    matchup_counts: Mapping[tuple[str, str], tuple[int, int]],
    alpha: float,
) -> tuple[ProbabilityMetrics, ProbabilityMetrics, ProbabilityMetrics]:
    labels = [example.label for example in examples]
    prior = [0.5] * len(labels)
    exact = [
        exact_matchup_probability(
            example.deck_a_hash, example.deck_b_hash, matchup_counts, alpha=alpha
        )
        for example in examples
    ]
    card = [
        predict_card_log_odds(example.side_a_keys, example.side_b_keys, effects)
        for example in examples
    ]
    return (
        _probability_metrics(labels, prior),
        _probability_metrics(labels, exact),
        _probability_metrics(labels, card),
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _inventory(path: Path, root: Path, kind: FileKind, chunk_size: int) -> ModelOutputFile:
    size_bytes, digest = hash_file(path, chunk_size)
    return ModelOutputFile(
        path=path.relative_to(root).as_posix(),
        kind=kind,
        size_bytes=size_bytes,
        sha256=digest,
    )


def load_processed_dataset(dataset: Path) -> ProcessedDatasetManifest:
    """Load a published processed-dataset manifest. Does not re-verify hashes."""
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        raise KaggleV6ModelTrainError("processed manifest is required")
    try:
        return ProcessedDatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise KaggleV6ModelTrainError("processed manifest is invalid") from error


def predict_matchup(
    artifact: Path, side_a: Sequence[str], side_b: Sequence[str]
) -> MatchupPrediction:
    """Load a published artifact and predict P(side A wins) from identity keys."""
    manifest_path = artifact / "manifest.json"
    predictor_path = artifact / "predictor.json"
    if not manifest_path.is_file() or not predictor_path.is_file():
        raise KaggleV6ModelTrainError("model artifact is incomplete")
    manifest = ModelArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    payload = loads(predictor_path.read_text(encoding="utf-8"))
    raw_effects = payload["effects"]
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


def train_matchup_baseline(
    dataset: Path,
    destination: Path,
    *,
    output_workspace: Path,
    temp_directory: Path,
    config: StagingConfig,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    mirror_seed: int = DEFAULT_MIRROR_SEED,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> Path:
    """Fit the card-log-odds baseline, evaluate splits, and publish one artifact version."""
    if destination.exists():
        raise KaggleV6ModelTrainError("published model version already exists")
    if output_workspace.exists():
        raise KaggleV6ModelTrainError("output workspace already exists")
    output_workspace.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        require_same_filesystem(output_workspace.parent, destination.parent)
    except KaggleV6PublishError as error:
        raise KaggleV6ModelTrainError(str(error)) from error

    processed = load_processed_dataset(dataset)
    if len(processed.accepted.eras) != 1:
        raise KaggleV6ModelTrainError("training requires exactly one accepted balance era")
    canonical = _kind_path(dataset, processed, "canonical")
    temporal = _kind_path(dataset, processed, "temporal_split")
    player = _kind_path(dataset, processed, "player_disjoint_split")
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = connect_staging_duckdb(
        memory_limit=config.memory_limit,
        threads=1,
        temp_directory=temp_directory,
    )
    output_workspace.mkdir(parents=True)
    try:
        try:
            card_counts = aggregate_card_counts(
                connection,
                canonical_path=canonical,
                split_path=temporal,
                partition="train",
                seed=mirror_seed,
            )
            matchup_counts = aggregate_matchup_counts(
                connection,
                canonical_path=canonical,
                split_path=temporal,
                partition="train",
                seed=mirror_seed,
            )
            effects = {
                identity: card_log_odds(wins, trials, alpha=smoothing_alpha)
                for identity, (wins, trials) in card_counts.items()
            }
            evaluations: list[SplitEvaluation] = []
            split_files: tuple[tuple[SplitName, Path], ...] = (
                ("temporal", temporal),
                ("player_disjoint", player),
            )
            for split_name, split_path in split_files:
                for partition in EVALUATION_PARTITIONS:
                    examples = tuple(
                        iter_oriented_examples(
                            connection,
                            canonical_path=canonical,
                            split_path=split_path,
                            partition=partition,
                            seed=mirror_seed,
                        )
                    )
                    prior, exact, card = _score_partition(
                        examples,
                        effects=effects,
                        matchup_counts=matchup_counts,
                        alpha=smoothing_alpha,
                    )
                    evaluations.append(
                        SplitEvaluation(
                            split=split_name,
                            partition=partition,
                            prior=prior,
                            exact_matchup=exact,
                            card_log_odds=card,
                        )
                    )
        except KaggleV6TrainError as error:
            raise KaggleV6ModelTrainError(str(error)) from error
        finally:
            connection.close()

        identities = tuple(sorted(entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries))
        _write_json(output_workspace / "predictor.json", {"effects": effects})
        _write_json(
            output_workspace / "feature-schema.json",
            {
                "feature_schema_version": "card-log-odds:v1",
                "identities": identities,
                "smoothing_alpha": smoothing_alpha,
            },
        )
        (output_workspace / "card-catalog.json").write_bytes(KAGGLE_V6_CARDS.serialize() + b"\n")
        report = EvaluationReport(splits=tuple(evaluations))
        (output_workspace / "evaluation.json").write_bytes(dump_evaluation_report(report))
        files = tuple(
            sorted(
                (
                    _inventory(
                        output_workspace / "card-catalog.json",
                        output_workspace,
                        "card_catalog",
                        config.chunk_size,
                    ),
                    _inventory(
                        output_workspace / "evaluation.json",
                        output_workspace,
                        "evaluation",
                        config.chunk_size,
                    ),
                    _inventory(
                        output_workspace / "feature-schema.json",
                        output_workspace,
                        "feature_schema",
                        config.chunk_size,
                    ),
                    _inventory(
                        output_workspace / "predictor.json",
                        output_workspace,
                        "predictor",
                        config.chunk_size,
                    ),
                ),
                key=lambda file: file.path,
            )
        )
        manifest = ModelArtifactManifest(
            model_version=model_version,
            dataset_version=processed.dataset_version,
            catalog_version=processed.catalog_version,
            balance_era_id=processed.accepted.eras[0].era_id,
            smoothing_alpha=smoothing_alpha,
            mirror_seed=mirror_seed,
            files=files,
        )
        (output_workspace / "manifest.json").write_bytes(dump_model_manifest(manifest))
        try:
            return publish_processed_version(output_workspace, destination)
        except KaggleV6PublishError as error:
            raise KaggleV6ModelTrainError(str(error)) from error
    except BaseException:
        rmtree(output_workspace, ignore_errors=True)
        raise
