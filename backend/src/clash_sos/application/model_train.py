"""Train, evaluate, and publish a card-log-odds matchup baseline artifact.

Fits only the temporal train partition. Evaluation scores temporal and
player-disjoint partitions. Destination versions are never overwritten.
"""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Literal

from pydantic import ValidationError

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.model_errors import KaggleV6ModelTrainError as KaggleV6ModelTrainError
from clash_sos.application.model_predict import predict_matchup as predict_matchup
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.matchup_baseline import (
    DEFAULT_MIRROR_SEED,
    DEFAULT_SMOOTHING_ALPHA,
    card_log_odds,
    exact_matchup_probability,
    predict_card_log_odds,
)
from clash_sos.domain.matchup_pair import CardPairPredictor
from clash_sos.domain.model_artifact import (
    DEFAULT_MODEL_VERSION,
    EVALUATION_PARTITIONS,
    EvaluationReport,
    ModelArtifactManifest,
    ModelOutputFile,
    ProbabilityMetricAccumulator,
    ProbabilityMetrics,
    SplitEvaluation,
    dump_evaluation_report,
    dump_model_manifest,
)
from clash_sos.domain.processed_manifest import (
    PlayerDisjointSplitManifest,
    ProcessedDatasetManifest,
    TemporalSplitManifest,
)
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
    require_partition_rows,
)

FileKind = Literal["card_catalog", "evaluation", "feature_schema", "predictor"]
SplitName = Literal["temporal", "player_disjoint"]


@dataclass(frozen=True)
class PartitionScores:
    """Held-out metrics for one split partition. card_pair is set when a pair model is scored."""

    prior: ProbabilityMetrics
    exact_matchup: ProbabilityMetrics
    card_log_odds: ProbabilityMetrics
    card_pair: ProbabilityMetrics | None = None


def kind_path(dataset: Path, manifest: ProcessedDatasetManifest, kind: str) -> Path:
    """Resolve an inventoried processed-dataset file under the published directory."""
    match = next(file for file in manifest.files if file.kind == kind)
    return dataset / match.path


def partition_row_count(
    split: TemporalSplitManifest | PlayerDisjointSplitManifest, partition: str
) -> int:
    """Return the processed-manifest row count for one split partition."""
    return next(item.row_count for item in split.partitions if item.partition == partition)


def write_canonical_json(path: Path, payload: object) -> None:
    """Write canonical JSON plus a trailing newline."""
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def inventory_model_file(
    path: Path, root: Path, kind: FileKind, chunk_size: int
) -> ModelOutputFile:
    """Hash an artifact file relative to the output workspace."""
    size_bytes, digest = hash_file(path, chunk_size)
    return ModelOutputFile(
        path=path.relative_to(root).as_posix(),
        kind=kind,
        size_bytes=size_bytes,
        sha256=digest,
    )


def require_publish_paths(destination: Path, output_workspace: Path) -> None:
    """Reject existing destinations and require a same-filesystem rename."""
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


def processed_training_paths(
    dataset: Path,
) -> tuple[ProcessedDatasetManifest, Path, Path, Path]:
    """Load a one-era processed dataset and resolve canonical plus split files."""
    processed = load_processed_dataset(dataset)
    if len(processed.accepted.eras) != 1:
        raise KaggleV6ModelTrainError("training requires exactly one accepted balance era")
    if processed.catalog_version != KAGGLE_V6_CARDS.version:
        raise KaggleV6ModelTrainError(
            "processed catalog version does not match the training catalog"
        )
    return (
        processed,
        kind_path(dataset, processed, "canonical"),
        kind_path(dataset, processed, "temporal_split"),
        kind_path(dataset, processed, "player_disjoint_split"),
    )


def score_partition(
    examples: Iterator[OrientedExample],
    *,
    effects: Mapping[str, float],
    matchup_counts: Mapping[tuple[str, str], tuple[int, int]],
    alpha: float,
    expected_rows: int | None = None,
    pair_predictor: CardPairPredictor | None = None,
) -> PartitionScores:
    """Score one partition in a single pass. expected_rows, when set, must match."""
    prior = ProbabilityMetricAccumulator()
    exact = ProbabilityMetricAccumulator()
    card = ProbabilityMetricAccumulator()
    pair = ProbabilityMetricAccumulator() if pair_predictor is not None else None
    for example in examples:
        prior.update(example.label, 0.5)
        exact.update(
            example.label,
            exact_matchup_probability(
                example.deck_a_hash, example.deck_b_hash, matchup_counts, alpha=alpha
            ),
        )
        card.update(
            example.label,
            predict_card_log_odds(example.side_a_keys, example.side_b_keys, effects),
        )
        if pair is not None and pair_predictor is not None:
            pair.update(
                example.label,
                pair_predictor.predict(example.side_a_keys, example.side_b_keys),
            )
    scores = PartitionScores(
        prior=prior.finalize(),
        exact_matchup=exact.finalize(),
        card_log_odds=card.finalize(),
        card_pair=pair.finalize() if pair is not None else None,
    )
    if expected_rows is not None and scores.prior.row_count != expected_rows:
        raise KaggleV6ModelTrainError(
            f"scored row count {scores.prior.row_count} != {expected_rows}"
        )
    return scores


def finalize_model_artifact(
    output_workspace: Path,
    destination: Path,
    *,
    processed: ProcessedDatasetManifest,
    model_version: str,
    smoothing_alpha: float,
    mirror_seed: int,
    chunk_size: int,
    predictor_name: str = "predictor.json",
) -> Path:
    """Inventory required files, write manifest.json, and rename onto destination.

    predictor_name is predictor.json for the logistic artifacts and predictor.txt
    for the LightGBM booster.
    """
    files = tuple(
        sorted(
            (
                inventory_model_file(
                    output_workspace / "card-catalog.json",
                    output_workspace,
                    "card_catalog",
                    chunk_size,
                ),
                inventory_model_file(
                    output_workspace / "evaluation.json",
                    output_workspace,
                    "evaluation",
                    chunk_size,
                ),
                inventory_model_file(
                    output_workspace / "feature-schema.json",
                    output_workspace,
                    "feature_schema",
                    chunk_size,
                ),
                inventory_model_file(
                    output_workspace / predictor_name,
                    output_workspace,
                    "predictor",
                    chunk_size,
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
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Fit the card-log-odds baseline, evaluate splits, and publish one artifact version.

    progress, when set, receives dataset partition counts and per-slice scoring updates.
    """
    require_publish_paths(destination, output_workspace)
    processed, canonical, temporal, player = processed_training_paths(dataset)
    if progress is not None:
        temporal_counts = " ".join(
            f"{item.partition}={item.row_count}" for item in processed.temporal_split.partitions
        )
        player_counts = " ".join(
            f"{item.partition}={item.row_count}"
            for item in processed.player_disjoint_split.partitions
        )
        progress(
            f"dataset {processed.dataset_version} temporal {temporal_counts} "
            f"player_disjoint {player_counts}"
        )
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = connect_staging_duckdb(
        memory_limit=config.memory_limit,
        threads=1,
        temp_directory=temp_directory,
    )
    output_workspace.mkdir(parents=True)
    try:
        try:
            train_rows = partition_row_count(processed.temporal_split, "train")
            require_partition_rows(
                connection,
                canonical_path=canonical,
                split_path=temporal,
                split="temporal",
                partition="train",
                seed=mirror_seed,
                expected_rows=train_rows,
            )
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
            split_files: tuple[
                tuple[SplitName, Path, TemporalSplitManifest | PlayerDisjointSplitManifest],
                ...,
            ] = (
                ("temporal", temporal, processed.temporal_split),
                ("player_disjoint", player, processed.player_disjoint_split),
            )
            for split_name, split_path, split_manifest in split_files:
                for partition in EVALUATION_PARTITIONS:
                    expected_rows = partition_row_count(split_manifest, partition)
                    if progress is not None:
                        progress(f"scoring {split_name} {partition} ({expected_rows} rows)")
                    if (split_name, partition) != ("temporal", "train"):
                        require_partition_rows(
                            connection,
                            canonical_path=canonical,
                            split_path=split_path,
                            split=split_name,
                            partition=partition,
                            seed=mirror_seed,
                            expected_rows=expected_rows,
                        )
                    scores = score_partition(
                        iter_oriented_examples(
                            connection,
                            canonical_path=canonical,
                            split_path=split_path,
                            partition=partition,
                            seed=mirror_seed,
                        ),
                        effects=effects,
                        matchup_counts=matchup_counts,
                        alpha=smoothing_alpha,
                        expected_rows=expected_rows,
                    )
                    evaluations.append(
                        SplitEvaluation(
                            split=split_name,
                            partition=partition,
                            prior=scores.prior,
                            exact_matchup=scores.exact_matchup,
                            card_log_odds=scores.card_log_odds,
                        )
                    )
        except KaggleV6TrainError as error:
            raise KaggleV6ModelTrainError(str(error)) from error
        finally:
            connection.close()

        identities = tuple(sorted(entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries))
        write_canonical_json(output_workspace / "predictor.json", {"effects": effects})
        write_canonical_json(
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
        return finalize_model_artifact(
            output_workspace,
            destination,
            processed=processed,
            model_version=model_version,
            smoothing_alpha=smoothing_alpha,
            mirror_seed=mirror_seed,
            chunk_size=config.chunk_size,
        )
    except BaseException:
        rmtree(output_workspace, ignore_errors=True)
        raise
