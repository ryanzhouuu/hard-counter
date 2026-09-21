"""Train, evaluate, and publish the antisymmetric card-pair logistic artifact.

Fits temporal train only. Scores the same 2x3 grid as the card-log-odds baseline
and promotes the pair model. Destination versions are never overwritten.
"""

from collections.abc import Callable
from pathlib import Path
from shutil import rmtree

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.model_train import (
    KaggleV6ModelTrainError,
    SplitName,
    finalize_model_artifact,
    partition_row_count,
    processed_training_paths,
    require_publish_paths,
    score_partition,
    write_canonical_json,
)
from clash_sos.domain.matchup_baseline import (
    DEFAULT_MIRROR_SEED,
    DEFAULT_SMOOTHING_ALPHA,
    card_log_odds,
)
from clash_sos.domain.matchup_pair import (
    DEFAULT_PAIR_EPOCHS,
    DEFAULT_PAIR_INIT_SCALE,
    DEFAULT_PAIR_L2,
    DEFAULT_PAIR_LEARNING_RATE,
    PAIR_FEATURE_SCHEMA_VERSION,
    accumulate_pair_counts,
    initialize_card_pair_predictor,
)
from clash_sos.domain.model_artifact import (
    DEFAULT_PAIR_MODEL_VERSION,
    EVALUATION_PARTITIONS,
    EvaluationReport,
    SplitEvaluation,
    dump_evaluation_report,
)
from clash_sos.domain.processed_manifest import (
    PlayerDisjointSplitManifest,
    TemporalSplitManifest,
)
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb
from clash_sos.infrastructure.kaggle_v6.train_io import (
    KaggleV6TrainError,
    aggregate_card_counts,
    aggregate_matchup_counts,
    iter_oriented_examples,
    require_partition_rows,
)


def train_card_pair_model(
    dataset: Path,
    destination: Path,
    *,
    output_workspace: Path,
    temp_directory: Path,
    config: StagingConfig,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    mirror_seed: int = DEFAULT_MIRROR_SEED,
    model_version: str = DEFAULT_PAIR_MODEL_VERSION,
    learning_rate: float = DEFAULT_PAIR_LEARNING_RATE,
    l2: float = DEFAULT_PAIR_L2,
    epochs: int = DEFAULT_PAIR_EPOCHS,
    pair_init_scale: float = DEFAULT_PAIR_INIT_SCALE,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Fit the card-pair logistic, evaluate splits, and publish one artifact version.

    progress, when set, receives dataset counts, pair-count/SGD updates, and scoring lines.
    """
    if epochs < 1:
        raise KaggleV6ModelTrainError("epochs must be positive")
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
    identities = tuple(sorted(entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries))
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
            if progress is not None:
                progress("counting pairs")
            pair_counts = accumulate_pair_counts(
                (example.label, example.side_a_keys, example.side_b_keys)
                for example in iter_oriented_examples(
                    connection,
                    canonical_path=canonical,
                    split_path=temporal,
                    partition="train",
                    seed=mirror_seed,
                )
            )
            predictor = initialize_card_pair_predictor(
                identities,
                card_counts,
                pair_counts,
                alpha=smoothing_alpha,
                pair_init_scale=pair_init_scale,
            )
            for epoch in range(epochs):
                if progress is not None:
                    progress(f"sgd epoch {epoch + 1}/{epochs}")
                for example in iter_oriented_examples(
                    connection,
                    canonical_path=canonical,
                    split_path=temporal,
                    partition="train",
                    seed=mirror_seed,
                ):
                    predictor.sgd_step(
                        example.side_a_keys,
                        example.side_b_keys,
                        label=example.label,
                        learning_rate=learning_rate,
                        l2=l2,
                    )
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
                        pair_predictor=predictor,
                    )
                    evaluations.append(
                        SplitEvaluation(
                            split=split_name,
                            partition=partition,
                            prior=scores.prior,
                            exact_matchup=scores.exact_matchup,
                            card_log_odds=scores.card_log_odds,
                            card_pair=scores.card_pair,
                        )
                    )
            write_canonical_json(output_workspace / "predictor.json", predictor.to_payload())
            write_canonical_json(
                output_workspace / "feature-schema.json",
                {
                    "epochs": epochs,
                    "feature_schema_version": PAIR_FEATURE_SCHEMA_VERSION,
                    "identities": list(identities),
                    "l2": l2,
                    "learning_rate": learning_rate,
                    "pair_init_scale": pair_init_scale,
                    "smoothing_alpha": smoothing_alpha,
                },
            )
            (output_workspace / "card-catalog.json").write_bytes(
                KAGGLE_V6_CARDS.serialize() + b"\n"
            )
            report = EvaluationReport(promoted_model="card_pair", splits=tuple(evaluations))
            (output_workspace / "evaluation.json").write_bytes(dump_evaluation_report(report))
        except KaggleV6TrainError as error:
            raise KaggleV6ModelTrainError(str(error)) from error
        finally:
            connection.close()

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
