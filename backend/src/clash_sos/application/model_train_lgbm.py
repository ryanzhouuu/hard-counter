"""Train and publish a LightGBM deck-summary matchup artifact.

Fits temporal train only. The latest slice of that partition chooses the round
count, then the booster is refit on the full partition. Published probabilities
use a zero skill gap and average both deck orientations.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from shutil import rmtree
from typing import Literal, Protocol, cast

import numpy as np
from scipy.sparse import csr_matrix

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.model_train import (
    KaggleV6ModelTrainError,
    SplitName,
    finalize_model_artifact,
    partition_row_count,
    processed_training_paths,
    require_publish_paths,
    write_canonical_json,
)
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES, SUMMARY_COLUMNS, CardAttributeTable
from clash_sos.domain.matchup_baseline import (
    DEFAULT_MIRROR_SEED,
    DEFAULT_SMOOTHING_ALPHA,
    card_log_odds,
    exact_matchup_probability,
    predict_card_log_odds,
)
from clash_sos.domain.matchup_lgbm import (
    CARDS_PER_SIDE,
    DEFAULT_BAGGING_FRACTION,
    DEFAULT_BAGGING_FREQ,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_FEATURE_FRACTION,
    DEFAULT_LAMBDA_L2,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MIN_DATA_IN_LEAF,
    DEFAULT_NUM_LEAVES,
    DEFAULT_NUM_THREADS,
    DEFAULT_SEED,
    DEFAULT_WATCH_FRACTION,
    LIGHTGBM_FEATURE_SCHEMA_VERSION,
    LIGHTGBM_PRESENCE_SCHEMA_VERSION,
    PresenceRow,
    PresenceSchema,
    lightgbm_training_params,
    skill_gap_then_observe,
    symmetrized_probability,
    watch_row_count,
)
from clash_sos.domain.model_artifact import (
    DEFAULT_LIGHTGBM_MODEL_VERSION,
    EVALUATION_PARTITIONS,
    EvaluationReport,
    ProbabilityMetricAccumulator,
    SplitEvaluation,
    dump_evaluation_report,
)
from clash_sos.domain.player_skill import DEFAULT_SKILL_ALPHA, PlayerSkillTracker
from clash_sos.domain.processed_manifest import (
    PlayerDisjointSplitManifest,
    TemporalSplitManifest,
)
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb
from clash_sos.infrastructure.kaggle_v6.train_io import (
    KaggleV6TrainError,
    OrientedExample,
    aggregate_card_counts,
    aggregate_matchup_counts,
    iter_oriented_examples,
    require_partition_rows,
)

PREDICTOR_NAME = "predictor.txt"
SCORE_BATCH_ROWS = 8192


def _row_nonzero_budget(schema: PresenceSchema) -> int:
    """Return the most values one legal row stores.

    That is eight cards on each side, both summary blocks when the schema has
    them, and the skill gap.
    """
    return CARDS_PER_SIDE * 2 + schema.summary_width * 2 + 1


class _Booster(Protocol):
    """The LightGBM operations this trainer uses."""

    @property
    def best_iteration(self) -> int: ...

    def current_iteration(self) -> int: ...

    def num_feature(self) -> int: ...

    def predict(self, data: csr_matrix) -> Sequence[float]: ...

    def save_model(self, filename: str) -> None: ...


def _lightgbm() -> object:
    """Import LightGBM when a booster is trained or loaded."""
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise KaggleV6ModelTrainError(
            "lightgbm is required for this model; install the dev dependency group"
        ) from error
    return lgb


def _as_booster(value: object) -> _Booster:
    return cast(_Booster, value)


def stream_presence_matrix(
    examples: Iterator[OrientedExample],
    schema: PresenceSchema,
    tracker: PlayerSkillTracker,
    row_count: int,
) -> tuple[csr_matrix, np.ndarray]:
    """Fill a CSR matrix in example order without retaining the example objects.

    Each row stores at most eight cards per side, the deck summaries when the
    schema has them, and the skill gap. The gap is recorded before that battle
    updates the tracker.
    """
    if row_count < 1:
        raise KaggleV6ModelTrainError("training rows are required")
    budget = _row_nonzero_budget(schema)
    capacity = row_count * budget
    indices = np.empty(capacity, dtype=np.int32)
    data = np.empty(capacity, dtype=np.float32)
    indptr = np.zeros(row_count + 1, dtype=np.int64)
    labels = np.empty(row_count, dtype=np.float64)
    cursor = 0
    seen = 0
    for example in examples:
        row = schema.row(
            example.side_a_keys,
            example.side_b_keys,
            skill_diff=skill_gap_then_observe(
                tracker,
                example.side_a_player_id,
                example.side_b_player_id,
                example.label,
            ),
        )
        width = len(row.indices)
        if width > budget or cursor + width > capacity:
            raise KaggleV6ModelTrainError("training row exceeds the presence budget")
        indices[cursor : cursor + width] = row.indices
        data[cursor : cursor + width] = row.values
        cursor += width
        seen += 1
        indptr[seen] = cursor
        labels[seen - 1] = example.label
    if seen != row_count:
        raise KaggleV6ModelTrainError(f"streamed row count {seen} != {row_count}")
    matrix = csr_matrix(
        (data[:cursor], indices[:cursor], indptr),
        shape=(row_count, schema.feature_count),
    )
    return matrix, labels


def _csr_from_rows(rows: Sequence[PresenceRow], feature_count: int) -> csr_matrix:
    indptr = np.empty(len(rows) + 1, dtype=np.int64)
    indptr[0] = 0
    total = 0
    for index, row in enumerate(rows, start=1):
        total += len(row.indices)
        indptr[index] = total
    indices = np.empty(total, dtype=np.int32)
    data = np.empty(total, dtype=np.float32)
    cursor = 0
    for row in rows:
        width = len(row.indices)
        indices[cursor : cursor + width] = row.indices
        data[cursor : cursor + width] = row.values
        cursor += width
    return csr_matrix((data, indices, indptr), shape=(len(rows), feature_count))


class LightGBMPresencePredictor:
    """Equal-skill matchup probability from a fitted presence booster."""

    def __init__(self, schema: PresenceSchema, booster: _Booster) -> None:
        if booster.num_feature() != schema.feature_count:
            raise ValueError("booster feature count does not match identities")
        self.schema = schema
        self._booster = booster

    def predict(self, side_a: Sequence[str], side_b: Sequence[str]) -> float:
        """Return P(side A wins) with the skill gap set to zero."""
        return self.predict_many(((side_a, side_b),))[0]

    def predict_many(self, sides: Sequence[tuple[Sequence[str], Sequence[str]]]) -> list[float]:
        """Score a batch. Each pair is oriented both ways and then averaged."""
        if not sides:
            return []
        forward = [self.schema.row(side_a, side_b, skill_diff=0.0) for side_a, side_b in sides]
        backward = [self.schema.row(side_b, side_a, skill_diff=0.0) for side_a, side_b in sides]
        raw_forward = self._probabilities(forward)
        raw_backward = self._probabilities(backward)
        return [
            symmetrized_probability(left, right)
            for left, right in zip(raw_forward, raw_backward, strict=True)
        ]

    def _probabilities(self, rows: Sequence[PresenceRow]) -> list[float]:
        raw = np.asarray(self._booster.predict(_csr_from_rows(rows, self.schema.feature_count)))
        return [float(value) for value in raw.reshape(-1)]


def predict_lightgbm_artifact(
    predictor_path: Path,
    schema_payload: Mapping[str, object],
    side_a: Sequence[str],
    side_b: Sequence[str],
) -> float:
    """Load predictor.txt and return the equal-skill symmetrized probability."""
    schema = _schema_from_payload(schema_payload)
    booster = _as_booster(_lightgbm().Booster(model_file=str(predictor_path)))  # type: ignore[attr-defined]
    return LightGBMPresencePredictor(schema, booster).predict(side_a, side_b)


def _identity_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("feature schema identities must be a list of strings")
    items: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            raise ValueError("feature schema identities must be a list of strings")
        items.append(item)
    if not items:
        raise ValueError("feature schema identities are required")
    return tuple(items)


def _schema_from_payload(schema_payload: Mapping[str, object]) -> PresenceSchema:
    """Build the column layout stored in a presence or deck-summary artifact."""
    identities = _identity_list(schema_payload.get("identities"))
    version = schema_payload.get("feature_schema_version")
    if version == LIGHTGBM_FEATURE_SCHEMA_VERSION:
        return PresenceSchema(
            identities, CardAttributeTable.from_payload(schema_payload.get("attributes"))
        )
    if version == LIGHTGBM_PRESENCE_SCHEMA_VERSION:
        return PresenceSchema(identities)
    raise ValueError("unsupported lightgbm feature schema")


def _fit_booster(
    matrix: csr_matrix,
    labels: np.ndarray,
    params: dict[str, object],
    *,
    max_rounds: int,
    early_stopping_rounds: int,
    watch_rows: int,
) -> tuple[_Booster, int]:
    """Choose a round count on the tail, then refit that many rounds on every row."""
    lgb = _lightgbm()
    fit_end = int(labels.shape[0]) - watch_rows
    fit_set = lgb.Dataset(matrix[:fit_end], label=labels[:fit_end])  # type: ignore[attr-defined]
    watch_set = lgb.Dataset(  # type: ignore[attr-defined]
        matrix[fit_end:],
        label=labels[fit_end:],
        reference=fit_set,
    )
    watched = _as_booster(
        lgb.train(  # type: ignore[attr-defined]
            params,
            fit_set,
            num_boost_round=max_rounds,
            valid_sets=[watch_set],
            valid_names=["watch"],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],  # type: ignore[attr-defined]
        )
    )
    best_iteration = watched.best_iteration
    if best_iteration < 1:
        best_iteration = watched.current_iteration()
    if best_iteration < 1:
        raise KaggleV6ModelTrainError("lightgbm did not complete a boosting round")
    refit = _as_booster(
        lgb.train(  # type: ignore[attr-defined]
            params,
            lgb.Dataset(matrix, label=labels),  # type: ignore[attr-defined]
            num_boost_round=best_iteration,
        )
    )
    return refit, best_iteration


def _score_partition(
    examples: Iterator[OrientedExample],
    *,
    split: SplitName,
    partition: Literal["train", "validation", "test"],
    effects: Mapping[str, float],
    matchup_counts: Mapping[tuple[str, str], tuple[int, int]],
    alpha: float,
    predictor: LightGBMPresencePredictor,
    expected_rows: int,
) -> SplitEvaluation:
    """Score comparators and the equal-skill booster in one pass. Omits card-pair."""
    prior = ProbabilityMetricAccumulator()
    exact = ProbabilityMetricAccumulator()
    card = ProbabilityMetricAccumulator()
    boosted = ProbabilityMetricAccumulator()
    batch: list[tuple[Sequence[str], Sequence[str]]] = []
    batch_labels: list[int] = []

    def flush() -> None:
        if not batch_labels:
            return
        for label, probability in zip(batch_labels, predictor.predict_many(batch), strict=True):
            boosted.update(label, probability)
        batch.clear()
        batch_labels.clear()

    for example in examples:
        prior.update(example.label, 0.5)
        exact.update(
            example.label,
            exact_matchup_probability(
                example.deck_a_hash,
                example.deck_b_hash,
                matchup_counts,
                alpha=alpha,
            ),
        )
        card.update(
            example.label,
            predict_card_log_odds(example.side_a_keys, example.side_b_keys, effects),
        )
        batch.append((example.side_a_keys, example.side_b_keys))
        batch_labels.append(example.label)
        if len(batch_labels) >= SCORE_BATCH_ROWS:
            flush()
    flush()
    scored = prior.finalize()
    if scored.row_count != expected_rows:
        raise KaggleV6ModelTrainError(f"scored row count {scored.row_count} != {expected_rows}")
    return SplitEvaluation(
        split=split,
        partition=partition,
        prior=scored,
        exact_matchup=exact.finalize(),
        card_log_odds=card.finalize(),
        lightgbm=boosted.finalize(),
    )


def train_lightgbm_model(
    dataset: Path,
    destination: Path,
    *,
    output_workspace: Path,
    temp_directory: Path,
    config: StagingConfig,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    mirror_seed: int = DEFAULT_MIRROR_SEED,
    model_version: str = DEFAULT_LIGHTGBM_MODEL_VERSION,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_leaves: int = DEFAULT_NUM_LEAVES,
    min_data_in_leaf: int = DEFAULT_MIN_DATA_IN_LEAF,
    feature_fraction: float = DEFAULT_FEATURE_FRACTION,
    bagging_fraction: float = DEFAULT_BAGGING_FRACTION,
    bagging_freq: int = DEFAULT_BAGGING_FREQ,
    lambda_l2: float = DEFAULT_LAMBDA_L2,
    num_threads: int = DEFAULT_NUM_THREADS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    watch_fraction: float = DEFAULT_WATCH_FRACTION,
    seed: int = DEFAULT_SEED,
    skill_alpha: float = DEFAULT_SKILL_ALPHA,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Fit the deck-summary booster, evaluate splits, and publish one artifact version.

    DuckDB stays on one thread. num_threads applies only to LightGBM. progress,
    when set, receives dataset counts, the chosen round count, and scoring lines.
    """
    if skill_alpha <= 0:
        raise KaggleV6ModelTrainError("skill alpha must be positive")
    if max_rounds < 1:
        raise KaggleV6ModelTrainError("max rounds must be positive")
    if early_stopping_rounds < 1:
        raise KaggleV6ModelTrainError("early stopping rounds must be positive")
    require_publish_paths(destination, output_workspace)
    processed, canonical, temporal, player = processed_training_paths(dataset)
    train_rows = partition_row_count(processed.temporal_split, "train")
    identities = tuple(sorted(entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries))
    schema = PresenceSchema(identities, CARD_ATTRIBUTES)
    try:
        watch_rows = watch_row_count(train_rows, watch_fraction)
        params = lightgbm_training_params(
            identity_count=len(identities),
            summary_count=len(SUMMARY_COLUMNS),
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            lambda_l2=lambda_l2,
            num_threads=num_threads,
            seed=seed,
        )
    except ValueError as error:
        raise KaggleV6ModelTrainError(str(error)) from error
    if progress is not None:
        progress(
            f"dataset {processed.dataset_version} temporal-train={train_rows} "
            f"watch_rows={watch_rows}"
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
                progress(f"presence matrix {train_rows} rows")
            matrix, labels = stream_presence_matrix(
                iter_oriented_examples(
                    connection,
                    canonical_path=canonical,
                    split_path=temporal,
                    partition="train",
                    seed=mirror_seed,
                    order_by_time=True,
                ),
                schema,
                PlayerSkillTracker(alpha=skill_alpha),
                train_rows,
            )
            if progress is not None:
                progress("boosting")
            booster, best_iteration = _fit_booster(
                matrix,
                labels,
                params,
                max_rounds=max_rounds,
                early_stopping_rounds=early_stopping_rounds,
                watch_rows=watch_rows,
            )
            del matrix, labels
            predictor = LightGBMPresencePredictor(schema, booster)
            if progress is not None:
                progress(f"refit rounds={best_iteration}")
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
                    evaluations.append(
                        _score_partition(
                            iter_oriented_examples(
                                connection,
                                canonical_path=canonical,
                                split_path=split_path,
                                partition=partition,
                                seed=mirror_seed,
                            ),
                            split=split_name,
                            partition=partition,
                            effects=effects,
                            matchup_counts=matchup_counts,
                            alpha=smoothing_alpha,
                            predictor=predictor,
                            expected_rows=expected_rows,
                        )
                    )
            booster.save_model(str(output_workspace / PREDICTOR_NAME))
            write_canonical_json(
                output_workspace / "feature-schema.json",
                {
                    "bagging_fraction": bagging_fraction,
                    "bagging_freq": bagging_freq,
                    "best_iteration": best_iteration,
                    "early_stopping_rounds": early_stopping_rounds,
                    "feature_fraction": feature_fraction,
                    "attribute_version": CARD_ATTRIBUTES.version,
                    "attributes": CARD_ATTRIBUTES.to_payload(),
                    "feature_schema_version": LIGHTGBM_FEATURE_SCHEMA_VERSION,
                    "identities": list(identities),
                    "lambda_l2": lambda_l2,
                    "learning_rate": learning_rate,
                    "max_rounds": max_rounds,
                    "min_data_in_leaf": min_data_in_leaf,
                    "num_leaves": num_leaves,
                    "num_threads": num_threads,
                    "seed": seed,
                    "skill_alpha": skill_alpha,
                    "skill_column": schema.skill_column,
                    "skill_control": "past_laplace",
                    "smoothing_alpha": smoothing_alpha,
                    "watch_fraction": watch_fraction,
                },
            )
            (output_workspace / "card-catalog.json").write_bytes(
                KAGGLE_V6_CARDS.serialize() + b"\n"
            )
            report = EvaluationReport(promoted_model="lightgbm", splits=tuple(evaluations))
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
            predictor_name=PREDICTOR_NAME,
        )
    except BaseException:
        rmtree(output_workspace, ignore_errors=True)
        raise
