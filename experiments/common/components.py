from dataclasses import dataclass

import numpy as np

from clash_sos.domain.attention_schema import AttentionCardSchema
from experiments.common.attention_reference import AttentionReference
from experiments.common.contracts import StudyConfig, Variant
from experiments.common.data_access import ResearchRow
from experiments.common.fit import FeatureBuilder, ModelFactory
from experiments.form_mechanics.features import extract as form_features
from experiments.higher_order.features import eligibility
from experiments.higher_order.features import extract as pattern_features
from experiments.matchup_features.features import extract as response_features
from experiments.matchup_features.model import ResearchModel
from experiments.mechanics.contracts import FeatureResult, MechanicsCatalog, MechanicsUnavailable
from experiments.player_adjustment.history import training_history
from experiments.player_adjustment.joint import JointPlayerEffects
from experiments.player_adjustment.model import PlayerModel
from experiments.tower_mechanics.features import extract as tower_features


@dataclass(frozen=True)
class Components:
    builder: FeatureBuilder
    factory: ModelFactory
    names: tuple[str, ...]
    formulas: tuple[str, ...]
    scale: bool


def components(
    config: StudyConfig,
    variant: Variant,
    catalog: MechanicsCatalog,
    schema: AttentionCardSchema,
    example: ResearchRow,
) -> Components:
    count = len(schema.identity_vocab)
    if variant.architecture == "attention":

        def empty(training: tuple[ResearchRow, ...], rows: tuple[ResearchRow, ...]) -> np.ndarray:
            return np.empty((len(rows), 0))

        def attention(training: tuple[ResearchRow, ...], columns: int) -> ResearchModel:
            return AttentionReference(schema, expected_schema_sha256=schema.fingerprint())

        return Components(empty, attention, (), (), True)
    if variant.nuisance == "history":

        def history(training: tuple[ResearchRow, ...], rows: tuple[ResearchRow, ...]) -> np.ndarray:
            gaps, state = training_history(training)
            return (gaps if rows is training else state.gaps(rows)).reshape(-1, 1)

        def history_model(training: tuple[ResearchRow, ...], columns: int) -> ResearchModel:
            return PlayerModel(count, branch="history")

        return Components(
            history, history_model, ("past_history_gap",), ("frozen past rating A minus B",), False
        )
    if variant.nuisance == "joint":

        def joint(training: tuple[ResearchRow, ...], rows: tuple[ResearchRow, ...]) -> np.ndarray:
            return JointPlayerEffects.from_training(training).vocabulary.encode(rows)

        def joint_model(training: tuple[ResearchRow, ...], columns: int) -> ResearchModel:
            return PlayerModel(
                count, branch="joint", player_effects=JointPlayerEffects.from_training(training)
            )

        return Components(
            joint,
            joint_model,
            ("player_a_index", "player_b_index"),
            ("fitted A vocabulary", "fitted B vocabulary"),
            False,
        )
    groups = variant.feature_groups

    def extractor(row: ResearchRow) -> FeatureResult:
        if not groups:
            return FeatureResult((), (), ())
        if config.study_id == "response-cycle":
            return response_features(
                row.tokens,
                catalog,
                response="response" in groups,
                cost="cycle" in groups,
            )
        if config.study_id == "form-mechanics":
            return form_features(row.tokens, catalog, inherited="inherited" in groups)
        if config.study_id == "tower-mechanics":
            return tower_features(row.tokens, catalog, quantitative="quantitative" in groups)
        if config.study_id == "higher-order":
            return pattern_features(row.tokens, catalog)
        raise ValueError("unregistered feature study")

    registry = extractor(example)

    def features(training: tuple[ResearchRow, ...], rows: tuple[ResearchRow, ...]) -> np.ndarray:
        values = np.array([extractor(row).values for row in rows], dtype=np.float64).reshape(
            len(rows), len(registry.names)
        )
        if config.study_id == "higher-order" and groups:
            support = eligibility(
                [extractor(row) for row in training],
                role="refit",
                minimum_support=config.rules.minimum_pattern_support,
            )
            if not any(support.active):
                raise MechanicsUnavailable("all registered patterns are inactive in training")
            values *= np.asarray(support.active)
        return values

    def model(training: tuple[ResearchRow, ...], columns: int) -> ResearchModel:
        return ResearchModel(count, columns, input_size=schema.input_size)

    return Components(features, model, registry.names, registry.formulas, True)
