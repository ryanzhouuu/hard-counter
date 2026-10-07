"""Independent shrinkage for matchup weights and removable player effects."""

from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from torch import Tensor

from clash_sos.domain.manifests import ManifestModel
from experiments.common.data_access import ResearchRow
from experiments.common.fit import FeatureBuilder, ModelFactory
from experiments.player_adjustment.history import training_history
from experiments.player_adjustment.joint import JointPlayerEffects
from experiments.player_adjustment.model import PlayerModel


class ModelSpec(ManifestModel):
    capacity: Literal["main", "pairs"] = "pairs"
    nuisance: Literal["unadjusted", "history", "joint"] = "unadjusted"
    card_l2: float = Field(default=0, ge=0, allow_inf_nan=False)
    pair_l2: float = Field(default=0, ge=0, allow_inf_nan=False)
    player_l2: float = Field(default=0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def applicable_penalties(self) -> "ModelSpec":
        if self.capacity == "main" and self.pair_l2:
            raise ValueError("main-effects models have no pair penalty")
        if self.nuisance == "unadjusted" and self.player_l2:
            raise ValueError("unadjusted models have no player penalty")
        return self


class DiagnosticModel(PlayerModel):
    def __init__(self, count: int, spec: ModelSpec, rows: tuple[ResearchRow, ...]) -> None:
        super().__init__(
            count,
            branch=spec.nuisance,
            player_effects=JointPlayerEffects.from_training(rows)
            if spec.nuisance == "joint"
            else None,
            input_size=len(rows[0].tokens[0]),
        )
        self.spec = spec
        if spec.capacity == "main":
            self.explicit.own_pair.requires_grad_(False)
            self.explicit.opposing_pair.requires_grad_(False)

    def penalty(self, strength: float) -> Tensor:
        """Use lambda * sum(weight squared) / 2, independent of vocabulary size."""
        if strength != 0:
            raise ValueError("diagnostic shrinkage is bound by ModelSpec, not the fit penalty")
        value = self.spec.card_l2 * self.explicit.card_strength.square().sum()
        value = value + self.spec.pair_l2 * (
            self.explicit.own_pair.square().sum() + self.explicit.opposing_pair.square().sum()
        )
        if self.branch == "history":
            value = value + self.spec.player_l2 * self.beta.square()
        elif self.player_effects is not None:
            value = value + self.spec.player_l2 * self.player_effects.centered().square().sum()
        return value / 2


def recipe(spec: ModelSpec, count: int) -> tuple[FeatureBuilder, ModelFactory]:
    def features(training: tuple[ResearchRow, ...], rows: tuple[ResearchRow, ...]) -> np.ndarray:
        if spec.nuisance == "history":
            gaps, frozen = training_history(training)
            return (gaps if training is rows else frozen.gaps(rows)).reshape(-1, 1)
        if spec.nuisance == "joint":
            return JointPlayerEffects.from_training(training).vocabulary.encode(rows)
        return np.empty((len(rows), 0))

    def factory(training: tuple[ResearchRow, ...], columns: int) -> DiagnosticModel:
        return DiagnosticModel(count, spec, training)

    return features, factory
