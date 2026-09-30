"""Separate matchup and observed-outcome scores for nuisance controls."""

from collections.abc import Iterable
from math import isfinite
from typing import Literal

import torch
from torch import Tensor, nn

from experiments.matchup_features.model import ResearchModel
from experiments.player_adjustment.joint import JointPlayerEffects

PlayerBranch = Literal["unadjusted", "history", "joint"]


class PlayerModel(ResearchModel):
    def __init__(
        self,
        identity_count: int,
        *,
        branch: PlayerBranch = "unadjusted",
        player_effects: JointPlayerEffects | None = None,
        input_size: int = 9,
    ) -> None:
        super().__init__(identity_count, 0, input_size=input_size)
        if branch not in ("unadjusted", "history", "joint"):
            raise ValueError("unknown player branch")
        if (branch == "joint") != (player_effects is not None):
            raise ValueError("joint branch requires a fitted player vocabulary")
        self.branch = branch
        self.player_effects = player_effects
        self.beta = nn.Parameter(torch.zeros(()), requires_grad=branch == "history")

    def matchup_logits(self, tokens: Tensor) -> Tensor:
        return self.explicit(tokens[:, 0], tokens[:, 1])

    def forward(self, tokens: Tensor, features: Tensor) -> Tensor:
        columns = {"unadjusted": 0, "history": 1, "joint": 2}[self.branch]
        if features.shape != (len(tokens), columns):
            raise ValueError("nuisance inputs do not match selected player branch")
        logits = self.matchup_logits(tokens)
        if self.branch == "history":
            return logits + self.beta * features[:, 0]
        if self.player_effects is not None:
            return logits + self.player_effects(features)
        return logits

    def nuisance_parameters(self) -> Iterable[nn.Parameter]:
        if self.branch == "history":
            return (self.beta,)
        if self.player_effects is not None:
            return (self.player_effects.effects,)
        return ()

    def nuisance_penalty(self, strength: float) -> Tensor:
        if not isfinite(strength) or strength < 0:
            raise ValueError("nuisance penalty must be nonnegative")
        if self.branch == "history":
            return self.beta.square() * strength / 2
        if self.player_effects is not None:
            return self.player_effects.penalty(strength)
        return self.beta.new_zeros(())

    def penalty(self, strength: float) -> Tensor:
        return self.nuisance_penalty(strength)

    def parameter_groups(self, weight_decay: float) -> list[dict[str, object]]:
        return [
            {"params": list(self.explicit.parameters()), "weight_decay": weight_decay},
            {"params": list(self.nuisance_parameters()), "weight_decay": 0.0},
        ]

    def center_nuisance(self) -> None:
        if self.player_effects is not None:
            self.player_effects.center_()
