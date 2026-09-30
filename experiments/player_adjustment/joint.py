"""Static player effects constrained within training opponent components."""

from collections.abc import Sequence
from math import isfinite
from typing import Self, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import model_validator
from torch import Tensor, nn

from clash_sos.domain.manifests import ManifestModel
from experiments.common.data_access import ResearchRow


class PlayerVocabulary(ManifestModel):
    players: tuple[str, ...]
    components: tuple[tuple[str, ...], ...]

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        if not self.players or list(self.players) != sorted(set(self.players)):
            raise ValueError("player vocabulary must be nonempty, unique, and sorted")
        members = [player for component in self.components for player in component]
        if sorted(members) != list(self.players) or any(not group for group in self.components):
            raise ValueError("components must partition the player vocabulary")
        return self

    def encode(self, rows: Sequence[ResearchRow]) -> NDArray[np.int64]:
        indices = {player: index for index, player in enumerate(self.players)}
        return np.asarray(
            [[indices.get(row.player_a, -1), indices.get(row.player_b, -1)] for row in rows],
            dtype=np.int64,
        ).reshape((-1, 2))


def training_vocabulary(rows: Sequence[ResearchRow]) -> PlayerVocabulary:
    parents = {player: player for row in rows for player in (row.player_a, row.player_b)}

    def root(player: str) -> str:
        while parents[player] != player:
            parents[player] = parents[parents[player]]
            player = parents[player]
        return player

    for row in rows:
        if not row.player_a or not row.player_b or row.player_a == row.player_b:
            raise ValueError("opponent graph requires distinct nonempty players")
        left, right = root(row.player_a), root(row.player_b)
        parents[max(left, right)] = min(left, right)
    grouped: dict[str, list[str]] = {}
    for player in sorted(parents):
        grouped.setdefault(root(player), []).append(player)
    return PlayerVocabulary(
        players=tuple(sorted(parents)),
        components=tuple(tuple(group) for _, group in sorted(grouped.items())),
    )


class FrozenPlayerEffects(ManifestModel):
    vocabulary: PlayerVocabulary
    effects: tuple[float, ...]

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if len(self.effects) != len(self.vocabulary.players) or not np.isfinite(self.effects).all():
            raise ValueError("effects must be finite and aligned to vocabulary")
        indices = {player: index for index, player in enumerate(self.vocabulary.players)}
        for component in self.vocabulary.components:
            mean = sum(self.effects[indices[player]] for player in component) / len(component)
            if abs(mean) > 1e-6:
                raise ValueError("effects must be centered within components")
        return self


class JointPlayerEffects(nn.Module):
    def __init__(self, vocabulary: PlayerVocabulary) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.effects = nn.Parameter(torch.zeros(len(vocabulary.players)))
        components = {
            player: index for index, group in enumerate(vocabulary.components) for player in group
        }
        self.register_buffer(
            "component_ids", torch.tensor([components[player] for player in vocabulary.players])
        )

    @classmethod
    def from_training(cls, rows: Sequence[ResearchRow]) -> Self:
        return cls(training_vocabulary(rows))

    def centered(self) -> Tensor:
        component_ids = cast(Tensor, self.component_ids)
        count = len(self.vocabulary.components)
        totals = self.effects.new_zeros(count).scatter_add(0, component_ids, self.effects)
        sizes = self.effects.new_zeros(count).scatter_add(
            0, component_ids, torch.ones_like(self.effects)
        )
        return self.effects - (totals / sizes)[component_ids]

    def forward(self, player_indices: Tensor) -> Tensor:
        if player_indices.ndim != 2 or player_indices.shape[1] != 2:
            raise ValueError("player indices require [batch, 2] shape")
        indices = player_indices.to(dtype=torch.long)
        if not torch.isfinite(player_indices).all() or not torch.equal(indices, player_indices):
            raise ValueError("player indices must be finite integers")
        if torch.any(indices < -1) or torch.any(indices >= len(self.vocabulary.players)):
            raise ValueError("player index outside fitted vocabulary")
        values = self.centered()[indices.clamp(min=0)] * (indices >= 0)
        return values[:, 0] - values[:, 1]

    def penalty(self, strength: float) -> Tensor:
        if not isfinite(strength) or strength < 0:
            raise ValueError("player penalty must be nonnegative")
        return self.centered().square().mean() * strength / 2

    def center_(self) -> None:
        with torch.no_grad():
            self.effects.copy_(self.centered())

    def freeze(self) -> FrozenPlayerEffects:
        values = [float(value.item()) for value in self.centered().detach().cpu()]
        return FrozenPlayerEffects(vocabulary=self.vocabulary, effects=tuple(values))

    @classmethod
    def from_frozen(cls, state: FrozenPlayerEffects) -> Self:
        model = cls(state.vocabulary)
        with torch.no_grad():
            model.effects.copy_(torch.tensor(state.effects))
        return model
