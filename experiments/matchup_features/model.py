import torch
from torch import Tensor, nn

from clash_sos.domain.attention_interactions import ExplicitInteractions


class ResearchModel(nn.Module):
    """Explicit baseline plus zero-initialized antisymmetric engineered effects."""

    def __init__(self, identity_count: int, feature_count: int, *, input_size: int = 9) -> None:
        super().__init__()
        if feature_count < 0:
            raise ValueError("feature count must be nonnegative")
        self.explicit = ExplicitInteractions(identity_count, input_size=input_size)
        self.feature_weights = nn.Parameter(torch.zeros(feature_count))

    def forward(self, tokens: Tensor, features: Tensor) -> Tensor:
        if features.shape != (len(tokens), self.feature_weights.numel()):
            raise ValueError("features must match the registered ordered columns")
        return self.explicit(tokens[:, 0], tokens[:, 1]) + features @ self.feature_weights

    def penalty(self, strength: float) -> Tensor:
        if strength < 0:
            raise ValueError("feature penalty must be nonnegative")
        return strength * self.feature_weights.square().sum() / 2

    def parameter_groups(self, weight_decay: float) -> list[dict[str, object]]:
        return [
            {"params": list(self.explicit.parameters()), "weight_decay": weight_decay},
            {"params": [self.feature_weights], "weight_decay": 0.0},
        ]

    def center_nuisance(self) -> None:
        pass
