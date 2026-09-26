"""Trainable card effects shared by the explicit baseline and attention model.

Within-deck weights are symmetric; opposing-card weights are antisymmetric.
The latter have no diagonal parameter, so swapping decks negates the score.
"""

from typing import cast

import torch
from torch import Tensor, nn


class ExplicitInteractions(nn.Module):
    """Score two eight-card decks using card, synergy, and counter effects."""

    def __init__(self, identity_count: int, *, input_size: int = 8) -> None:
        super().__init__()
        if identity_count < 2:
            raise ValueError("at least two card identities are required")
        pair_count = identity_count * (identity_count - 1) // 2
        self.identity_count = identity_count
        self.input_size = input_size
        self.card_strength = nn.Parameter(torch.zeros(identity_count))
        self.own_pair = nn.Parameter(torch.zeros(pair_count))
        self.opposing_pair = nn.Parameter(torch.zeros(pair_count))
        positions = torch.triu_indices(input_size, input_size, offset=1)
        self.register_buffer("pair_left", positions[0], persistent=False)
        self.register_buffer("pair_right", positions[1], persistent=False)

    def _pair_index(self, left: Tensor, right: Tensor) -> Tensor:
        """Map an unordered identity pair to one row-major upper-triangle slot."""
        lower = torch.minimum(left, right)
        upper = torch.maximum(left, right)
        return lower * (2 * self.identity_count - lower - 1) // 2 + upper - lower - 1

    def _own_score(self, deck: Tensor) -> Tensor:
        """Count each unordered pair once, including own card/tower synergies."""
        strength = self.card_strength[deck].sum(dim=1)
        left = cast(Tensor, self.pair_left)
        right = cast(Tensor, self.pair_right)
        pairs = self._pair_index(deck[:, left], deck[:, right])
        return strength + self.own_pair[pairs].sum(dim=1)

    def forward(self, side_a: Tensor, side_b: Tensor) -> Tensor:
        """Return antisymmetric matchup logits for aligned deck batches."""
        if side_a.shape != side_b.shape or side_a.ndim != 2 or side_a.shape[1] != self.input_size:
            raise ValueError(f"opposing sides must have matching [batch, {self.input_size}] shapes")
        left = side_a.unsqueeze(2)
        right = side_b.unsqueeze(1)
        indices = self._pair_index(left, right)
        sign = torch.sign(left - right).neg()
        counter = (self.opposing_pair[indices] * sign).sum(dim=(1, 2))
        return self._own_score(side_a) - self._own_score(side_b) + counter
