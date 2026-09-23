"""Deck-only matchup logits with exact shared-parameter swap construction.

The explicit baseline and attention ablations use the same input schema. Training
and serving call the same forward pass; row metadata never enters the network.
"""

import torch
from torch import Tensor, nn

from clash_sos.domain.attention_interactions import ExplicitInteractions
from clash_sos.domain.attention_layers import CrossDeckBlock, DeckEncoder
from clash_sos.domain.attention_schema import AttentionCardSchema


class AttentionMatchupModel(nn.Module):
    """Return P(side A wins) logits under the deck-only equal-skill assumption."""

    def __init__(self, schema: AttentionCardSchema) -> None:
        super().__init__()
        config = schema.network
        self.identity_count = len(schema.identity_vocab)
        self.explicit = (
            ExplicitInteractions(self.identity_count) if config.explicit_interactions else None
        )
        self.deck_encoder = DeckEncoder(schema) if config.neural_component else None
        self.cross_blocks = nn.ModuleList(
            CrossDeckBlock(config)
            for _ in range(
                config.cross_deck_blocks
                if config.neural_component and config.cross_deck_attention
                else 0
            )
        )
        width = config.embedding_width
        self.ordered_head = (
            nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
            if config.neural_component
            else None
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Return one antisymmetric logit per [batch, two sides, eight cards]."""
        if tokens.ndim != 3 or tokens.shape[1:] != (2, 8):
            raise ValueError("attention inputs must have shape [batch, 2, 8]")
        side_a, side_b = tokens[:, 0], tokens[:, 1]
        score = self.explicit(side_a, side_b) if self.explicit is not None else None
        if self.deck_encoder is not None and self.ordered_head is not None:
            a = self.deck_encoder(side_a)
            b = self.deck_encoder(side_b)
            for block in self.cross_blocks:
                a, b = block(a, b)
            pooled_a, pooled_b = a.mean(dim=1), b.mean(dim=1)
            forward = self.ordered_head(torch.cat((pooled_a, pooled_b), dim=1)).squeeze(1)
            reverse = self.ordered_head(torch.cat((pooled_b, pooled_a), dim=1)).squeeze(1)
            neural_score = (forward - reverse) / 2
            score = neural_score if score is None else score + neural_score
        if score is None:
            raise RuntimeError("attention model has no enabled score component")
        return score
