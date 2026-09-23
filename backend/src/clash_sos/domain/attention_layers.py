"""Shared set encoder for eight-card decks.

Identity lookup stays frozen; learned base/form embeddings and attention layers
receive gradients. No layer uses source card position as a feature.
"""

from typing import cast

import torch
from torch import Tensor, nn

from clash_sos.domain.attention_schema import AttentionCardSchema, AttentionModelConfig


class CardTokenEmbedding(nn.Module):
    """Combine base identity, form, and fixed attributes for each card token."""

    def __init__(self, schema: AttentionCardSchema) -> None:
        super().__init__()
        width = schema.network.embedding_width
        self.base = nn.Embedding(len(schema.base_vocab), width)
        self.form = nn.Embedding(len(schema.form_vocab), width)
        self.attributes = nn.Linear(len(schema.attribute_columns), width, bias=False)
        self.register_buffer(
            "base_index", torch.tensor(schema.token_base_indices, dtype=torch.long)
        )
        self.register_buffer(
            "form_index", torch.tensor(schema.token_form_indices, dtype=torch.long)
        )
        self.register_buffer(
            "attribute_values", torch.tensor(schema.token_attributes, dtype=torch.float32)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Look up the schema's frozen token features on the module's device."""
        base_index = cast(Tensor, self.base_index)
        form_index = cast(Tensor, self.form_index)
        attribute_values = cast(Tensor, self.attribute_values)
        return (
            self.base(base_index[tokens])
            + self.form(form_index[tokens])
            + self.attributes(attribute_values[tokens])
        )


class WithinDeckBlock(nn.Module):
    """Update each card from the other cards in the same unordered deck."""

    def __init__(self, config: AttentionModelConfig) -> None:
        super().__init__()
        width = config.embedding_width
        self.attention = nn.MultiheadAttention(
            width, config.attention_heads, dropout=config.dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, config.feed_forward_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feed_forward_width, width),
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, cards: Tensor) -> Tensor:
        """Preserve token permutation equivariance through shared attention."""
        update, _ = self.attention(cards, cards, cards, need_weights=False)
        cards = self.attention_norm(cards + update)
        return self.output_norm(cards + self.feed_forward(cards))


class DeckEncoder(nn.Module):
    """Use one parameter set for either side of a matchup."""

    def __init__(self, schema: AttentionCardSchema) -> None:
        super().__init__()
        self.embedding = CardTokenEmbedding(schema)
        config = schema.network
        self.blocks = nn.ModuleList(
            WithinDeckBlock(config)
            for _ in range(config.within_deck_blocks if config.within_deck_attention else 0)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Return contextual card vectors; the caller chooses how to pool."""
        cards = self.embedding(tokens)
        for block in self.blocks:
            cards = block(cards)
        return cards
