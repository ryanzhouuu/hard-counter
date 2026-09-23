"""Verify set behavior and gradient paths in the shared deck encoder."""

from typing import cast

import torch
from torch import Tensor

from clash_sos.domain.attention_layers import DeckEncoder, WithinDeckBlock
from clash_sos.domain.attention_schema import AttentionModelConfig, build_attention_schema
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def _encoder() -> tuple[DeckEncoder, dict[str, int]]:
    schema = build_attention_schema(
        KAGGLE_V6_CARDS.serialize(),
        attributes=CARD_ATTRIBUTES,
        network=AttentionModelConfig(embedding_width=16, feed_forward_width=32),
    )
    return DeckEncoder(schema), schema.identity_index


def test_deck_attention_is_permutation_equivariant() -> None:
    torch.manual_seed(7)  # type: ignore[reportUnknownMemberType]
    encoder, _ = _encoder()
    encoder.eval()
    deck = torch.arange(8).unsqueeze(0)
    order = torch.tensor([7, 0, 5, 2, 3, 1, 6, 4])
    normal = encoder(deck)
    permuted = encoder(deck[:, order])
    torch.testing.assert_close(permuted, normal[:, order], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(permuted.mean(dim=1), normal.mean(dim=1))


def test_forms_are_distinct_and_gradients_reach_embedding_and_attention() -> None:
    torch.manual_seed(11)  # type: ignore[reportUnknownMemberType]
    encoder, identities = _encoder()
    base = identities["knight:base"]
    evolution = identities["knight:evolution"]
    first = encoder.embedding(torch.tensor([[base]]))
    second = encoder.embedding(torch.tensor([[evolution]]))
    assert not torch.allclose(first, second)
    deck = torch.arange(8).unsqueeze(0)
    encoder(deck).square().mean().backward()
    assert encoder.embedding.base.weight.grad is not None
    assert encoder.embedding.form.weight.grad is not None
    assert encoder.embedding.attributes.weight.grad is not None
    attention = cast(WithinDeckBlock, encoder.blocks[0]).attention
    weight = cast(Tensor, attention.in_proj_weight)
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()
