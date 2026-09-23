"""Check watch-only selection, deterministic refit, and fit failure paths."""

from collections.abc import Iterator

import numpy as np
import pytest
import torch

import clash_sos.application.attention_fit as attention_fit
from clash_sos.application.attention_fit import (
    AttentionFitConfig,
    AttentionFitTimeLimit,
    fit_attention_model,
)
from clash_sos.domain.attention_cache import SliceRole
from clash_sos.domain.attention_schema import (
    AttentionCardSchema,
    AttentionModelConfig,
    build_attention_schema,
)
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


class TinySource:
    """Expose fit roles and flag any attempt to inspect evaluation labels."""

    def __init__(self, *, invalid_label: bool = False, development_label: int = 0) -> None:
        a = np.arange(8, dtype=np.uint8)
        b = np.arange(4, 12, dtype=np.uint8)
        self.tokens = np.stack(([a, b], [b, a], [a, b], [b, a]))
        self.labels = np.array([1, 0, 1, 0], dtype=np.float32)
        if invalid_label:
            self.labels[0] = np.nan
        self.development_label = development_label
        self.roles: list[SliceRole] = []

    def iter_batches(
        self, role: SliceRole, *, batch_size: int, seed: int = 0, epoch: int = 0
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Keep development data reachable so forbidden reads are detectable."""
        self.roles.append(role)
        if role == "selection_fit":
            indices = np.array([0, 1, 2])
            np.random.default_rng(seed + epoch).shuffle(indices)
        elif role == "watch":
            indices = np.array([3])
        elif role == "refit":
            indices = np.arange(4)
            np.random.default_rng(seed + epoch).shuffle(indices)
        else:
            indices = np.array([0])
            self.labels[0] = self.development_label
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            yield self.tokens[selected], self.labels[selected]


@pytest.fixture
def schema() -> AttentionCardSchema:
    """Use a small explicit-only network for deterministic optimizer tests."""
    return build_attention_schema(
        KAGGLE_V6_CARDS.serialize(),
        attributes=CARD_ATTRIBUTES,
        network=AttentionModelConfig(neural_component=False),
    )


def test_fit_selects_minimum_watch_checkpoint_and_refits_from_seed(
    schema: AttentionCardSchema,
) -> None:
    config = AttentionFitConfig(batch_size=2, max_epochs=5, patience=2, seed=14, device="cpu")
    first_source = TinySource(development_label=0)
    first = fit_attention_model(schema, first_source, config)
    second_source = TinySource(development_label=1)
    second = fit_attention_model(schema, second_source, config)

    assert first.selected_epoch == min(first.history, key=lambda item: item.watch_loss).epoch
    assert first.selection_reason in {"patience", "epoch_cap"}
    assert first.history == second.history
    assert first_source.roles == second_source.roles
    assert set(first_source.roles) == {"selection_fit", "watch", "refit"}
    assert first.runtime.device == "cpu"
    assert first.process_peak_rss_bytes > 0
    assert first.rows_per_second > 0
    assert first.trained_rows == 3 * len(first.history) + 4 * first.selected_epoch
    for name, value in first.refit_model.state_dict().items():
        torch.testing.assert_close(value, second.refit_model.state_dict()[name], rtol=0, atol=0)

    watch = next(first_source.iter_batches("watch", batch_size=2))
    with torch.inference_mode():
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            first.selection_model(torch.tensor(watch[0], dtype=torch.long)),
            torch.tensor(watch[1]),
        )
    assert loss.item() == pytest.approx(min(item.watch_loss for item in first.history))


def test_fit_rejects_nonfinite_data_and_time_budget(schema: AttentionCardSchema) -> None:
    config = AttentionFitConfig(batch_size=2, max_epochs=2, seed=2, device="cpu")
    with pytest.raises(ValueError, match="nonfinite loss"):
        fit_attention_model(schema, TinySource(invalid_label=True), config)
    with pytest.raises(AttentionFitTimeLimit, match="no result was published"):
        fit_attention_model(
            schema,
            TinySource(),
            config.model_copy(update={"time_limit_seconds": 1e-12}),
        )
    with pytest.raises(ValueError, match="device must be"):
        fit_attention_model(schema, TinySource(), config.model_copy(update={"device": "gpu"}))


def test_fit_rejects_deadline_crossed_during_final_refit_batch(
    schema: AttentionCardSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    readings = iter((0.0, 0.2, 0.4, 0.6, 2.0))
    monkeypatch.setattr(attention_fit, "monotonic", lambda: next(readings))
    config = AttentionFitConfig(
        batch_size=4, max_epochs=1, patience=1, seed=2, device="cpu", time_limit_seconds=1
    )
    with pytest.raises(AttentionFitTimeLimit, match="no result was published"):
        fit_attention_model(schema, TinySource(), config)
