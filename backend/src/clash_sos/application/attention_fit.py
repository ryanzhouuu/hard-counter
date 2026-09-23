"""Select attention epochs on a watch slice, then refit from the same seed.

This module consumes validated cache batches. It does not publish artifacts or
inspect development, calibration, or reporting labels during fitting.
"""

import copy
import resource
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

import numpy as np
import torch
from pydantic import Field
from torch import Tensor

from clash_sos.domain.attention_cache import SliceRole
from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.manifests import ManifestModel
from clash_sos.infrastructure.ml.attention_runtime import (
    AttentionRuntime,
    initialize_attention_runtime,
)


class AttentionBatchSource(Protocol):
    """Expose bounded cache batches without granting access to row metadata."""

    def iter_batches(
        self, role: SliceRole, *, batch_size: int, seed: int = 0, epoch: int = 0
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield aligned token and label arrays for a declared slice."""
        ...


class AttentionFitConfig(ManifestModel):
    """Validated optimizer and stopping choices recorded with a fit."""

    batch_size: int = Field(default=256, gt=0)
    max_epochs: int = Field(default=20, gt=0)
    patience: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    device: str = "auto"
    time_limit_seconds: float | None = Field(default=None, gt=0)


@dataclass(frozen=True)
class AttentionEpoch:
    """Mean per-battle losses and fit support for one selection epoch."""

    epoch: int
    fit_loss: float
    watch_loss: float
    fit_rows: int
    watch_rows: int


@dataclass(frozen=True)
class AttentionFitResult:
    """Keep watch-selected weights distinct from full-partition refit weights."""

    selection_model: AttentionMatchupModel
    refit_model: AttentionMatchupModel
    history: tuple[AttentionEpoch, ...]
    selected_epoch: int
    selection_reason: str
    config: AttentionFitConfig
    runtime: AttentionRuntime
    elapsed_seconds: float
    trained_rows: int
    rows_per_second: float
    process_peak_rss_bytes: int


class AttentionFitTimeLimit(TimeoutError):
    """Stop without a result when the wall-time budget expires."""


def _check_deadline(deadline: float | None) -> float:
    """Return the checked time or fail when the fit budget has expired."""
    checked = monotonic()
    if deadline is not None and checked >= deadline:
        raise AttentionFitTimeLimit(
            "attention fit exceeded its time limit; no result was published"
        )
    return checked


def _batches(
    source: AttentionBatchSource,
    role: SliceRole,
    config: AttentionFitConfig,
    epoch: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Copy compact cache arrays into the selected runtime tensor dtypes."""
    for tokens, labels in source.iter_batches(
        role, batch_size=config.batch_size, seed=config.seed, epoch=epoch
    ):
        yield (
            torch.tensor(tokens, dtype=torch.long, device=config.device),
            torch.tensor(labels, dtype=torch.float32, device=config.device),
        )


def _run_epoch(
    model: AttentionMatchupModel,
    source: AttentionBatchSource,
    role: SliceRole,
    config: AttentionFitConfig,
    epoch: int,
    deadline: float | None,
    optimizer: torch.optim.AdamW | None,
) -> tuple[float, int]:
    """Aggregate loss by battle; only fit and refit roles update parameters."""
    model.train(optimizer is not None)
    total_loss, rows = 0.0, 0
    for tokens, labels in _batches(source, role, config, epoch):
        _check_deadline(deadline)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tokens), labels)
            if not torch.isfinite(loss).item():
                raise ValueError("attention fit encountered a nonfinite loss")
            loss.backward()  # type: ignore[reportUnknownMemberType]
            gradients = (p.grad for p in model.parameters() if p.grad is not None)
            if any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                raise ValueError("attention fit encountered a nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()  # type: ignore[reportUnknownMemberType]
            if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                raise ValueError("attention fit encountered a nonfinite parameter")
        else:
            with torch.inference_mode():
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tokens), labels)
            if not torch.isfinite(loss).item():
                raise ValueError("attention fit encountered a nonfinite watch loss")
        count = labels.numel()
        total_loss += float(loss.item()) * count
        rows += count
    if rows == 0:
        raise ValueError(f"attention {role} slice is empty")
    return total_loss / rows, rows


def _peak_rss_bytes() -> int:
    """Normalize the process high-water RSS units on macOS and Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def fit_attention_model(
    schema: AttentionCardSchema,
    source: AttentionBatchSource,
    config: AttentionFitConfig,
) -> AttentionFitResult:
    """Select on watch, then refit without reading any evaluation-only slice."""
    started = monotonic()
    runtime = initialize_attention_runtime(config.seed, config.device)
    effective = config.model_copy(update={"device": runtime.device})
    deadline = started + config.time_limit_seconds if config.time_limit_seconds else None
    model = AttentionMatchupModel(schema).to(runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[AttentionEpoch] = []
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    trained_rows = 0
    reason = "epoch_cap"
    for epoch in range(1, config.max_epochs + 1):
        fit_loss, fit_rows = _run_epoch(
            model, source, "selection_fit", effective, epoch, deadline, optimizer
        )
        watch_loss, watch_rows = _run_epoch(
            model, source, "watch", effective, epoch, deadline, None
        )
        trained_rows += fit_rows
        history.append(AttentionEpoch(epoch, fit_loss, watch_loss, fit_rows, watch_rows))
        if watch_loss < best_loss:
            best_loss = watch_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            reason = "patience"
            break
    if best_state is None:
        raise ValueError("attention fit did not produce a selectable checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    initialize_attention_runtime(config.seed, runtime.device)
    refit = AttentionMatchupModel(schema).to(runtime.device)
    refit_optimizer = torch.optim.AdamW(
        refit.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for epoch in range(1, best_epoch + 1):
        _, rows = _run_epoch(refit, source, "refit", effective, epoch, deadline, refit_optimizer)
        trained_rows += rows
    refit.eval()
    elapsed = _check_deadline(deadline) - started
    return AttentionFitResult(
        selection_model=model,
        refit_model=refit,
        history=tuple(history),
        selected_epoch=best_epoch,
        selection_reason=reason,
        config=config,
        runtime=runtime,
        elapsed_seconds=elapsed,
        trained_rows=trained_rows,
        rows_per_second=trained_rows / elapsed,
        process_peak_rss_bytes=_peak_rss_bytes(),
    )
