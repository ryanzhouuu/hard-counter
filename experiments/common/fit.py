import resource
import sys
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

import numpy as np
import torch

from clash_sos.infrastructure.ml.attention_runtime import initialize_attention_runtime
from experiments.common.batches import aligned_batches, rms_scales
from experiments.common.contracts import OptimizerConfig
from experiments.common.data_access import ResearchRow, RoleAccess
from experiments.matchup_features.model import ResearchModel

FeatureBuilder = Callable[[tuple[ResearchRow, ...], tuple[ResearchRow, ...]], np.ndarray]
ModelFactory = Callable[[tuple[ResearchRow, ...], int], ResearchModel]


@dataclass(frozen=True)
class FitResult:
    model: ResearchModel
    selected_epoch: int
    scales: np.ndarray
    selection_scales: np.ndarray
    watch_losses: tuple[float, ...]
    elapsed_seconds: float
    rows_per_second: float
    peak_memory_bytes: int
    device: str


def fit_research(
    access: RoleAccess,
    features: FeatureBuilder,
    factory: ModelFactory,
    config: OptimizerConfig,
    *,
    seed: int,
    penalty: float,
    scale_features: bool = True,
) -> FitResult:
    """Watch chooses epochs; transforms/maps and weights restart from refit only."""
    started = monotonic()
    deadline = started + config.time_limit_seconds
    runtime = initialize_attention_runtime(seed, config.device)
    fit_rows = access.read("selection_fit", "fit")
    watch_rows = access.read("watch", "fit")
    values = features(fit_rows, fit_rows)
    scales = rms_scales(values) if scale_features else np.ones(values.shape[1])
    watch_values = features(fit_rows, watch_rows) / scales
    model = factory(fit_rows, values.shape[1]).to(runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(config.weight_decay), lr=config.learning_rate
    )
    losses: list[float] = []
    trained = 0
    best, selected = float("inf"), 0

    def epoch(
        current: ResearchModel,
        rows: tuple[ResearchRow, ...],
        x: np.ndarray,
        number: int,
        updating: torch.optim.AdamW | None,
    ) -> float:
        nonlocal trained
        current.train(updating is not None)
        total = 0.0
        for batch in aligned_batches(
            rows,
            x,
            batch_size=config.batch_size,
            shuffle=updating is not None,
            seed=seed,
            epoch=number,
        ):
            if monotonic() >= deadline:
                raise TimeoutError("research time limit; run incomplete")
            tokens = torch.tensor(batch.tokens, dtype=torch.long, device=runtime.device)
            inputs = torch.tensor(batch.features, device=runtime.device)
            labels = torch.tensor(batch.labels, device=runtime.device)
            with torch.set_grad_enabled(updating is not None):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    current(tokens, inputs), labels
                )
                if not torch.isfinite(loss).item():
                    raise ValueError("nonfinite research loss")
                total += float(loss.item()) * len(labels)
                if updating is not None:
                    updating.zero_grad(set_to_none=True)
                    objective = loss + current.penalty(penalty)
                    objective.backward()  # type: ignore[reportUnknownMemberType]
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in current.parameters()
                    ):
                        raise ValueError("nonfinite research gradient")
                    torch.nn.utils.clip_grad_norm_(current.parameters(), config.gradient_clip_norm)
                    updating.step()  # type: ignore[reportUnknownMemberType]
                    current.center_nuisance()
                    if any(not torch.isfinite(p).all() for p in current.parameters()):
                        raise ValueError("nonfinite research parameter")
                    trained += len(labels)
        return total / len(rows)

    for number in range(1, config.max_epochs + 1):
        epoch(model, fit_rows, values / scales, number, optimizer)
        loss = epoch(model, watch_rows, watch_values, number, None)
        losses.append(loss)
        if loss < best:
            best, selected = loss, number
        elif number - selected >= config.patience:
            break
    refit_rows = access.read("refit", "fit")
    refit_values = features(refit_rows, refit_rows)
    refit_scales = rms_scales(refit_values) if scale_features else np.ones(refit_values.shape[1])
    initialize_attention_runtime(seed, runtime.device)
    refit = factory(refit_rows, refit_values.shape[1]).to(runtime.device)
    refit_optimizer = torch.optim.AdamW(
        refit.parameter_groups(config.weight_decay), lr=config.learning_rate
    )
    for number in range(1, selected + 1):
        epoch(refit, refit_rows, refit_values / refit_scales, number, refit_optimizer)
    refit.eval()
    elapsed = monotonic() - started
    if elapsed >= config.time_limit_seconds:
        raise TimeoutError("research time limit; run incomplete")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return FitResult(
        refit,
        selected,
        refit_scales,
        scales,
        tuple(losses),
        elapsed,
        trained / elapsed,
        int(peak if sys.platform == "darwin" else peak * 1024),
        runtime.device,
    )


def predict_logits(
    model: ResearchModel, rows: tuple[ResearchRow, ...], features: np.ndarray
) -> np.ndarray:
    device = next(model.parameters()).device
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in aligned_batches(rows, features, batch_size=1024, shuffle=False):
            result.append(
                model(
                    torch.tensor(batch.tokens, dtype=torch.long, device=device),
                    torch.tensor(batch.features, device=device),
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(result)
