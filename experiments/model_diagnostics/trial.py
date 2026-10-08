"""Fit bounded exploratory trials through the existing chronological training loop."""

from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import cast

import numpy as np
import torch
from pydantic_core import to_jsonable_python
from torch import Tensor

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.cache import oriented_digest
from experiments.common.checkpoints import swap_error
from experiments.common.data_access import RoleAccess
from experiments.common.fit import fit_research, predict_logits
from experiments.common.session import Session
from experiments.model_diagnostics.artifacts import (
    TrialManifest,
    TrialSpec,
    bind_inputs,
    finish_trial,
    load_trial,
    trial_stage,
)
from experiments.model_diagnostics.model import DiagnosticModel, recipe
from experiments.model_diagnostics.scoring import evaluate


def shuffled_training(access: RoleAccess, seed: int | None) -> RoleAccess:
    """Permute labels within fit/watch separately; retain real calibration/development labels."""
    if seed is None:
        return access
    rng = np.random.default_rng(seed)
    labels: dict[str, int] = {}
    for role in ("selection_fit", "watch"):
        rows = access.read(role, "fit")
        shuffled = rng.permutation([r.label for r in rows])
        labels.update((r.event_key, int(y)) for r, y in zip(rows, shuffled, strict=True))
    rows = (
        *(replace(r, label=labels[r.event_key]) for r in access.read("refit", "fit")),
        *access.read("calibration", "calibrate"),
        *access.read("development", "compare"),
    )
    return RoleAccess(access.protocol, rows)


def load_model(directory: Path, session: Session, root: Path) -> DiagnosticModel:
    """Reject failed validation or timed-out trials; calibration failure permits raw scoring."""
    manifest = load_trial(directory)
    if manifest.status in ("failed", "time_limited"):
        raise ValueError(f"cannot load model from {manifest.status} trial")
    spec = manifest.inputs.spec
    expected = bind_inputs(session, spec, root)
    load_trial(directory, expected)
    training = shuffled_training(session.access, spec.shuffle_seed).read("refit", "fit")
    model = DiagnosticModel(len(session.schema.identity_vocab), spec.model, training)
    state = cast(dict[str, Tensor], torch.load(directory / "checkpoint.pt", weights_only=True))
    model.load_state_dict(state)
    model.eval()
    return model


def run_trial(destination: Path, session: Session, spec: TrialSpec, root: Path) -> TrialManifest:
    expected = bind_inputs(session, spec, root)
    if destination.exists():
        return load_trial(destination, expected)
    started = monotonic()
    with trial_stage(destination) as stage:
        status: str = "failed"
        failure: str | None = None
        try:
            access = shuffled_training(session.access, spec.shuffle_seed)
            builder, factory = recipe(spec.model, len(session.schema.identity_vocab))
            fitted = fit_research(
                access,
                builder,
                factory,
                spec.optimizer,
                seed=spec.seed,
                penalty=0,
                scale_features=False,
            )
            model = cast(DiagnosticModel, fitted.model)
            training = access.read("refit", "fit")
            cal = access.read("calibration", "calibrate")
            dev = access.read("development", "compare")
            values = (builder(training, cal), builder(training, dev))
            actual = tuple(
                predict_logits(model, rows, x) for rows, x in zip((cal, dev), values, strict=True)
            )
            neutral = tuple(
                np.full_like(x, -1) if spec.model.nuisance == "joint" else np.zeros_like(x)
                for x in values
            )
            matchup = tuple(
                predict_logits(model, rows, x) for rows, x in zip((cal, dev), neutral, strict=True)
            )
            evaluated = evaluate(
                stage, cal, dev, (actual[0], actual[1]), (matchup[0], matchup[1]), training
            )
            torch.save(model.state_dict(), stage / "checkpoint.pt")
            restored = DiagnosticModel(len(session.schema.identity_vocab), spec.model, training)
            restored.load_state_dict(
                cast(
                    dict[str, Tensor],
                    torch.load(stage / "checkpoint.pt", weights_only=True, map_location="cpu"),
                )
            )
            restored.eval()
            np.testing.assert_allclose(
                predict_logits(restored, dev, values[1]), actual[1], atol=1e-6
            )
            check_rows = dev[:128]
            features = builder(training, check_rows)
            tokens = torch.tensor(
                [r.tokens for r in check_rows], device=next(model.parameters()).device
            )
            error = swap_error(model, tokens, features, spec.model.nuisance)
            if error > 1e-5:
                raise ValueError("side-swap logit check failed")
            report = {
                **evaluated.report,
                "selected_epoch": fitted.selected_epoch,
                "selection_training_losses": fitted.selection_training_losses,
                "watch_losses": fitted.watch_losses,
                "refit_training_losses": fitted.refit_training_losses,
                "fit_seconds": fitted.elapsed_seconds,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "fit_population_digest": oriented_digest(training),
                "swap_error": error,
                "objective_penalty": float(model.penalty(0).detach().cpu().item()),
                "player_effects": (
                    {
                        "maximum_absolute": float(
                            model.player_effects.centered().abs().max().item()
                        ),
                        "count": len(model.player_effects.vocabulary.players),
                    }
                    if model.player_effects is not None
                    else None
                ),
                "optimizer": spec.optimizer.model_dump(),
            }
            (stage / "report.json").write_bytes(canonical_json_bytes(to_jsonable_python(report)))
            status = "complete" if evaluated.selection_loss is not None else "calibration_failed"
        except (ValueError, RuntimeError, TimeoutError, OSError, AssertionError) as error:
            failure = str(error)
            status = "time_limited" if isinstance(error, TimeoutError) else "failed"
            (stage / "failure.json").write_bytes(canonical_json_bytes({"reason": failure}))
        finish_trial(
            stage,
            expected,
            status,
            monotonic() - started,
            failure,
        )
    return load_trial(destination, expected)
