from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from pydantic_core import to_jsonable_python

from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.artifacts import (
    completed_manifest,
    file_record,
    finish_run,
    load_run,
    publication,
)
from experiments.common.cache import cache_refit_features
from experiments.common.checkpoints import save_checkpoint, swap_error, write_support
from experiments.common.components import components
from experiments.common.contracts import (
    PopulationIdentity,
    RunManifest,
    StudyConfig,
    Variant,
)
from experiments.common.data_access import RoleAccess
from experiments.common.fit import fit_research
from experiments.common.predictions import read_predictions
from experiments.common.provenance import code_digest, revision, versions
from experiments.common.scoring import score_fit
from experiments.common.statistics import score_predictions
from experiments.higher_order.features import eligibility
from experiments.higher_order.features import extract as pattern_features
from experiments.mechanics.contracts import MechanicsCatalog, MechanicsUnavailable
from experiments.player_adjustment.history import training_history
from experiments.player_adjustment.model import PlayerModel
from experiments.player_adjustment.reports import development_diagnostics


def run_variant(
    root: Path,
    destination: Path,
    config: StudyConfig,
    variant: Variant,
    access: RoleAccess,
    population: PopulationIdentity,
    catalog: MechanicsCatalog,
    schema: AttentionCardSchema,
    *,
    seed: int,
    penalty: float,
    phase: str,
) -> RunManifest:
    """Publish a complete checkpoint last, retaining failures as explicit run states."""
    if config.stage not in ("preparation/smoke", "development-frozen"):
        raise PermissionError("frozen candidates and reporting cannot fit or calibrate")
    started = monotonic()
    git_sha, dirty, lock = revision(root)
    runtime = [
        *versions(),
        ("code_sha256", code_digest(root)),
        ("variant_id", variant.variant_id),
        ("seed", str(seed)),
        ("penalty", str(penalty)),
        ("phase", phase),
    ]
    selected: tuple[int, ...] = ()
    temperatures: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    failures: tuple[str, ...] = ()
    status = "complete"
    with publication(destination) as stage:
        try:
            if not variant.enabled:
                raise MechanicsUnavailable(variant.disabled_reason or "disabled")
            if config.stage == "development-frozen" and catalog.synthetic:
                raise MechanicsUnavailable("synthetic mechanics cannot enter controlled fitting")
            if config.population is not None and config.population != population:
                raise ValueError("frozen population identity does not match loaded inputs")
            if config.mechanics_sha256 is not None and config.mechanics_sha256 != catalog.digest:
                raise ValueError("frozen mechanics digest mismatch")
            if (
                variant.feature_groups
                and not catalog.synthetic
                and (
                    population.start.date().isoformat() < catalog.era_start
                    or population.end.date().isoformat() > catalog.era_end
                )
            ):
                raise MechanicsUnavailable("population is outside mechanics applicability bounds")
            refit = access.read("refit", "fit")
            write_support(stage, access, refit)
            recipe = components(config, variant, catalog, schema, refit[0])
            definitions = sha256(canonical_json_bytes((recipe.names, recipe.formulas))).hexdigest()
            if config.stage == "development-frozen" and (
                variant.schema_sha256 != schema.fingerprint()
                or variant.feature_sha256 != definitions
                or variant.feature_names != recipe.names
            ):
                raise ValueError("frozen variant schema or feature registry mismatch")
            fitted = fit_research(
                access,
                recipe.builder,
                recipe.factory,
                config.optimizer,
                seed=seed,
                penalty=penalty,
                scale_features=recipe.scale,
            )
            cache_refit_features(
                stage / "feature-cache",
                access,
                recipe.builder,
                recipe.names,
                recipe.formulas,
                fitted.scales,
                schema.fingerprint(),
                catalog.digest,
            )
            selected = (fitted.selected_epoch,)
            scales = tuple(float(s) for s in fitted.scales)
            runtime.extend(
                (
                    ("device", fitted.device),
                    ("elapsed_seconds", str(fitted.elapsed_seconds)),
                    ("rows_per_second", str(fitted.rows_per_second)),
                    ("peak_memory_bytes", str(fitted.peak_memory_bytes)),
                    (
                        "seconds_per_epoch",
                        str(
                            fitted.elapsed_seconds
                            / (len(fitted.watch_losses) + fitted.selected_epoch)
                        ),
                    ),
                )
            )
            matchup_temperature, actual_temperature, dev_predictions = score_fit(
                stage,
                access,
                fitted,
                recipe,
            )
            temperatures = (matchup_temperature.temperature, actual_temperature.temperature)
            transform: dict[str, object] = {}
            if variant.nuisance == "history":
                transform["history"] = training_history(refit)[1].model_dump(mode="json")
            if isinstance(fitted.model, PlayerModel) and fitted.model.player_effects is not None:
                transform["joint"] = fitted.model.player_effects.freeze().model_dump(mode="json")
            if config.study_id == "higher-order" and variant.feature_groups:
                pattern_reports = {
                    role: asdict(
                        eligibility(
                            [pattern_features(r.tokens, catalog) for r in access.read(role, "fit")],
                            role=role,
                            minimum_support=config.rules.minimum_pattern_support,
                        )
                    )
                    for role in ("selection_fit", "refit")
                }
                transform["patterns"] = pattern_reports["refit"]
                (stage / "patterns.json").write_bytes(canonical_json_bytes(pattern_reports))
            save_checkpoint(
                stage / "checkpoint.pt",
                config,
                variant,
                schema,
                catalog,
                population,
                recipe.names,
                recipe.formulas,
                fitted,
                transform,
                temperatures,
            )
            gradient_finite = all(
                p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in fitted.model.parameters()
            )
            tokens = torch.tensor(
                [refit[0].tokens], dtype=torch.long, device=next(fitted.model.parameters()).device
            )
            feature = np.asarray(recipe.builder(refit, (refit[0],)) / fitted.scales)
            complement_error = swap_error(fitted.model, tokens, feature, variant.nuisance)
            report = {
                "purpose": "correctness/resources"
                if config.stage == "preparation/smoke"
                else "development comparison",
                "interpretation": (
                    "matchup-only calibration against observed outcomes; "
                    "equal skill is not identified"
                ),
                "calibration": asdict(matchup_temperature),
                "watch_losses": fitted.watch_losses,
                "selection_scales": tuple(float(s) for s in fitted.selection_scales),
                "development_metrics": asdict(score_predictions(dev_predictions)),
                "finite_gradients": gradient_finite,
                "swap_error": complement_error,
                "feature_names": recipe.names,
                "feature_definition_digest": sha256(
                    canonical_json_bytes((recipe.names, recipe.formulas))
                ).hexdigest(),
            }
            if config.study_id == "player-adjustment":
                report["actual_outcome_calibration"] = asdict(actual_temperature)
                report["player_diagnostics"] = development_diagnostics(
                    access,
                    fitted.model,
                    dev_predictions,
                    read_predictions(stage / "actual-development.json")
                    if variant.nuisance != "none"
                    else dev_predictions,
                    config.player_support_bins,
                    penalty,
                )
            (stage / "report.json").write_bytes(canonical_json_bytes(to_jsonable_python(report)))
        except (ValueError, TimeoutError, RuntimeError, OSError, TypeError) as error:
            status = "time_limited" if isinstance(error, TimeoutError) else "failed"
            failures = (str(error),)
            (stage / "failure.json").write_bytes(
                canonical_json_bytes(
                    {
                        "reason": str(error),
                        "elapsed_seconds": monotonic() - started,
                        "unavailable": isinstance(error, MechanicsUnavailable),
                    }
                )
            )
        outputs = tuple(file_record(p, stage) for p in sorted(stage.rglob("*")) if p.is_file())
        manifest = completed_manifest(
            destination.name,
            config,
            population,
            git_sha,
            dirty,
            lock,
            status,
            outputs,
            tuple(runtime),
            selected,
            temperatures,
            scales,
            failures,
        )
        finish_run(stage, manifest)
    return load_run(destination)
