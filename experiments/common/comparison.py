"""Frozen development comparisons retain seed variability and calibration provenance."""

from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from pydantic_core import to_jsonable_python

from clash_sos.domain.analytics import PredictionProvenance
from clash_sos.domain.attention_protocol import digest_row_keys
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.comparison_inputs import ComparisonRun, fixed_slices, load_comparison_run
from experiments.common.contracts import StudyConfig, Variant, fingerprint
from experiments.common.ensembles import (
    EnsemblePredictions,
    FrozenEnsemble,
    calibrated_ensemble,
    ensemble_bytes,
    publish_comparison,
)
from experiments.common.matrix import confirmation_jobs, screen_jobs
from experiments.common.predictions import pair_predictions
from experiments.common.provenance import code_digest
from experiments.common.slices import compare_slices
from experiments.common.statistics import paired_comparison, score_predictions
from experiments.common.windows import compare_windows, summarize_windows, window_seed_variability


def _directory(
    model_root: Path, config: StudyConfig, variant: Variant, seed: int, penalty: float
) -> Path:
    if variant == config.variants[0] and config.study_id != "response-cycle":
        return (
            model_root.parent.parent
            / "response-cycle"
            / model_root.name
            / f"A0-seed{seed}-penalty0"
        )
    return model_root / f"{variant.variant_id}-seed{seed}-penalty{penalty:g}"


def _provenance(run: ComparisonRun, model: str) -> PredictionProvenance:
    meta = run.checkpoint.metadata
    return PredictionProvenance(
        model_version=model,
        dataset_version=run.support.protocol.dataset_version,
        card_catalog_version=meta.input_schema.catalog_version,
        balance_era_id=meta.input_schema.balance_era_id,
    )


def comparison_report(
    config: StudyConfig,
    selected_penalties: dict[str, float],
    model_root: Path,
    report_root: Path,
) -> dict[str, object]:
    if config.stage != "development-frozen" or config.population is None:
        raise PermissionError(
            "smoke and reporting runs cannot enter controlled development comparisons"
        )
    confirmation_jobs(config, selected_penalties)
    baseline = config.variants[0]
    if (
        not baseline.enabled
        or baseline.feature_groups
        or baseline.nuisance != "none"
        or (baseline.architecture != "explicit")
    ):
        raise ValueError("first registered variant must be the enabled shared explicit baseline")
    variants = tuple(variant for variant in config.variants if variant.enabled)
    runs: dict[str, tuple[ComparisonRun, ...]] = {}
    all_runs: list[ComparisonRun] = []
    for variant in variants:
        penalty = (
            0.0
            if variant == baseline or variant.architecture == "attention"
            else selected_penalties[variant.variant_id]
        )
        loaded = tuple(
            load_comparison_run(
                _directory(model_root, config, variant, seed, penalty),
                config,
                variant,
                seed,
                penalty,
                shared=variant == baseline and config.study_id != "response-cycle",
            )
            for seed in config.seeds
        )
        runs[variant.variant_id] = loaded
        all_runs.extend(loaded)
    for job in screen_jobs(config):
        if job.penalty > 0 and job.penalty != selected_penalties[job.variant_id]:
            variant = next(item for item in variants if item.variant_id == job.variant_id)
            all_runs.append(
                load_comparison_run(
                    _directory(model_root, config, variant, job.seed, job.penalty),
                    config,
                    variant,
                    job.seed,
                    job.penalty,
                    shared=False,
                )
            )
    reference = runs[baseline.variant_id][0]
    code_sha256 = code_digest(Path.cwd())
    for run in all_runs:
        if (run.manifest.git_sha, run.manifest.lock_sha256) != (
            reference.manifest.git_sha,
            reference.manifest.lock_sha256,
        ) or dict(run.manifest.runtime).get("code_sha256") != code_sha256:
            raise ValueError(
                "controlled runs must share the unchanged code revision and dependency lock"
            )
        pair_predictions(run.calibration, reference.calibration)
        pair_predictions(run.development, reference.development)
        if (run.support.refit, run.support.development) != (
            reference.support.refit,
            reference.support.development,
        ):
            raise ValueError(
                "comparisons require identical oriented training/evaluation covariates"
            )
    ensembles: dict[str, EnsemblePredictions] = {}
    payloads: dict[str, bytes] = {}
    assets: dict[str, str] = {}
    model_reports: dict[str, object] = {}
    for variant in variants:
        identity = variant.variant_id
        seeds = runs[identity]
        ensemble = calibrated_ensemble(
            tuple(run.calibration for run in seeds),
            tuple(run.development for run in seeds),
        )
        ensembles[identity] = ensemble
        asset = FrozenEnsemble(
            variant_id=identity,
            config_sha256=fingerprint(config),
            code_sha256=code_sha256,
            git_sha=reference.manifest.git_sha,
            lock_sha256=reference.manifest.lock_sha256,
            population=config.population,
            temperature=ensemble.fit.temperature,
            calibration_rows_sha256=digest_row_keys(row.row_key for row in ensemble.calibration),
            calibration_row_count=len(ensemble.calibration),
            seed_runs=tuple(run.asset for run in seeds),
        )
        asset_name = f"ensemble-{identity}.json"
        payloads[asset_name] = ensemble_bytes(asset)
        assets[identity] = str((report_root / asset_name).resolve())
        for role in ("calibration", "development"):
            payloads[f"ensemble-{identity}-{role}.json"] = canonical_json_bytes(
                to_jsonable_python(getattr(ensemble, role))
            )
        model_reports[identity] = {
            "calibration": asdict(ensemble.fit),
            "development": score_predictions(ensemble.development),
            "seeds": [
                {
                    "seed": run.asset.seed,
                    "calibration": score_predictions(run.calibration),
                    "development": score_predictions(run.development),
                    "checkpoint_bytes": next(
                        member.size_bytes
                        for member in run.manifest.outputs
                        if member.path == "checkpoint.pt"
                    ),
                    "runtime": dict(run.manifest.runtime),
                }
                for run in seeds
            ],
        }
    slices = fixed_slices(reference, ensembles[baseline.variant_id].development)
    comparisons: dict[str, object] = {}
    for comparator_variant, candidate_variant in combinations(variants, 2):
        candidate_id, comparator_id = candidate_variant.variant_id, comparator_variant.variant_id
        candidate, comparator = ensembles[candidate_id], ensembles[comparator_id]
        windows = compare_windows(
            candidate.development,
            comparator.development,
            candidate_provenance=_provenance(runs[candidate_id][0], f"ensemble:{candidate_id}"),
            comparator_provenance=_provenance(runs[comparator_id][0], f"ensemble:{comparator_id}"),
        )
        seed_windows = tuple(
            compare_windows(
                first.development,
                second.development,
                candidate_provenance=_provenance(first, f"{candidate_id}:seed{first.asset.seed}"),
                comparator_provenance=_provenance(
                    second, f"{comparator_id}:seed{second.asset.seed}"
                ),
            )
            for first, second in zip(runs[candidate_id], runs[comparator_id], strict=True)
        )
        comparisons[f"{candidate_id} minus {comparator_id}"] = {
            "paired": paired_comparison(candidate.development, comparator.development),
            "slices": [
                {
                    "name": item.name,
                    "row_count": item.row_count,
                    "support_status": "empty"
                    if not item.row_count
                    else "unsupported"
                    if item.row_count < config.rules.minimum_slice_rows
                    else "supported",
                    "comparison": item.comparison,
                }
                for item in compare_slices(candidate.development, comparator.development, slices)
            ],
            "windows": summarize_windows(windows),
            "player_windows": windows,
            "seed_window_variability": window_seed_variability(seed_windows),
            "per_seed": [
                {
                    "seed": first.asset.seed,
                    "paired": paired_comparison(first.development, second.development),
                }
                for first, second in zip(runs[candidate_id], runs[comparator_id], strict=True)
            ],
        }
    report: dict[str, object] = {
        "stage": config.stage,
        "config_sha256": fingerprint(config),
        "code_sha256": code_sha256,
        "population": config.population,
        "selected_penalties": selected_penalties,
        "models": model_reports,
        "comparisons": comparisons,
        "ensemble_assets": assets,
        "interpretation": (
            "matchup-only calibration against observed outcomes; equal skill is not identified"
        ),
        "seed_uncertainty": "seed dispersion is not independent evaluation sampling uncertainty",
    }
    payloads["comparison.json"] = canonical_json_bytes(to_jsonable_python(report)) + b"\n"
    publish_comparison(report_root, payloads)
    return report
