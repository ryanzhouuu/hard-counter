import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.components import components
from experiments.common.contracts import OptimizerConfig, StudyConfig
from experiments.common.execution import execute_jobs, job_name
from experiments.common.matrix import (
    DevelopmentResult,
    confirmation_jobs,
    screen_jobs,
    smoke_jobs,
)
from experiments.common.readiness import readiness
from experiments.common.selection import select_once
from experiments.common.session import prepare_session

STUDIES = {
    "matchup_features": "response-cycle",
    "player_adjustment": "player-adjustment",
    "form_mechanics": "form-mechanics",
    "tower_mechanics": "tower-mechanics",
    "higher_order": "higher-order",
}


def parser(study: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Reproducible research; smoke outcomes cannot select models"
    )
    result.add_argument(
        "command", choices=("validate", "smoke", "run-development", "compare", "confirm")
    )
    result.add_argument(
        "--config", type=Path, default=Path(__file__).parent.parent / "configs" / (study + ".json")
    )
    result.add_argument("--synthetic", action="store_true")
    result.add_argument("--dataset", type=Path)
    result.add_argument("--protocol", type=Path)
    result.add_argument("--schema", type=Path)
    result.add_argument("--cache", type=Path)
    result.add_argument("--mechanics", type=Path)
    result.add_argument("--row-cap", type=int)
    result.add_argument("--output", type=Path, default=Path("data/experiments"))
    result.add_argument("--models", type=Path, default=Path("models/experiments"))
    result.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    result.add_argument("--freeze", type=Path)
    result.add_argument("--reporting-contract", type=Path)
    result.add_argument("--reporting-source", type=Path)
    return result


def main(study: str, argv: list[str] | None = None) -> None:
    args = parser(study).parse_args(argv)
    config = StudyConfig.model_validate_json(args.config.read_bytes())
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("run ID must be a single safe path component")
    if args.row_cap is not None and args.row_cap < 12:
        raise ValueError("row cap must be at least twelve")
    if config.study_id != study:
        raise ValueError("configuration belongs to another study")
    if args.command == "confirm":
        from experiments.common.confirm import confirm

        confirm(args, config)
        return
    if args.command in ("run-development", "compare") and config.stage != "development-frozen":
        raise PermissionError("development execution/comparison requires explicit frozen inputs")
    if args.synthetic and config.stage != "preparation/smoke":
        raise PermissionError("synthetic runs cannot advance stages")
    if args.command == "smoke":
        effective = OptimizerConfig.model_validate(
            {
                **config.optimizer.model_dump(),
                "max_epochs": min(config.optimizer.max_epochs, config.smoke_epoch_cap),
                "time_limit_seconds": min(
                    config.optimizer.time_limit_seconds, config.smoke_time_cap
                ),
            }
        )
        config = StudyConfig.model_validate(
            {
                **config.model_dump(),
                "stage": "preparation/smoke",
                "optimizer": effective,
                "smoke_row_cap": args.row_cap or config.smoke_row_cap,
            }
        )
    reports = args.output / study / args.run_id
    models = args.models / study / args.run_id
    session = prepare_session(
        config,
        reports / "inputs",
        synthetic=args.synthetic,
        dataset=args.dataset,
        protocol=args.protocol,
        schema=args.schema,
        cache=args.cache,
        mechanics=args.mechanics,
        row_cap=args.row_cap or config.smoke_row_cap,
    )
    if args.command == "validate":
        document = readiness(session.access, session.catalog, session.schema)
        from hashlib import sha256

        from experiments.common.session import attention_session
        from experiments.mechanics.contracts import MechanicsUnavailable

        registry: dict[str, object] = {}
        for variant in config.variants:
            if not variant.enabled:
                registry[variant.variant_id] = {
                    "status": "disabled",
                    "reason": variant.disabled_reason,
                }
                continue
            try:
                selected = (
                    attention_session(
                        session, reports / "inputs", args.row_cap or config.smoke_row_cap
                    )
                    if variant.architecture == "attention"
                    else session
                )
                recipe = components(
                    config,
                    variant,
                    selected.catalog,
                    selected.schema,
                    selected.access.read("refit", "fit")[0],
                )
                registry[variant.variant_id] = {
                    "status": "implemented",
                    "feature_names": recipe.names,
                    "feature_sha256": sha256(
                        canonical_json_bytes((recipe.names, recipe.formulas))
                    ).hexdigest(),
                    "schema_sha256": selected.schema.fingerprint(),
                }
            except MechanicsUnavailable as error:
                registry[variant.variant_id] = {"status": "unavailable", "reason": str(error)}
        document["variant_registry"] = registry
        print(json.dumps(document, sort_keys=True))
        return
    if args.command == "compare":
        selection_path = reports / "selection.json"
        results = [
            DevelopmentResult(job, models / job_name(job), "development.json")
            for job in screen_jobs(config)
            if job.penalty > 0
        ]
        selected = select_once(config, results, session.access, selection_path, Path.cwd())
        jobs = confirmation_jobs(config, selected)
        output = execute_jobs(config, session, jobs, models, reports, Path.cwd())
        if any(item["status"] != "complete" for item in output):
            raise RuntimeError("confirmation matrix is incomplete")
        from experiments.common.comparison import comparison_report

        comparison_report(config, selected, models, reports)
        print(
            json.dumps({"selected_penalties": selected, "report": str(reports / "comparison.json")})
        )
        return
    jobs = smoke_jobs(config) if args.command == "smoke" else screen_jobs(config)
    output = execute_jobs(config, session, jobs, models, reports, Path.cwd())
    (reports / "readiness.json").write_bytes(
        canonical_json_bytes(readiness(session.access, session.catalog, session.schema))
    )
    print(json.dumps(output, sort_keys=True))
    if any(item["status"] != "complete" for item in output):
        raise RuntimeError("one or more research branches are incomplete; inspect failures")
