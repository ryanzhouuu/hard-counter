import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.artifacts import file_record, load_run, publication, verify_files
from experiments.common.calibration import ensemble_logits
from experiments.common.candidate import CandidateFreeze, validate_candidate_population
from experiments.common.checkpoints import load_checkpoint, predictions
from experiments.common.confirmation_artifacts import finish_confirmation
from experiments.common.contracts import StudyConfig, fingerprint
from experiments.common.data_access import ReportingContract
from experiments.common.predictions import Prediction
from experiments.common.prospective_io import load_reporting_jsonl
from experiments.common.provenance import code_digest, revision
from experiments.common.statistics import holm_adjust, paired_comparison, score_predictions


def confirm(args: argparse.Namespace, config: StudyConfig) -> None:
    """Evaluate a frozen ensemble once; no fitter or calibrator is called."""
    if config.stage != "prospective-reporting":
        raise PermissionError(
            "confirmation requires an explicit prospective-reporting configuration"
        )
    if args.freeze is None or args.reporting_contract is None or args.reporting_source is None:
        raise ValueError("confirmation requires freeze, reporting-contract, and reporting-source")
    freeze_path = Path(args.freeze)
    frozen = CandidateFreeze.model_validate_json(freeze_path.read_bytes())
    verify_files(freeze_path.parent, frozen.assets)
    if frozen.code_sha256 != code_digest(Path.cwd()):
        raise ValueError("candidate code changed after freezing")
    lock = revision(Path.cwd())[2]
    contract = TypeAdapter(ReportingContract).validate_json(
        Path(args.reporting_contract).read_bytes()
    )
    if contract.fit_population != frozen.fit_population or contract.candidate_sha256 != fingerprint(
        frozen
    ):
        raise ValueError("reporting contract belongs to another frozen candidate")
    if config.population != contract.fit_population or config.prospective != contract.population:
        raise ValueError("reporting configuration population mismatch")
    validate_candidate_population(frozen, contract.population)
    from experiments.common.ensembles import FrozenEnsemble

    ensembles = [
        FrozenEnsemble.model_validate_json((freeze_path.parent / member.path).read_bytes())
        for member in frozen.assets
        if Path(member.path).name.startswith("ensemble-")
        and not Path(member.path).name.endswith(("-calibration.json", "-development.json"))
    ]
    if not ensembles:
        raise ValueError("freeze has no inventoried ensemble assets")
    baseline = [e for e in ensembles if e.variant_id.endswith("0")]
    if len(baseline) != 1 or {s.variant_id for s in frozen.selections} != {
        e.variant_id
        for e in ensembles
        if not e.variant_id.endswith("0") and e.variant_id != "full_attention"
    }:
        raise ValueError("freeze requires one baseline and exactly its selected challengers")
    if config.rules.confirmation == "single" and len(frozen.selections) != 1:
        raise ValueError("single-challenger design requires exactly one frozen challenger")
    for ensemble in ensembles:
        if (
            ensemble.config_sha256
            != dict(frozen.source_configs).get(ensemble.variant_id, frozen.source_config_sha256)
            or ensemble.code_sha256 != frozen.code_sha256
            or ensemble.population != frozen.fit_population
            or ensemble.lock_sha256 != lock
        ):
            raise ValueError("ensemble provenance differs from candidate freeze")
        selection = next(
            (s for s in frozen.selections if s.variant_id == ensemble.variant_id), None
        )
        if selection is not None and set(selection.seeds) != {s.seed for s in ensemble.seed_runs}:
            raise ValueError("ensemble seeds differ from frozen selection")
        for seed in ensemble.seed_runs:
            path = Path(seed.directory)
            run = load_run(path)
            runtime = dict(run.runtime)
            if (
                run.config_sha256 != seed.run_config_sha256
                or runtime.get("seed") != str(seed.seed)
                or (
                    selection is not None
                    and runtime.get("penalty") != str(float(selection.penalty))
                )
            ):
                raise ValueError("seed run settings differ from frozen candidate")

            for name, digest in (
                ("manifest.json", seed.manifest_sha256),
                ("checkpoint.pt", seed.checkpoint_sha256),
            ):
                if file_record(path / name, path).sha256 != digest:
                    raise ValueError("frozen seed asset changed")
    destination = args.output / config.study_id / args.run_id
    marker = freeze_path.with_suffix(".reporting-consumed.json")
    failure: Exception | None = None
    row_count = 0
    with publication(destination) as stage:
        with marker.open("xb") as consumed:
            consumed.write(
                canonical_json_bytes(
                    {"population": contract.population.model_dump(mode="json"), "status": "started"}
                )
            )
        try:
            first = load_checkpoint(Path(ensembles[0].seed_runs[0].directory) / "checkpoint.pt")
            access = load_reporting_jsonl(
                Path(args.reporting_source),
                first.metadata.input_schema,
                contract,
                row_cap=args.row_cap or config.smoke_row_cap,
            )
            rows = access.read("reporting", "report")
            row_count = len(rows)
            outputs: dict[str, tuple[Prediction, ...]] = {}
            for ensemble in ensembles:
                raw = [
                    tuple(
                        float(z)
                        for z in load_checkpoint(Path(seed.directory) / "checkpoint.pt").logits(
                            rows
                        )
                    )
                    for seed in ensemble.seed_runs
                ]
                from experiments.common.calibration import sigmoid

                logits = ensemble_logits(tuple(tuple(sigmoid(z) for z in seed) for seed in raw))
                import numpy as np

                outputs[ensemble.variant_id] = predictions(
                    rows, np.asarray(logits), ensemble.temperature
                )
            comparator = outputs[baseline[0].variant_id]
            report = {
                "stage": config.stage,
                "population": contract.population.model_dump(mode="json"),
                "metrics": {name: score_predictions(value) for name, value in outputs.items()},
                "comparisons": {
                    name: paired_comparison(value, comparator)
                    for name, value in outputs.items()
                    if name != baseline[0].variant_id
                },
                "confirmation_design": config.rules.confirmation,
            }
            from scipy.stats import norm

            pvalues: dict[str, float | None] = {}
            for name, value in outputs.items():
                if name == baseline[0].variant_id:
                    continue
                interval = paired_comparison(value, comparator).log_loss
                if (
                    interval is None
                    or interval.standard_error is None
                    or interval.standard_error == 0
                ):
                    pvalues[name] = None
                else:
                    score = interval.mean_difference / interval.standard_error
                    pvalues[name] = (
                        float(norm.cdf(score))
                        if config.rules.test_alternative == "improvement"
                        else float(2 * norm.sf(abs(score)))
                    )
            report["primary_pvalues"] = pvalues
            report["adjusted_pvalues"] = (
                holm_adjust(pvalues) if config.rules.confirmation == "holm" else pvalues
            )
            (stage / "confirmation.json").write_bytes(
                canonical_json_bytes(to_jsonable_python(report))
            )
        except Exception as error:
            (stage / "failure.json").write_bytes(
                canonical_json_bytes({"status": "incomplete", "failure": str(error)})
            )
            failure = error
        finish_confirmation(
            stage,
            destination.name,
            config,
            contract,
            frozen,
            freeze_path,
            Path(args.reporting_contract),
            failure,
        )
    if failure is not None:
        raise failure
    marker.write_bytes(
        canonical_json_bytes(
            {
                "population": contract.population.model_dump(mode="json"),
                "status": "complete",
                "report": str((destination / "confirmation.json").resolve()),
            }
        )
    )
    print(json.dumps({"report": str(destination / "confirmation.json"), "rows": row_count}))
