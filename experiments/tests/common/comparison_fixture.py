from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.common.artifacts import file_record, finish_run
from experiments.common.calibration import fit_temperature
from experiments.common.checkpoints import predictions, save_checkpoint, write_support
from experiments.common.components import components
from experiments.common.contracts import (
    ComparisonRules,
    RunManifest,
    StudyConfig,
    Variant,
    fingerprint,
)
from experiments.common.fit import FitResult
from experiments.common.matrix import MatrixJob, confirmation_jobs, screen_jobs
from experiments.common.predictions import write_predictions
from experiments.common.provenance import code_digest
from experiments.common.session import Session, prepare_session

from clash_sos.domain.canonical_dataset import canonical_json_bytes


def comparison_fixture(
    tmp_path: Path,
    *,
    shared: bool = False,
) -> tuple[StudyConfig, Path, Path, Session]:
    initial = StudyConfig(
        study_id="form-mechanics" if shared else "response-cycle",
        variants=(
            Variant(variant_id="C0" if shared else "A0"),
            Variant(
                variant_id="C1" if shared else "A1",
                feature_groups=("inherited",) if shared else ("response",),
            ),
        ),
        penalties=(0.1, 0.2),
    )
    session = prepare_session(
        initial,
        tmp_path / "inputs",
        synthetic=True,
        dataset=None,
        protocol=None,
        schema=None,
        cache=None,
        mechanics=None,
        row_cap=128,
    )
    variants: list[Variant] = []
    for variant in initial.variants:
        recipe = components(
            initial,
            variant,
            session.catalog,
            session.schema,
            session.access.read("refit", "fit")[0],
        )
        variants.append(
            variant.model_copy(
                update={
                    "schema_sha256": session.schema.fingerprint(),
                    "feature_names": recipe.names,
                    "feature_sha256": sha256(
                        canonical_json_bytes((recipe.names, recipe.formulas))
                    ).hexdigest(),
                }
            )
        )
    config = StudyConfig.model_validate(
        {
            **initial.model_dump(),
            "stage": "development-frozen",
            "population": session.population,
            "schema_sha256": session.schema.fingerprint(),
            "feature_definitions_sha256": "a" * 64,
            "decision_sha256": "a" * 64,
            "variants": variants,
            "rules": ComparisonRules(practical_margin=0.001, maximum_brier_regression=0.001),
        }
    )
    models, reports = tmp_path / "models" / config.study_id / "frozen", tmp_path / "reports"
    selected = {config.variants[1].variant_id: config.penalties[0]}
    for job in (*screen_jobs(config), *confirmation_jobs(config, selected)):
        local = config
        variant = next(item for item in config.variants if item.variant_id == job.variant_id)
        target = models / f"{job.variant_id}-seed{job.seed}-penalty{job.penalty:g}"
        if job.shared:
            variant = variant.model_copy(update={"variant_id": "A0"})
            local = StudyConfig.model_validate(
                {
                    **config.model_dump(),
                    "study_id": "response-cycle",
                    "variants": (variant,),
                }
            )
            target = (
                models.parent.parent
                / "response-cycle"
                / models.name
                / f"A0-seed{job.seed}-penalty0"
            )
            job = replace(job, variant_id="A0", shared=False)
        write_comparison_run(target, local, variant, job, session)
    return config, models, reports, session


def write_comparison_run(
    destination: Path,
    config: StudyConfig,
    variant: Variant,
    job: MatrixJob,
    session: Session,
) -> None:
    destination.mkdir(parents=True)
    access, schema, catalog = session.access, session.schema, session.catalog
    refit = access.read("refit", "fit")
    recipe = components(config, variant, catalog, schema, refit[0])
    model = recipe.factory(refit, len(recipe.names))
    result = FitResult(
        model,
        1,
        np.ones(len(recipe.names)),
        np.ones(len(recipe.names)),
        (0.7,),
        0.01,
        1000,
        64,
        "cpu",
    )
    calibration, development = (
        access.read("calibration", "calibrate"),
        access.read("development", "compare"),
    )
    amplitude = 1.0 + job.seed * 2
    cal_logits = np.asarray(
        [
            amplitude * (1 if row.label else -1) * (1 if i % 4 < 3 else -1)
            for i, row in enumerate(calibration)
        ]
    )
    dev_logits = np.asarray(
        [
            amplitude * (1 if row.label else -1) * (1 if i % 4 < 3 else -1)
            for i, row in enumerate(development)
        ]
    )
    temperature = fit_temperature(
        tuple(float(value) for value in cal_logits), tuple(row.label for row in calibration)
    ).temperature
    save_checkpoint(
        destination / "checkpoint.pt",
        config,
        variant,
        schema,
        catalog,
        session.population,
        recipe.names,
        recipe.formulas,
        result,
        {},
        (temperature, temperature),
    )
    write_support(destination, access, refit)
    write_predictions(
        destination / "calibration.json", predictions(calibration, cal_logits, temperature)
    )
    write_predictions(
        destination / "development.json", predictions(development, dev_logits, temperature)
    )
    outputs = tuple(file_record(path, destination) for path in sorted(destination.iterdir()))
    manifest = RunManifest(
        run_id=destination.name,
        config=config,
        config_sha256=fingerprint(config),
        git_sha="b" * 40,
        lock_sha256="a" * 64,
        stage=config.stage,
        status="complete",
        eligible_for_comparison=True,
        inputs=session.population.snapshot_files,
        outputs=outputs,
        population=session.population,
        runtime=(
            ("variant_id", job.variant_id),
            ("seed", str(job.seed)),
            ("penalty", str(job.penalty)),
            ("phase", job.phase),
            ("code_sha256", code_digest(Path.cwd())),
        ),
        temperatures=(temperature, temperature),
        selected_epochs=(1,),
        scales=tuple(result.scales),
    )
    finish_run(destination, manifest)
