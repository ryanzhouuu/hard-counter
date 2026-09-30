from datetime import timedelta

from experiments.common.cache import oriented_digest
from experiments.common.contracts import (
    FileRecord,
    PopulationIdentity,
    RunManifest,
    StudyConfig,
    Variant,
    fingerprint,
)
from experiments.tests.common.research_fixture import HASH, rows

from clash_sos.domain.attention_protocol import digest_row_keys


def population_identity() -> PopulationIdentity:
    population = rows()
    return PopulationIdentity(
        snapshot_files=(FileRecord(path="source.json", sha256=HASH, size_bytes=1),),
        row_keys_sha256=digest_row_keys(r.key for r in population),
        event_mapping_sha256=HASH,
        oriented_sha256=oriented_digest(population),
        mode="synthetic",
        era="synthetic",
        mirror_seed=0,
        start=population[0].key[0],
        end=population[-1].key[0] + timedelta(hours=1),
        catalog_sha256=HASH,
    )


def manifest(outputs: tuple[FileRecord, ...]) -> RunManifest:
    config = StudyConfig(study_id="synthetic", variants=(Variant(variant_id="A0"),))
    return RunManifest(
        run_id="test",
        config=config,
        config_sha256=fingerprint(config),
        git_sha="a" * 40,
        lock_sha256=HASH,
        stage=config.stage,
        status="complete",
        eligible_for_comparison=False,
        inputs=(),
        outputs=outputs,
        population=population_identity(),
        runtime=(("device", "cpu"),),
    )
