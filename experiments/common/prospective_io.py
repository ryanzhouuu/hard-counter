"""Load a physically separate future JSONL population without constructing fit partitions."""

from pathlib import Path

from clash_sos.domain.attention_dataset import TowerBattleRowV2
from clash_sos.domain.attention_schema import AttentionCardSchema
from experiments.common.artifacts import file_record
from experiments.common.data_access import ReportAccess, ReportingContract, ResearchRow
from experiments.common.source_rows import encode_official_row, population_identity

MAX_JSONL_ROW_BYTES = 1_048_576


def load_reporting_jsonl(
    path: Path,
    schema: AttentionCardSchema,
    contract: ReportingContract,
    *,
    row_cap: int,
) -> ReportAccess:
    if row_cap <= 0:
        raise ValueError("reporting source loading requires a positive row cap")
    if schema.canonical_schema_version != "official-ranked16-schema:v2":
        raise ValueError("current prospective rows require the frozen official v2 schema")
    population = contract.population
    expected_files = population.snapshot_files
    if (
        len(expected_files) != 1
        or file_record(path, path.parent, row_count=expected_files[0].row_count)
        != expected_files[0]
    ):
        raise ValueError("prospective source bytes must match the separate reporting inventory")
    rows: list[ResearchRow] = []
    with path.open("rb") as source:
        while line := source.readline(MAX_JSONL_ROW_BYTES + 1):
            if len(line) > MAX_JSONL_ROW_BYTES:
                raise ValueError("prospective JSONL row exceeds byte limit")
            if len(rows) >= row_cap:
                raise ValueError("prospective source exceeds row cap")
            row = TowerBattleRowV2.model_validate_json(line)
            rows.append(encode_official_row(row, schema, population.mirror_seed))
    if expected_files[0].row_count is not None and len(rows) != expected_files[0].row_count:
        raise ValueError("prospective row count differs from reporting inventory")
    ordered = tuple(sorted(rows, key=lambda row: row.key))
    if len({row.key[1] for row in ordered}) != len(ordered):
        raise ValueError("prospective fingerprints must be unique")
    observed = population_identity(
        ordered,
        expected_files,
        schema,
        population.mirror_seed,
        start=population.start,
        end=population.end,
    )
    if observed != population:
        raise ValueError("prospective population inventory or frozen schema mismatch")
    return ReportAccess(contract, ordered)
