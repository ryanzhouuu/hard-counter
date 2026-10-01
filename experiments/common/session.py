from dataclasses import asdict, dataclass
from pathlib import Path

from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.attention_schema import (
    AttentionCardSchema,
    AttentionModelConfig,
    build_attention_schema,
)
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.card_attributes import CardAttributeTable
from experiments.common.artifacts import file_record
from experiments.common.attention_reference import require_identical_reference_population
from experiments.common.contracts import PopulationIdentity, StudyConfig
from experiments.common.data_access import RoleAccess
from experiments.common.source_io import load_snapshot
from experiments.common.source_rows import population_identity
from experiments.common.synthetic import synthetic_smoke_population
from experiments.mechanics.contracts import MechanicsCatalog
from experiments.mechanics.load import bind_tokens, load, load_partial_catalog


@dataclass(frozen=True)
class Session:
    access: RoleAccess
    population: PopulationIdentity
    catalog: MechanicsCatalog
    schema: AttentionCardSchema
    dataset: Path | None = None
    cache: Path | None = None


def prepare_session(
    config: StudyConfig,
    source_root: Path,
    *,
    synthetic: bool,
    dataset: Path | None,
    protocol: Path | None,
    schema: Path | None,
    cache: Path | None,
    mechanics: Path | None,
    row_cap: int,
) -> Session:
    if synthetic:
        access, catalog, input_schema = synthetic_smoke_population(
            min(row_cap, config.smoke_row_cap)
        )
        rows = (
            *access.read("refit", "fit"),
            *access.read("calibration", "calibrate"),
            *access.read("development", "compare"),
        )
        source_root.mkdir(parents=True, exist_ok=True)
        path = source_root / "synthetic.json"
        payload = canonical_json_bytes(tuple(asdict(r) for r in rows))
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("synthetic source destination has a different population")
        if not path.exists():
            path.write_bytes(payload)
        population = population_identity(
            rows,
            (file_record(path, source_root, row_count=len(rows)),),
            input_schema,
            0,
            start=access.protocol.refit.start,
            end=access.protocol.development.end,
        )
        return Session(access, population, catalog, input_schema)
    if dataset is None or protocol is None or schema is None or cache is None:
        raise ValueError("snapshot runs require dataset, protocol, schema, and cache paths")
    access, population, input_schema = load_snapshot(
        dataset, protocol, schema, cache, row_cap=row_cap
    )
    catalog = bind_tokens(
        load(mechanics) if mechanics else load_partial_catalog(), input_schema.identity_vocab
    )
    return Session(access, population, catalog, input_schema, dataset, cache)


def attention_session(session: Session, source_root: Path, row_cap: int) -> Session:
    schema = session.schema
    network = (
        AttentionModelConfig()
        if not session.catalog.synthetic
        else schema.network.model_copy(update={"neural_component": True})
    )
    native = build_attention_schema(
        canonical_json_bytes(schema.catalog_snapshot),
        attributes=CardAttributeTable.from_payload(schema.attribute_snapshot),
        network=network,
        balance_era_id=schema.balance_era_id,
        tower_catalog=schema.tower_catalog,
        official_schema_version=schema.canonical_schema_version,  # type: ignore[arg-type]
    )
    protocol = AttentionProtocol.model_validate(
        {**session.access.protocol.model_dump(), "encoding_sha256": native.fingerprint()}
    )
    original = (
        *session.access.read("refit", "fit"),
        *session.access.read("calibration", "calibrate"),
        *session.access.read("development", "compare"),
    )
    if session.dataset is None:
        access = RoleAccess(protocol, original)
    else:
        source_root.mkdir(parents=True, exist_ok=True)
        schema_path = source_root / "attention-schema.json"
        protocol_path = source_root / "attention-protocol.json"
        schema_path.write_bytes(canonical_json_bytes(native.model_dump()))
        protocol_path.write_bytes(canonical_json_bytes(protocol.model_dump()))
        assert session.cache is not None
        access, population, native = load_snapshot(
            session.dataset,
            protocol_path,
            schema_path,
            session.cache.with_name(session.cache.name + "-attention"),
            row_cap=row_cap,
        )
        if population != session.population:
            raise ValueError("architecture-specific cache changed population identity")
    scored = (
        *access.read("refit", "fit"),
        *access.read("calibration", "calibrate"),
        *access.read("development", "compare"),
    )
    require_identical_reference_population(schema, original, native, scored)
    return Session(
        access, session.population, session.catalog, native, session.dataset, session.cache
    )
