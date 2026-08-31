"""Application workflow that stages adapted Kaggle v6 rows without publishing."""

import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from zipfile import ZipFile

from pydantic import Field

from clash_sos.domain.canonical import (
    BalanceEraRegistry,
    BattleOutcome,
    BattleSide,
    DomainModel,
    RecordState,
)
from clash_sos.domain.canonical_dataset import canonical_json_bytes, deck_content_hash
from clash_sos.domain.manifests import DatasetManifest
from clash_sos.domain.staged_dataset import (
    POPULATION_OBSERVATION_ISSUES,
    StagedBattleRow,
    UnadaptableRow,
)
from clash_sos.infrastructure.kaggle_v6.adapter import (
    AdaptedKaggleRecord,
    SourceRowLocation,
    adapt_row,
)
from clash_sos.infrastructure.kaggle_v6.audit import audit_kaggle_v6_archive, write_dataset_manifest
from clash_sos.infrastructure.kaggle_v6.audit_io import (
    extracted_member,
    hash_file,
    validate_archive_members,
    validate_card_mapping,
)
from clash_sos.infrastructure.kaggle_v6.audit_profile import read_parquet_schema
from clash_sos.infrastructure.kaggle_v6.balance_eras import KAGGLE_V6_ERA_REGISTRY
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS, KaggleCardCatalog
from clash_sos.infrastructure.kaggle_v6.schema import (
    IncompatibleKaggleSchemaError,
    validate_kaggle_v6_schema,
)
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_CARD_MAPPING_NAME
from clash_sos.infrastructure.kaggle_v6.staging_io import (
    KaggleV6StagingError,
    connect_staging_duckdb,
    directory_byte_size,
    list_parquet_members,
    part_path,
    row_mapping,
    write_staged_part,
    write_unadaptable_part,
)

__all__ = ["KaggleV6StagingError", "StagingConfig", "StagingResult", "stage_kaggle_v6"]


class StagingConfig(DomainModel):
    memory_limit: str = Field(min_length=1, default="1GB")
    threads: int = Field(ge=1, default=2)
    chunk_size: int = Field(ge=1, default=8 * 1024 * 1024)
    batch_rows: int = Field(ge=1, default=10_000)
    parquet_row_group_rows: Literal[131072] = 131072


@dataclass(frozen=True)
class StagingResult:
    source_row_count: int
    staged_row_count: int
    unadaptable_row_count: int
    members: tuple[str, ...]
    staging_files: tuple[Path, ...]
    unadaptable_files: tuple[Path, ...]
    file_sha256: tuple[tuple[str, str], ...]
    logical_staging_sha256: str
    logical_unadaptable_sha256: str
    rows_per_second: float
    temp_disk_bytes: int


def _side_fields(
    side: BattleSide,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], str]:
    card_ids = tuple(card.card_id.value for card in side.deck.cards)
    card_forms = tuple(card.form.value for card in side.deck.cards)
    levels = side.card_levels
    deck_hash = deck_content_hash(card_ids, card_forms, levels)
    return card_ids, card_forms, levels, deck_hash


def adapted_to_staged_row(record: AdaptedKaggleRecord) -> StagedBattleRow:
    """Map an adapted record with a battle into a staging row."""
    if record.battle is None:
        raise KaggleV6StagingError("adapted record has no battle to stage")
    battle = record.battle
    observation_issues = tuple(
        sorted(
            (
                issue
                for issue in record.disposition.issues
                if issue in POPULATION_OBSERVATION_ISSUES
            ),
            key=lambda issue: issue.value,
        )
    )

    side_a_ids, side_a_forms, side_a_levels, side_a_hash = _side_fields(battle.side_a)
    side_b_ids, side_b_forms, side_b_levels, side_b_hash = _side_fields(battle.side_b)

    return StagedBattleRow(
        source_id=battle.source_id,
        timestamp=battle.timestamp,
        mode=battle.mode,
        balance_era_id=record.balance_era_id,
        outcome=BattleOutcome.SIDE_A_WIN,
        event_key=battle.event_key,
        fingerprint=battle.fingerprint,
        side_a_player_id=battle.side_a.player_id,
        side_b_player_id=battle.side_b.player_id,
        side_a_card_ids=side_a_ids,
        side_a_card_forms=side_a_forms,
        side_a_card_levels=side_a_levels,
        side_a_deck_hash=side_a_hash,
        side_b_card_ids=side_b_ids,
        side_b_card_forms=side_b_forms,
        side_b_card_levels=side_b_levels,
        side_b_deck_hash=side_b_hash,
        observation_issues=observation_issues,
        archive_member=record.location.archive_member,
        row_number=record.location.row_number,
    )


def adapted_to_unadaptable_row(record: AdaptedKaggleRecord) -> UnadaptableRow:
    """Map an unadaptable adapted record into a compact ledger row without detail text."""
    if record.battle is not None:
        raise KaggleV6StagingError("adapted record with a battle cannot be unadaptable")
    state = record.disposition.state
    if state not in {RecordState.INVALID, RecordState.QUARANTINED}:
        raise KaggleV6StagingError("unadaptable rows must be invalid or quarantined")
    if state is RecordState.INVALID:
        unadaptable_state = RecordState.INVALID
    else:
        unadaptable_state = RecordState.QUARANTINED
    return UnadaptableRow(
        archive_member=record.location.archive_member,
        row_number=record.location.row_number,
        state=unadaptable_state,
        issues=record.disposition.issues,
    )


def ensure_raw_audit_manifest(
    archive_path: Path,
    raw_manifest_path: Path,
    *,
    temp_directory: Path,
    config: StagingConfig,
    expected_archive_size: int | None,
    expected_archive_sha256: str | None,
) -> DatasetManifest:
    """Reuse or create the raw audit manifest after verifying archive identity."""
    size, digest = hash_file(archive_path, config.chunk_size)
    if expected_archive_size is not None and size != expected_archive_size:
        raise KaggleV6StagingError("archive size does not match the pinned Kaggle v6 source")
    if expected_archive_sha256 is not None and digest != expected_archive_sha256:
        raise KaggleV6StagingError("archive checksum does not match the pinned Kaggle v6 source")
    if raw_manifest_path.exists():
        manifest = DatasetManifest.model_validate_json(
            raw_manifest_path.read_text(encoding="utf-8")
        )
        if manifest.archive.size_bytes != size or manifest.archive.sha256 != digest:
            raise KaggleV6StagingError("raw audit manifest does not match archive")
        return manifest
    manifest = audit_kaggle_v6_archive(
        archive_path,
        temp_directory=temp_directory,
        memory_limit=config.memory_limit,
        threads=config.threads,
        chunk_size=config.chunk_size,
        expected_archive_size=expected_archive_size,
        expected_archive_sha256=expected_archive_sha256,
    )
    write_dataset_manifest(manifest, raw_manifest_path)
    return manifest


def stage_kaggle_v6(
    archive_path: Path,
    workspace: Path,
    *,
    temp_directory: Path,
    config: StagingConfig | None = None,
    era_registry: BalanceEraRegistry = KAGGLE_V6_ERA_REGISTRY,
    catalog: KaggleCardCatalog = KAGGLE_V6_CARDS,
    members: Sequence[str] | None = None,
    raw_manifest_path: Path | None = None,
    expected_archive_size: int | None = None,
    expected_archive_sha256: str | None = None,
) -> StagingResult:
    """Scan one archive member at a time and write bounded staging partitions."""
    config = config or StagingConfig()
    if raw_manifest_path is not None:
        ensure_raw_audit_manifest(
            archive_path,
            raw_manifest_path,
            temp_directory=temp_directory,
            config=config,
            expected_archive_size=expected_archive_size,
            expected_archive_sha256=expected_archive_sha256,
        )
    temp_directory.mkdir(parents=True, exist_ok=True)
    if workspace.exists():
        raise KaggleV6StagingError("staging workspace already exists")

    started = time.perf_counter()
    workspace.mkdir(parents=True)
    peak_disk = 0
    source_row_count = 0
    staged_row_count = 0
    unadaptable_row_count = 0
    staged_digest = sha256()
    unadaptable_digest = sha256()
    staging_files: list[Path] = []
    unadaptable_files: list[Path] = []
    selected_members: tuple[str, ...] = ()

    try:
        with (
            ZipFile(archive_path) as archive,
            TemporaryDirectory(prefix="clash-sos-stage-", dir=temp_directory) as work,
        ):
            work_path = Path(work)
            connection = connect_staging_duckdb(
                memory_limit=config.memory_limit,
                threads=config.threads,
                temp_directory=work_path,
            )
            try:
                members_info = validate_archive_members(archive)
                mapping_member = next(
                    info for info in members_info if info.filename == KAGGLE_V6_CARD_MAPPING_NAME
                )
                with extracted_member(archive, mapping_member, work_path, config.chunk_size) as (
                    mapping_path,
                    _,
                ):
                    validate_card_mapping(mapping_path)

                parquet_names = list_parquet_members(archive)
                if members is not None:
                    requested = tuple(members)
                    if len(requested) != len(set(requested)):
                        raise KaggleV6StagingError("member names must be unique")
                    unknown = [name for name in requested if name not in parquet_names]
                    if unknown:
                        raise KaggleV6StagingError(
                            f"unknown archive members: {', '.join(sorted(unknown))}"
                        )
                    requested_set = set(requested)
                    selected = tuple(name for name in parquet_names if name in requested_set)
                else:
                    selected = parquet_names
                selected_members = selected

                parquet_infos = {info.filename: info for info in members_info}
                for name in selected:
                    with extracted_member(
                        archive, parquet_infos[name], work_path, config.chunk_size
                    ) as (extracted, _):
                        observed = read_parquet_schema(connection, extracted)
                        try:
                            validate_kaggle_v6_schema(observed)
                        except IncompatibleKaggleSchemaError as error:
                            raise KaggleV6StagingError(f"incompatible schema in {name}") from error

                        cursor = connection.execute(
                            "SELECT * FROM read_parquet(?)", [str(extracted)]
                        )
                        columns = [str(item[0]) for item in cursor.description]
                        row_number = 0
                        staged_part = 0
                        unadaptable_part = 0

                        while True:
                            batch = cursor.fetchmany(config.batch_rows)
                            if not batch:
                                break
                            staged_batch: list[StagedBattleRow] = []
                            unadaptable_batch: list[UnadaptableRow] = []
                            for values in batch:
                                record = adapt_row(
                                    row_mapping(columns, values),
                                    location=SourceRowLocation(
                                        archive_member=name, row_number=row_number
                                    ),
                                    era_registry=era_registry,
                                    catalog=catalog,
                                )
                                if record.battle is None:
                                    row = adapted_to_unadaptable_row(record)
                                    unadaptable_batch.append(row)
                                    unadaptable_digest.update(
                                        canonical_json_bytes(row.model_dump(mode="python")) + b"\n"
                                    )
                                    unadaptable_row_count += 1
                                else:
                                    row = adapted_to_staged_row(record)
                                    staged_batch.append(row)
                                    staged_digest.update(
                                        canonical_json_bytes(row.model_dump(mode="python")) + b"\n"
                                    )
                                    staged_row_count += 1
                                row_number += 1

                            if staged_batch:
                                path = part_path(workspace, "staging", name, staged_part)
                                write_staged_part(
                                    path,
                                    staged_batch,
                                    row_group_rows=config.parquet_row_group_rows,
                                )
                                staging_files.append(path)
                                staged_part += 1
                            if unadaptable_batch:
                                path = part_path(workspace, "unadaptable", name, unadaptable_part)
                                write_unadaptable_part(
                                    path,
                                    unadaptable_batch,
                                    row_group_rows=config.parquet_row_group_rows,
                                )
                                unadaptable_files.append(path)
                                unadaptable_part += 1
                            peak_disk = max(
                                peak_disk,
                                directory_byte_size(workspace) + directory_byte_size(work_path),
                            )
                        source_row_count += row_number
            finally:
                connection.close()
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise

    if staged_row_count + unadaptable_row_count != source_row_count:
        raise KaggleV6StagingError("staged and unadaptable counts must reconcile with source rows")

    elapsed = time.perf_counter() - started
    rows_per_second = 0.0 if source_row_count == 0 else source_row_count / max(elapsed, 1e-9)
    all_files = [*staging_files, *unadaptable_files]
    file_sha256 = tuple(
        sorted(
            (path.relative_to(workspace).as_posix(), hash_file(path, config.chunk_size)[1])
            for path in all_files
        )
    )
    return StagingResult(
        source_row_count=source_row_count,
        staged_row_count=staged_row_count,
        unadaptable_row_count=unadaptable_row_count,
        members=selected_members,
        staging_files=tuple(staging_files),
        unadaptable_files=tuple(unadaptable_files),
        file_sha256=file_sha256,
        logical_staging_sha256=staged_digest.hexdigest(),
        logical_unadaptable_sha256=unadaptable_digest.hexdigest(),
        rows_per_second=rows_per_second,
        temp_disk_bytes=peak_disk,
    )
