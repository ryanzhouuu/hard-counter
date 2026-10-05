"""CLI preparation for normalized current-era official battle snapshots."""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from clash_sos.application.attention_dataset_prepare import prepare_official_attention_dataset
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.attention_dataset import OfficialSchemaVersion
from clash_sos.domain.attention_schema import AttentionModelConfig, build_attention_schema
from clash_sos.infrastructure.clash_royale.catalog import (
    CURRENT_CARD_ATTRIBUTES,
    CURRENT_CARD_CATALOG,
    CURRENT_TOWER_CATALOG,
)


def parse_snapshot_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise typer.BadParameter("snapshot bounds must be ISO-8601 timestamps") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter("snapshot bounds must be timezone-aware")
    return parsed


def prepare_official_attention(
    source: Annotated[Path, typer.Option(help="Normalized TowerBattleRow JSONL")],
    destination: Annotated[Path, typer.Option()],
    dataset_version: Annotated[str, typer.Option()],
    balance_era: Annotated[str, typer.Option()],
    start: Annotated[datetime, typer.Option(parser=parse_snapshot_datetime)],
    train_end: Annotated[datetime, typer.Option(parser=parse_snapshot_datetime)],
    validation_end: Annotated[datetime, typer.Option(parser=parse_snapshot_datetime)],
    end: Annotated[datetime, typer.Option(parser=parse_snapshot_datetime)],
    official_schema_version: Annotated[
        OfficialSchemaVersion, typer.Option()
    ] = "official-ranked16-schema:v3",
    network_config: Annotated[Path | None, typer.Option()] = None,
    watch_fraction: float = 0.1,
    mirror_seed: int = 0,
    memory_limit: str = "1GB",
    threads: int = 1,
    batch_rows: int = 10_000,
) -> None:
    """Freeze current card/tower inputs, split rows, and publish a resolved protocol."""
    try:
        network = (
            AttentionModelConfig()
            if network_config is None
            else AttentionModelConfig.model_validate_json(network_config.read_bytes())
        )
        schema = build_attention_schema(
            CURRENT_CARD_CATALOG.serialize(),
            attributes=CURRENT_CARD_ATTRIBUTES,
            tower_catalog=CURRENT_TOWER_CATALOG,
            official_schema_version=official_schema_version,
            balance_era_id=balance_era,
            network=network,
        )
        published = prepare_official_attention_dataset(
            source,
            destination,
            schema=schema,
            dataset_version=dataset_version,
            start=start,
            train_end=train_end,
            validation_end=validation_end,
            end=end,
            watch_fraction=watch_fraction,
            mirror_seed=mirror_seed,
            config=StagingConfig(memory_limit=memory_limit, threads=threads, batch_rows=batch_rows),
        )
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(published)
