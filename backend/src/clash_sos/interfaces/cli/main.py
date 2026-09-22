"""Typer entry point for clash-sos. Dataset and model commands stay thin adapters."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import typer

from clash_sos import __version__
from clash_sos.application.dataset_prepare import prepare_kaggle_v6_dataset
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.dataset_verify import verify_kaggle_v6_dataset
from clash_sos.application.model_train import train_matchup_baseline
from clash_sos.application.model_train_pair import train_card_pair_model
from clash_sos.domain.matchup_baseline import DEFAULT_MIRROR_SEED, DEFAULT_SMOOTHING_ALPHA
from clash_sos.domain.matchup_lgbm import DEFAULT_NUM_THREADS
from clash_sos.domain.model_artifact import (
    DEFAULT_LIGHTGBM_MODEL_VERSION,
    DEFAULT_MODEL_VERSION,
    DEFAULT_PAIR_MODEL_VERSION,
)
from clash_sos.domain.processed_manifest import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_PLAYER_HASH_SEED,
    DEFAULT_PLAYER_TRAIN_MAX,
    DEFAULT_PLAYER_VALIDATION_MAX,
)
from clash_sos.infrastructure.kaggle_v6.audit import (
    audit_kaggle_v6_archive,
    write_dataset_manifest,
)
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_ARCHIVE_NAME

app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")


def parse_aware_datetime(value: str) -> datetime:
    """Parse a timezone-aware ISO-8601 timestamp for temporal cutovers."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise typer.BadParameter(
            "temporal cutovers must be timezone-aware ISO-8601 timestamps"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter("temporal cutovers must be timezone-aware ISO-8601 timestamps")
    return parsed


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo(__version__)


@dataset_app.command("audit-kaggle-v6")
def audit_kaggle_v6(
    archive: Path = Path("data/raw/kaggle") / KAGGLE_V6_ARCHIVE_NAME,
    output: Path = Path("data/metadata/kaggle-v6-dataset.json"),
    temp_directory: Path = Path("data/tmp"),
    memory_limit: str = "1GB",
    threads: int = 2,
) -> None:
    """Write the raw Kaggle v6 audit manifest after verifying archive identity."""
    manifest = audit_kaggle_v6_archive(
        archive,
        temp_directory=temp_directory,
        memory_limit=memory_limit,
        threads=threads,
    )
    write_dataset_manifest(manifest, output)
    typer.echo(output)


@dataset_app.command("inspect-kaggle-v6")
def inspect_kaggle_v6(
    archive: Path = Path("data/raw/kaggle") / KAGGLE_V6_ARCHIVE_NAME,
    temp_directory: Path = Path("data/tmp"),
    memory_limit: str = "1GB",
    threads: int = 2,
) -> None:
    """Print the Kaggle v6 audit manifest JSON without writing any files."""
    manifest = audit_kaggle_v6_archive(
        archive,
        temp_directory=temp_directory,
        memory_limit=memory_limit,
        threads=threads,
    )
    typer.echo(manifest.model_dump_json(by_alias=True, indent=2))


@dataset_app.command("prepare-kaggle-v6")
def prepare_kaggle_v6(
    train_end: Annotated[datetime, typer.Option(parser=parse_aware_datetime)],
    validation_end: Annotated[datetime, typer.Option(parser=parse_aware_datetime)],
    archive: Path = Path("data/raw/kaggle") / KAGGLE_V6_ARCHIVE_NAME,
    destination: Path = Path("data/processed") / DEFAULT_DATASET_VERSION,
    staging_workspace: Path = Path("data/tmp/kaggle-v6-staging"),
    output_workspace: Path = Path("data/tmp/kaggle-v6-output"),
    temp_directory: Path = Path("data/tmp"),
    raw_manifest: Path = Path("data/metadata/kaggle-v6-dataset.json"),
    memory_limit: str = "1GB",
    threads: int = 2,
    chunk_size: int = 8 * 1024 * 1024,
    batch_rows: int = 10_000,
    player_seed: int = DEFAULT_PLAYER_HASH_SEED,
    player_train_max: float = DEFAULT_PLAYER_TRAIN_MAX,
    player_validation_max: float = DEFAULT_PLAYER_VALIDATION_MAX,
    dataset_version: str = DEFAULT_DATASET_VERSION,
) -> None:
    """Publish one processed Kaggle v6 version. Fails if the destination already exists.

    Identity grouping uses 8GB DuckDB RAM regardless of --memory-limit.
    """
    published = prepare_kaggle_v6_dataset(
        archive,
        destination,
        staging_workspace=staging_workspace,
        output_workspace=output_workspace,
        temp_directory=temp_directory,
        train_end=train_end,
        validation_end=validation_end,
        config=StagingConfig(
            memory_limit=memory_limit,
            threads=threads,
            chunk_size=chunk_size,
            batch_rows=batch_rows,
        ),
        raw_manifest_path=raw_manifest,
        player_seed=player_seed,
        player_train_max=player_train_max,
        player_validation_max=player_validation_max,
        dataset_version=dataset_version,
    )
    typer.echo(published)


@dataset_app.command("verify-kaggle-v6")
def verify_kaggle_v6(
    dataset: Path = Path("data/processed") / DEFAULT_DATASET_VERSION,
    temp_directory: Path = Path("data/tmp"),
    memory_limit: str = "1GB",
    threads: int = 2,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Validate a published Kaggle v6 version in place without rewriting it."""
    verify_kaggle_v6_dataset(
        dataset,
        config=StagingConfig(
            memory_limit=memory_limit,
            threads=threads,
            chunk_size=chunk_size,
        ),
        temp_directory=temp_directory,
    )
    typer.echo(dataset)


@model_app.command("train")
def train(
    dataset: Path = Path("data/processed") / DEFAULT_DATASET_VERSION,
    destination: Annotated[Path | None, typer.Option()] = None,
    output_workspace: Path = Path("data/tmp/kaggle-v6-model-output"),
    temp_directory: Path = Path("data/tmp"),
    memory_limit: str = "1GB",
    threads: int = DEFAULT_NUM_THREADS,
    chunk_size: int = 8 * 1024 * 1024,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    mirror_seed: int = DEFAULT_MIRROR_SEED,
    model_version: Annotated[str | None, typer.Option()] = None,
    promoted_model: Literal["lightgbm", "card_pair", "card_log_odds"] = "lightgbm",
) -> None:
    """Fit a matchup model, evaluate splits, and write one artifact version.

    The default model is the LightGBM cluster-matchup booster. threads applies to that
    booster, while DuckDB joins stay on one thread.
    """
    if promoted_model == "lightgbm":
        from clash_sos.application.model_train_lgbm import train_lightgbm_model

        resolved_version = model_version or DEFAULT_LIGHTGBM_MODEL_VERSION
        resolved_destination = destination or Path("models") / resolved_version
        published = train_lightgbm_model(
            dataset,
            resolved_destination,
            output_workspace=output_workspace,
            temp_directory=temp_directory,
            config=StagingConfig(
                memory_limit=memory_limit,
                threads=1,
                chunk_size=chunk_size,
            ),
            smoothing_alpha=smoothing_alpha,
            mirror_seed=mirror_seed,
            model_version=resolved_version,
            num_threads=threads,
            progress=lambda message: typer.echo(message, err=True),
        )
    else:
        resolved_version = model_version or (
            DEFAULT_PAIR_MODEL_VERSION if promoted_model == "card_pair" else DEFAULT_MODEL_VERSION
        )
        resolved_destination = destination or Path("models") / resolved_version
        trainer = train_card_pair_model if promoted_model == "card_pair" else train_matchup_baseline
        published = trainer(
            dataset,
            resolved_destination,
            output_workspace=output_workspace,
            temp_directory=temp_directory,
            config=StagingConfig(
                memory_limit=memory_limit,
                threads=threads,
                chunk_size=chunk_size,
            ),
            smoothing_alpha=smoothing_alpha,
            mirror_seed=mirror_seed,
            model_version=resolved_version,
            progress=lambda message: typer.echo(message, err=True),
        )
    typer.echo(published)


@model_app.command("train-lgbm")
def train_lgbm(
    dataset: Path = Path("data/processed") / DEFAULT_DATASET_VERSION,
    destination: Annotated[Path | None, typer.Option()] = None,
    output_workspace: Path = Path("data/tmp/kaggle-v6-model-output"),
    temp_directory: Path = Path("data/tmp"),
    memory_limit: str = "1GB",
    threads: int = DEFAULT_NUM_THREADS,
    chunk_size: int = 8 * 1024 * 1024,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    mirror_seed: int = DEFAULT_MIRROR_SEED,
    model_version: str = DEFAULT_LIGHTGBM_MODEL_VERSION,
) -> None:
    """Fit the LightGBM cluster-matchup model and write one artifact version.

    threads applies to LightGBM. DuckDB joins stay on one thread. This command
    does not change the card-pair model published by model train.
    """
    from clash_sos.application.model_train_lgbm import train_lightgbm_model

    resolved_destination = destination or Path("models") / model_version
    published = train_lightgbm_model(
        dataset,
        resolved_destination,
        output_workspace=output_workspace,
        temp_directory=temp_directory,
        config=StagingConfig(
            memory_limit=memory_limit,
            threads=1,
            chunk_size=chunk_size,
        ),
        smoothing_alpha=smoothing_alpha,
        mirror_seed=mirror_seed,
        model_version=model_version,
        num_threads=threads,
        progress=lambda message: typer.echo(message, err=True),
    )
    typer.echo(published)
