"""Typer entry point for clash-sos. Dataset commands stay thin service adapters."""

from pathlib import Path

import typer

from clash_sos import __version__
from clash_sos.infrastructure.kaggle_v6.audit import (
    audit_kaggle_v6_archive,
    write_dataset_manifest,
)
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_ARCHIVE_NAME

app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


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
