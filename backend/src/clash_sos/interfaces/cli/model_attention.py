"""Thin CLI adapter for explicit, resolved attention experiments.

Neural imports stay inside the command so legacy CLI help and commands remain
available with the default torch-free installation.
"""

from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.attention_artifact import DEFAULT_ATTENTION_MODEL_VERSION
from clash_sos.domain.processed_manifest import DEFAULT_DATASET_VERSION
from clash_sos.infrastructure.ml.attention_runtime import (
    NeuralRuntimeUnavailable,
    initialize_attention_runtime,
)


def train_attention(
    protocol: Annotated[Path, typer.Option(help="Resolved attention protocol JSON")],
    dataset: Path = Path("data/processed") / DEFAULT_DATASET_VERSION,
    destination: Annotated[Path | None, typer.Option()] = None,
    cache_directory: Path = Path("data/cache/attention-temporal-v1"),
    output_workspace: Path = Path("data/tmp/attention-model-output"),
    network_config: Annotated[Path | None, typer.Option()] = None,
    catalog: Annotated[
        Path | None, typer.Option(help="Card catalog JSON; defaults to June")
    ] = None,
    attributes: Annotated[
        Path | None, typer.Option(help="Card attributes JSON; defaults to June")
    ] = None,
    model_version: str = DEFAULT_ATTENTION_MODEL_VERSION,
    seed: int = 0,
    device: Literal["auto", "cpu", "mps"] = "auto",
    batch_size: int = 256,
    max_epochs: int = 20,
    patience: int = 3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    time_limit_seconds: Annotated[float | None, typer.Option()] = None,
    memory_limit: str = "1GB",
    threads: int = 1,
    chunk_size: int = 8 * 1024 * 1024,
    batch_rows: int = 10_000,
) -> None:
    """Fit a deck-only model and publish one version with scored row sidecars."""
    try:
        initialize_attention_runtime(seed, device)
        from clash_sos.application.attention_fit import AttentionFitConfig
        from clash_sos.application.model_train_attention import train_attention_artifact

        fit_config = AttentionFitConfig(
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            seed=seed,
            device=device,
            time_limit_seconds=time_limit_seconds,
        )
        staging_config = StagingConfig(
            memory_limit=memory_limit,
            threads=threads,
            chunk_size=chunk_size,
            batch_rows=batch_rows,
        )
    except (NeuralRuntimeUnavailable, ValidationError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    resolved = destination or Path("models") / model_version
    try:
        published = train_attention_artifact(
            dataset,
            resolved,
            protocol_path=protocol,
            cache_directory=cache_directory,
            output_workspace=output_workspace,
            fit_config=fit_config,
            staging_config=staging_config,
            model_version=model_version,
            network_config_path=network_config,
            catalog_path=catalog,
            attributes_path=attributes,
            progress=lambda message: typer.echo(message, err=True),
        )
    except (ValueError, TimeoutError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(published)
