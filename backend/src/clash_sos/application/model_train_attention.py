"""Coordinate one temporal attention fit from resolved inputs to publication.

The protocol file fixes row identities before fitting. The processed dataset is
read-only; failed output and support-index workspaces remain local and owned.
"""

from collections.abc import Callable
from pathlib import Path
from shutil import rmtree

from pydantic import ValidationError

from clash_sos.application.attention_evaluate import (
    AttentionEvaluationReport,
    EvaluationRole,
    evaluate_attention_cache,
)
from clash_sos.application.attention_fit import AttentionFitConfig, fit_attention_model
from clash_sos.application.attention_support import AttentionSupportIndex
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.model_train import require_publish_paths
from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.attention_schema import AttentionModelConfig, build_attention_schema
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.kaggle_v6.attention_io import build_attention_cache
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.ml.attention_artifact_io import finalize_attention_artifact


class AttentionTrainError(ValueError):
    """The requested fit inputs or configuration cannot produce this artifact."""


def _load_protocol(path: Path) -> AttentionProtocol:
    """Require a resolved JSON protocol rather than choosing row windows here."""
    if not path.is_file():
        raise AttentionTrainError(f"resolved protocol JSON is required: {path}")
    try:
        protocol = AttentionProtocol.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise AttentionTrainError("resolved attention protocol is invalid") from error
    if protocol.family != "temporal":
        raise AttentionTrainError(
            "player-disjoint attention fits require the later isolation runner"
        )
    return protocol


def _load_network(path: Path | None) -> AttentionModelConfig:
    """Use default architecture or a validated ablation configuration file."""
    if path is None:
        return AttentionModelConfig()
    if not path.is_file():
        raise AttentionTrainError(f"attention network JSON is required: {path}")
    try:
        return AttentionModelConfig.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise AttentionTrainError("attention network configuration is invalid") from error


def train_attention_artifact(
    dataset: Path,
    destination: Path,
    *,
    protocol_path: Path,
    cache_directory: Path,
    output_workspace: Path,
    fit_config: AttentionFitConfig,
    staging_config: StagingConfig,
    model_version: str,
    network_config_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Fit only protocol training rows and publish checked evaluation sidecars."""
    require_publish_paths(destination, output_workspace)
    protocol = _load_protocol(protocol_path)
    network = _load_network(network_config_path)
    schema = build_attention_schema(
        KAGGLE_V6_CARDS.serialize(), attributes=CARD_ATTRIBUTES, network=network
    )
    if schema.fingerprint() != protocol.encoding_sha256:
        raise AttentionTrainError("protocol encoding does not match the selected network")
    if progress is not None:
        progress(f"verifying attention inputs ({protocol.refit.row_count} refit rows)")
    cache = build_attention_cache(dataset, cache_directory, protocol, schema, config=staging_config)
    output_workspace.mkdir(parents=True)
    try:
        if progress is not None:
            progress(
                f"fitting attention model on {protocol.selection_fit.row_count} selection rows"
            )
        fit = fit_attention_model(schema, cache, fit_config)
        if progress is not None:
            progress(
                f"selected epoch {fit.selected_epoch}; refit on {protocol.refit.row_count} rows"
            )
        work_directory = output_workspace / "_working"
        work_directory.mkdir()
        support = AttentionSupportIndex.build(cache, work_directory / "support.db")
        try:
            reports: list[AttentionEvaluationReport] = []
            roles: tuple[EvaluationRole, ...] = (
                ("development", "reporting") if protocol.reporting is not None else ("development",)
            )
            for role in roles:
                if progress is not None:
                    progress(f"scoring attention {role}")
                evaluation_directory = output_workspace / "evaluations" / role
                report = evaluate_attention_cache(
                    cache,
                    fit.refit_model,
                    fit_protocol=protocol,
                    fit_artifact_id=model_version,
                    role=role,
                    support=support,
                    output_directory=evaluation_directory,
                    batch_size=fit_config.batch_size,
                )
                (evaluation_directory / "report.json").unlink()
                reports.append(report)
        finally:
            support.close()
            rmtree(work_directory, ignore_errors=True)
        if progress is not None:
            progress("publishing verified attention artifact")
        return finalize_attention_artifact(
            output_workspace,
            destination,
            schema=schema,
            protocol=protocol,
            fit=fit,
            reports=reports,
            model_version=model_version,
            chunk_size=staging_config.chunk_size,
        )
    finally:
        if output_workspace.exists():
            rmtree(output_workspace)
