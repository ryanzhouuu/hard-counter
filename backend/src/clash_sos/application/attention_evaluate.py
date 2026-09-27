"""Score aligned attention cache rows and publish local evaluation sidecars.

The caller supplies a verified cache and train-only support index. This runner
never fits weights, and its output directory is removed after any failure.
"""

from collections.abc import Callable
from itertools import islice
from pathlib import Path
from shutil import rmtree
from typing import Literal, Self, cast

import numpy as np
import polars as pl
import torch
from pydantic import Field, model_validator

from clash_sos.application.attention_support import AttentionSupportIndex, iter_attention_sidecars
from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_protocol import AttentionProtocol, require_matching_fit
from clash_sos.domain.attention_schema import PROBABILITY_INTERPRETATION
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel
from clash_sos.domain.model_evaluation import MatchupEvaluation, MatchupEvaluationAccumulator
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import AttentionCache

Comparator = Callable[[np.ndarray, dict[str, object]], float]
EvaluationRole = Literal["development", "calibration", "reporting"]


class AttentionEvaluationReport(ManifestModel):
    """Identify scored rows, their fit provenance, and fixed slice definitions."""

    report_version: Literal[1] = 1
    fit_artifact_id: str = Field(min_length=1)
    role: EvaluationRole
    protocol: AttentionProtocol
    row_count: int = Field(gt=0)
    probability_interpretation: Literal[
        "deck-only matchup estimate under an equal-skill assumption",
        "deck-and-tower matchup estimate under an equal-skill assumption",
    ] = PROBABILITY_INTERPRETATION
    evaluation: MatchupEvaluation

    @model_validator(mode="after")
    def check_support(self) -> Self:
        """Reject reports whose evaluated population is incomplete."""
        declared = getattr(self.protocol, self.role)
        if declared is None or self.row_count != declared.row_count:
            raise ValueError("evaluation row count does not match protocol slice")
        if self.evaluation.overall.metrics.row_count != self.row_count:
            raise ValueError("evaluation metrics do not cover declared rows")
        return self


def _write_prediction_chunk(directory: Path, chunk: list[dict[str, object]], index: int) -> None:
    """Bound resident prediction metadata by flushing at most 50,000 rows."""
    pl.DataFrame(chunk).write_parquet(
        directory / f"predictions-{index:05d}.parquet", compression="zstd"
    )
    chunk.clear()


def evaluate_attention_cache(
    cache: AttentionCache,
    model: AttentionMatchupModel,
    *,
    fit_protocol: AttentionProtocol,
    fit_artifact_id: str,
    role: EvaluationRole,
    support: AttentionSupportIndex,
    output_directory: Path,
    batch_size: int = 1024,
    comparator: Comparator | None = None,
) -> AttentionEvaluationReport:
    """Write a report and aligned Parquet predictions without loading a split."""
    if batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    if model.schema_fingerprint != cache.manifest.encoding_sha256:
        raise ValueError("evaluation model and cache encoding do not match")
    require_matching_fit(fit_protocol, cache.manifest.protocol)
    if (support.fit_sha256, support.encoding_sha256) != (
        fit_protocol.fit_sha256(),
        cache.manifest.encoding_sha256,
    ):
        raise ValueError("evaluation support index does not match model fit")
    if output_directory.exists():
        raise FileExistsError(f"evaluation output already exists: {output_directory}")
    offset = next((item for item in cache.manifest.slices if item.role == role), None)
    if offset is None:
        raise ValueError(f"cache has no {role} slice")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    try:
        accumulator = MatchupEvaluationAccumulator()
        metadata = iter_attention_sidecars(cache, role)
        device = next(model.parameters()).device
        model.eval()
        predicted: list[dict[str, object]] = []
        count = 0
        chunk_index = 0
        for tokens, labels in cache.iter_batches(role, batch_size=batch_size):
            with torch.inference_mode():
                logits = model(torch.tensor(tokens, dtype=torch.long, device=device))
                probabilities = cast(
                    list[float],
                    torch.sigmoid(logits.to(device="cpu", dtype=torch.float64)).tolist(),  # type: ignore[reportUnknownMemberType]
                )
            sidecars = list(islice(metadata, len(labels)))
            if len(sidecars) != len(labels):
                raise ValueError("evaluation sidecars ended before cache arrays")
            for token_row, label, probability, row in zip(
                tokens, labels, probabilities, sidecars, strict=True
            ):
                if row.get("row_ordinal") != offset.start + count:
                    raise ValueError("evaluation sidecar and token ordinals diverged")
                first, second = row.get("deck_a_hash"), row.get("deck_b_hash")
                if not isinstance(first, str) or not isinstance(second, str):
                    raise ValueError("evaluation sidecar deck hash is invalid")
                a_support, b_support, pair_support = support.lookup(first, second)
                comparison = comparator(token_row, row) if comparator is not None else None
                difference = accumulator.update(
                    label=int(label),
                    probability=float(probability),
                    deck_a_support=a_support,
                    deck_b_support=b_support,
                    unordered_pair_support=pair_support,
                    comparator_probability=comparison,
                )
                predicted.append(
                    {
                        "row_ordinal": row["row_ordinal"],
                        "timestamp": row["timestamp"],
                        "fingerprint": row["fingerprint"],
                        "archive_member": row["archive_member"],
                        "row_number": row["row_number"],
                        "label": int(label),
                        "probability": float(probability),
                        "comparator_probability": comparison,
                        "paired_loss_difference": difference,
                    }
                )
                count += 1
                if len(predicted) == 50_000:
                    _write_prediction_chunk(output_directory, predicted, chunk_index)
                    chunk_index += 1
        if count != offset.stop - offset.start:
            raise ValueError("evaluation count does not match cache slice")
        if next(metadata, None) is not None:
            raise ValueError("evaluation sidecars exceed cache arrays")
        if predicted:
            _write_prediction_chunk(output_directory, predicted, chunk_index)
        report = AttentionEvaluationReport(
            probability_interpretation=model.probability_interpretation,
            fit_artifact_id=fit_artifact_id,
            role=role,
            protocol=cache.manifest.protocol,
            row_count=count,
            evaluation=accumulator.finalize(),
        )
        (output_directory / "report.json").write_bytes(
            canonical_json_bytes(report.model_dump(mode="python")) + b"\n"
        )
        return report
    except BaseException:
        rmtree(output_directory)
        raise
