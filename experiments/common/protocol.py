from datetime import datetime
from pathlib import Path

import duckdb

from clash_sos.application.attention_protocol_resolve import resolve_attention_slice
from clash_sos.domain.attention_protocol import AttentionProtocol


def search_protocol(
    original: AttentionProtocol, dataset: Path, *, calibration_end: datetime
) -> AttentionProtocol:
    """Resolve validation halves without exposing the physical test partition."""
    development = original.development
    if not development.start < calibration_end < development.end:
        raise ValueError("calibration boundary must split validation into nonempty intervals")
    with duckdb.connect() as connection:
        calibration = resolve_attention_slice(
            connection,
            canonical_path=dataset / "canonical.parquet",
            split_path=dataset / original.split_file,
            partition="validation",
            start=development.start,
            end=calibration_end,
        )
        resolved = resolve_attention_slice(
            connection,
            canonical_path=dataset / "canonical.parquet",
            split_path=dataset / original.split_file,
            partition="validation",
            start=calibration_end,
            end=development.end,
        )
    return AttentionProtocol.model_validate(
        {
            **original.model_dump(),
            "calibration": calibration.model_dump(),
            "development": resolved.model_dump(),
            "reporting": None,
        }
    )


def require_search(protocol: AttentionProtocol) -> None:
    if protocol.reporting is not None:
        raise ValueError("search protocol must exclude reporting")
    if protocol.calibration is None:
        raise ValueError("search requires a separate nonempty calibration slice")
