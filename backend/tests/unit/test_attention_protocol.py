"""Contract checks for declared attention fit and evaluation populations."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from clash_sos.domain.attention_protocol import (
    AttentionProtocol,
    AttentionSlice,
    digest_row_keys,
    require_matching_fit,
)

H = "a" * 64


def day(number: int) -> datetime:
    """Use short, explicit UTC dates for half-open protocol fixtures."""
    return datetime(2026, 6, number, tzinfo=UTC)


def slice_at(partition: str, start: int, end: int, count: int) -> AttentionSlice:
    """Build one fixture slice with valid but synthetic row provenance."""
    return AttentionSlice.model_validate(
        {
            "partition": partition,
            "start": day(start),
            "end": day(end),
            "row_count": count,
            "row_keys_sha256": H,
        }
    )


def protocol(**changes: object) -> AttentionProtocol:
    """Create a valid temporal protocol, then apply requested invalid cases."""
    payload: dict[str, object] = {
        "family": "temporal",
        "dataset_version": "kaggle-v6-ranked16-v2",
        "balance_era_id": "2026-06",
        "processed_manifest_sha256": H,
        "canonical_sha256": H,
        "split_file": "splits-temporal.parquet",
        "split_sha256": H,
        "encoding_sha256": H,
        "mirror_seed": 0,
        "selection_fit": slice_at("train", 2, 10, 30),
        "watch": slice_at("train", 10, 11, 5),
        "refit": slice_at("train", 2, 11, 35),
        "development": slice_at("validation", 11, 15, 8),
        "reporting": slice_at("test", 15, 17, 4),
    }
    payload.update(changes)
    return AttentionProtocol.model_validate(payload)


def test_raw_and_calibrated_protocols_share_fit_but_not_evaluation_rows() -> None:
    raw = protocol()
    calibrated = protocol(
        calibration=slice_at("validation", 11, 12, 2),
        development=slice_at("validation", 12, 15, 6),
    )
    assert raw.development.row_count == 8
    assert calibrated.development.row_count == 6
    assert raw.fit_sha256() == calibrated.fit_sha256()
    require_matching_fit(raw, calibrated)


def test_player_protocol_allows_time_overlap_across_disjoint_partitions() -> None:
    player = protocol(
        family="player_disjoint",
        split_file="splits-player-disjoint.parquet",
        development=slice_at("validation", 4, 16, 8),
        reporting=slice_at("test", 3, 17, 4),
    )
    assert player.family == "player_disjoint"
    with pytest.raises(ValueError, match="does not match model fit"):
        require_matching_fit(protocol(), player)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"watch": slice_at("train", 9, 11, 5)}, "contiguous"),
        ({"refit": slice_at("train", 2, 11, 34)}, "refit count"),
        ({"development": slice_at("validation", 10, 15, 8)}, "must follow refit"),
        ({"development": slice_at("test", 11, 15, 8)}, "cannot use the test"),
        ({"reporting": slice_at("validation", 15, 17, 4)}, "reporting must use"),
        ({"split_file": "splits-player-disjoint.parquet"}, "split file"),
        (
            {
                "calibration": slice_at("validation", 11, 13, 2),
                "development": slice_at("validation", 12, 15, 6),
            },
            "calibration must precede",
        ),
        (
            {
                "calibration": slice_at("validation", 10, 12, 2),
                "development": slice_at("validation", 12, 15, 6),
            },
            "temporal calibration must follow refit",
        ),
    ],
)
def test_protocol_rejects_invalid_boundaries_and_roles(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        protocol(**changes)


def test_slice_normalizes_offsets_and_rejects_naive_or_empty_bounds() -> None:
    central = day(12).astimezone(timezone(timedelta(hours=-5)))
    normalized = AttentionSlice(
        partition="validation", start=central, end=day(13), row_count=1, row_keys_sha256=H
    )
    assert normalized.start == day(12) and normalized.start.tzinfo is UTC
    with pytest.raises(ValidationError, match="timezone-aware"):
        AttentionSlice(
            partition="validation",
            start=datetime(2026, 6, 12),
            end=day(13),
            row_count=1,
            row_keys_sha256=H,
        )
    with pytest.raises(ValidationError, match="start must precede"):
        slice_at("validation", 12, 12, 1)


def test_row_digest_is_timezone_invariant_and_requires_sorted_unique_keys() -> None:
    first = (day(12), "a", "file.parquet", 1)
    second = (day(12), "b", "file.parquet", 2)
    central = day(12).astimezone(timezone(timedelta(hours=-5)))
    assert digest_row_keys((first, second)) == digest_row_keys(
        ((central, "a", "file.parquet", 1), second)
    )
    with pytest.raises(ValueError, match="unique and sorted"):
        digest_row_keys((second, first))
    with pytest.raises(ValueError, match="unique and sorted"):
        digest_row_keys((first, first))
    with pytest.raises(ValueError, match="timezone-aware"):
        digest_row_keys(((datetime(2026, 6, 12), "a", "file.parquet", 1),))
