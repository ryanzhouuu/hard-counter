from dataclasses import replace
from datetime import date

import pytest
from experiments.common.slices import SliceMetadata, SliceSpec, build_slices, compare_slices
from experiments.tests.common.reporting_fixtures import prediction


def test_fixed_slices_use_comparator_bins_and_retain_empty_memberships() -> None:
    support = (0, 0, 1, 3, 4, 19, 20)
    rows = tuple(prediction(i, 0.1 + i / 10) for i in range(7))
    metadata = tuple(
        SliceMetadata(
            row.row_key,
            "era",
            support[i],
            i,
            int(i > 2),
            forms=("form",) if i == 0 else (),
            tower_a="tower",
        )
        for i, row in enumerate(rows)
    )
    spec = SliceSpec(
        days=(date(2026, 1, 1), date(2026, 1, 2)),
        eras=("era",),
        forms=("form", "absent"),
        towers=("tower", "absent"),
    )
    slices = build_slices(metadata, rows, spec)
    groups = dict(slices.members)
    for prefix in ("deck:", "support:", "pair:", "player:", "probability:", "history:"):
        assert sum(len(keys) for name, keys in groups.items() if name.startswith(prefix)) == len(
            rows
        )
    assert groups["form:identity:absent"] == ()
    assert len(groups["support:1_3"]) == 2
    candidates = tuple(replace(row, probability=0.9) for row in rows)
    reports = compare_slices(candidates, rows, slices)
    assert next(report for report in reports if report.name == "probability:band_0").row_count == 1
    assert next(report for report in reports if report.name == "tower:absent").comparison is None
    with pytest.raises(ValueError):
        compare_slices(candidates[:-1], rows[:-1], slices)
    with pytest.raises(ValueError):
        build_slices(metadata[:-1], rows, spec)
    with pytest.raises(ValueError):
        build_slices((*metadata, metadata[0]), rows, spec)


def test_slice_membership_does_not_depend_on_labels() -> None:
    rows = (prediction(0, 0.3), prediction(1, 0.8))
    metadata = tuple(SliceMetadata(row.row_key, "era", 0, 0, 0) for row in rows)
    spec = SliceSpec()
    assert build_slices(metadata, rows, spec) == build_slices(
        metadata, tuple(replace(row, label=1 - row.label) for row in rows), spec
    )


def test_slice_configuration_rejects_ambiguous_bins_and_negative_support() -> None:
    for boundaries in ((), (1,), (0, 0), (0, 10, 1)):
        with pytest.raises(ValueError):
            SliceSpec(history_boundaries=boundaries)
    with pytest.raises(ValueError):
        SliceSpec(eras=("era", "era"))
    with pytest.raises(ValueError):
        SliceMetadata(prediction(0).row_key, "era", -1, 0, 0)
