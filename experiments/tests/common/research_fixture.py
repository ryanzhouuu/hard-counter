from datetime import UTC, datetime, timedelta

from experiments.common.data_access import ResearchRow

from clash_sos.domain.attention_protocol import AttentionProtocol, AttentionSlice, digest_row_keys

START = datetime(2026, 1, 1, tzinfo=UTC)
HASH = "a" * 64


def rows(count: int = 24) -> tuple[ResearchRow, ...]:
    result: list[ResearchRow] = []
    for i in range(count):
        a = (0, 1, 2, 3, 4, 5, 6, 7, 13 + i % 4)
        b = (1, 2, 3, 4, 5, 6, 7, 10, 13 + (i + 1) % 4)
        row = ResearchRow(
            (START + timedelta(hours=i // 2), f"{i:064x}", "synthetic", i),
            f"event-{i}",
            f"player-{i % 5}",
            f"opponent-{i % 7}",
            int(i % 3 != 0),
            (a, b),
        )
        result.append(row.swapped() if i % 2 else row)
    return tuple(result)


def protocol(population: tuple[ResearchRow, ...]) -> AttentionProtocol:
    n = len(population)
    cuts = (0, n // 3, n // 2, 3 * n // 4, n)
    end = population[-1].key[0] + timedelta(hours=1)

    def part(first: int, stop: int, name: str) -> AttentionSlice:
        return AttentionSlice(
            partition="train" if name in ("selection_fit", "watch", "refit") else "validation",
            start=population[first].key[0],
            end=population[stop].key[0] if stop < n else end,
            row_count=stop - first,
            row_keys_sha256=digest_row_keys(r.key for r in population[first:stop]),
        )

    return AttentionProtocol(
        family="temporal",
        dataset_version="synthetic",
        balance_era_id="synthetic",
        processed_manifest_sha256=HASH,
        canonical_sha256=HASH,
        split_sha256=HASH,
        split_file="splits-temporal.parquet",
        encoding_sha256=HASH,
        mirror_seed=0,
        selection_fit=part(cuts[0], cuts[1], "selection_fit"),
        watch=part(cuts[1], cuts[2], "watch"),
        refit=part(cuts[0], cuts[2], "refit"),
        calibration=part(cuts[2], cuts[3], "calibration"),
        development=part(cuts[3], cuts[4], "development"),
    )
