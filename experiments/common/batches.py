from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from experiments.common.data_access import ResearchRow


def batch_indices(
    count: int, *, batch_size: int, shuffle: bool, seed: int = 0, epoch: int = 0
) -> Iterator[np.ndarray]:
    """Use one bounded block permutation for tokens, features, labels, and players."""
    if min(count, seed, epoch) < 0 or batch_size < 1:
        raise ValueError("positive batch size and nonnegative count/seed/epoch required")
    block_rows = max(batch_size, 8192)
    rng = np.random.default_rng(seed + epoch)
    blocks = np.arange((count + block_rows - 1) // block_rows)
    if shuffle:
        rng.shuffle(blocks)
    for block in blocks:
        first = int(block) * block_rows
        indices = np.arange(first, min(first + block_rows, count))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]


@dataclass(frozen=True)
class ResearchBatch:
    indices: np.ndarray
    tokens: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    players: tuple[tuple[str, str], ...]


def aligned_batches(
    rows: tuple[ResearchRow, ...],
    features: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    epoch: int = 0,
) -> Iterator[ResearchBatch]:
    if features.ndim != 2 or len(features) != len(rows) or not np.isfinite(features).all():
        raise ValueError("finite features must align with the row population")
    for indices in batch_indices(
        len(rows), batch_size=batch_size, shuffle=shuffle, seed=seed, epoch=epoch
    ):
        selected = [rows[int(i)] for i in indices]
        yield ResearchBatch(
            indices=indices,
            tokens=np.asarray([r.tokens for r in selected], dtype=np.int64),
            labels=np.asarray([r.label for r in selected], dtype=np.float32),
            features=np.asarray(features[indices], dtype=np.float32),
            players=tuple((r.player_a, r.player_b) for r in selected),
        )


def rms_scales(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or len(features) == 0 or not np.isfinite(features).all():
        raise ValueError("nonempty finite training features required")
    scales = np.sqrt(np.mean(np.square(features, dtype=np.float64), axis=0))
    return np.where(scales == 0, 1.0, scales)
