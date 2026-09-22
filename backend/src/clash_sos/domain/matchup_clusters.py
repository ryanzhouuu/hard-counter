"""Shrunk cluster-versus-cluster log-odds for one LightGBM column.

Centroids come from deck composition in the temporal-train fit prefix.
Win counts come from that same prefix. Later battles only look up the frozen table.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Self, cast

import numpy as np

from clash_sos.domain.matchup_baseline import laplace_probability, logit

CLUSTER_COUNT = 64
CLUSTER_BATCH_SIZE = 4096
CLUSTER_MAX_ITER = 100
CLUSTER_N_INIT = 1
_REASSIGNMENT_RATIO = 0.01


@dataclass(frozen=True)
class ClusterBattle:
    """One oriented battle. The label is 1 when side A won."""

    deck_a_hash: str
    deck_b_hash: str
    side_a: tuple[str, ...]
    side_b: tuple[str, ...]
    label: int


def _presence_indices(index: Mapping[str, int], side: Sequence[str]) -> tuple[int, ...]:
    """Return sorted known identity indexes. Unknown keys and repeats are omitted."""
    seen: set[int] = set()
    chosen: list[int] = []
    for key in side:
        slot = index.get(key)
        if slot is None or slot in seen:
            continue
        seen.add(slot)
        chosen.append(slot)
    chosen.sort()
    return tuple(chosen)


def _nearest_cluster(
    centroids: np.ndarray, centroid_sq: np.ndarray, indices: tuple[int, ...]
) -> int:
    """Return the lowest-index nearest centroid for a binary presence vector."""
    dots = centroids[:, list(indices)].sum(axis=1)
    distance = centroid_sq + len(indices) - 2.0 * dots
    return int(np.argmin(distance))


class ClusterMatchupTable:
    """Frozen centroids and directed win counts. Assignment is cached by card indexes."""

    def __init__(
        self,
        identities: Sequence[str],
        centroids: np.ndarray,
        wins: np.ndarray,
        *,
        alpha: float,
        seed: int,
        batch_size: int,
        max_iter: int,
        n_init: int,
    ) -> None:
        keys = tuple(identities)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("identities must be unique")
        if alpha <= 0:
            raise ValueError("smoothing alpha must be positive")
        if centroids.ndim != 2 or centroids.shape[1] != len(keys):
            raise ValueError("cluster centroids must match the identity count")
        if wins.shape != (centroids.shape[0], centroids.shape[0]):
            raise ValueError("cluster wins must be square")
        if centroids.shape[0] < 1:
            raise ValueError("cluster count must be positive")
        if np.any(wins < 0):
            raise ValueError("cluster wins must be non-negative")
        self.identities = keys
        self.alpha = float(alpha)
        self.seed = seed
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.n_init = n_init
        self._index = {key: slot for slot, key in enumerate(keys)}
        self._centroids = np.asarray(centroids, dtype=np.float64)
        self._centroid_sq = np.sum(self._centroids * self._centroids, axis=1)
        self._wins = np.asarray(wins, dtype=np.int64)
        self._assignment: dict[tuple[int, ...], int] = {}

    @property
    def cluster_count(self) -> int:
        return int(self._centroids.shape[0])

    def indices_for(self, side: Sequence[str]) -> tuple[int, ...]:
        return _presence_indices(self._index, side)

    def cluster_of(self, side: Sequence[str]) -> int | None:
        """Return the cached nearest centroid. An empty side has no cluster."""
        indices = self.indices_for(side)
        if not indices:
            return None
        cached = self._assignment.get(indices)
        if cached is not None:
            return cached
        cluster = _nearest_cluster(self._centroids, self._centroid_sq, indices)
        self._assignment[indices] = cluster
        return cluster

    def remember(self, side: Sequence[str], cluster: int) -> None:
        """Cache an assignment already computed for a prefix deck."""
        indices = self.indices_for(side)
        if not indices:
            return
        if not 0 <= cluster < self.cluster_count:
            raise ValueError("cluster index is outside the table")
        self._assignment[indices] = cluster

    def log_odds(self, side_a: Sequence[str], side_b: Sequence[str]) -> float:
        """Return logit(P(cluster A beats cluster B)). An empty side is 0."""
        cluster_a = self.cluster_of(side_a)
        cluster_b = self.cluster_of(side_b)
        if cluster_a is None or cluster_b is None:
            return 0.0
        wins_ab = int(self._wins[cluster_a, cluster_b])
        wins_ba = int(self._wins[cluster_b, cluster_a])
        probability = laplace_probability(wins_ab, wins_ab + wins_ba, alpha=self.alpha)
        return logit(probability)

    def wins_total(self) -> int:
        return int(self._wins.sum())

    def to_payload(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "centroids": self._centroids.tolist(),
            "cluster_count": self.cluster_count,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "seed": self.seed,
            "wins": self._wins.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: object, identities: Sequence[str]) -> Self:
        """Rebuild a table from the object stored under feature-schema clusters."""
        if not isinstance(payload, dict):
            raise ValueError("cluster payload must be an object")
        body = cast(dict[object, object], payload)
        centroids = _float_matrix(body.get("centroids"), "cluster centroids")
        wins = _int_matrix(body.get("wins"), "cluster wins")
        declared = body.get("cluster_count")
        if isinstance(declared, bool) or not isinstance(declared, int):
            raise ValueError("cluster count must match the centroid rows")
        if declared != centroids.shape[0]:
            raise ValueError("cluster count must match the centroid rows")
        return cls(
            identities,
            centroids,
            wins,
            alpha=_positive_float(body.get("alpha"), "alpha"),
            seed=_non_negative_int(body.get("seed"), "seed"),
            batch_size=_positive_int(body.get("batch_size"), "batch_size"),
            max_iter=_positive_int(body.get("max_iter"), "max_iter"),
            n_init=_positive_int(body.get("n_init"), "n_init"),
        )


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{field} must be positive")
    return float(value)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _float_matrix(value: object, field: str) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a matrix")
    rows: list[list[float]] = []
    width: int | None = None
    for row in cast(list[object], value):
        if not isinstance(row, list) or not row:
            raise ValueError(f"{field} must be a matrix")
        parsed: list[float] = []
        for item in cast(list[object], row):
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise ValueError(f"{field} must be numeric")
            parsed.append(float(item))
        if width is None:
            width = len(parsed)
        if len(parsed) != width:
            raise ValueError(f"{field} rows must have equal length")
        rows.append(parsed)
    return np.asarray(rows, dtype=np.float64)


def _int_matrix(value: object, field: str) -> np.ndarray:
    matrix = _float_matrix(value, field)
    if np.any(matrix != np.floor(matrix)):
        raise ValueError(f"{field} must be integers")
    return matrix.astype(np.int64)


def _record_deck(
    decks: dict[str, tuple[int, ...]],
    deck_hash: str,
    indices: tuple[int, ...],
) -> None:
    previous = decks.get(deck_hash)
    if previous is None:
        decks[deck_hash] = indices
        return
    if previous != indices:
        raise ValueError("deck hash has two card lists")


def _collect_prefix(
    battles: Iterable[ClusterBattle],
    index: Mapping[str, int],
    prefix_rows: int,
) -> dict[str, tuple[int, ...]]:
    decks: dict[str, tuple[int, ...]] = {}
    seen = 0
    for battle in battles:
        if seen >= prefix_rows:
            break
        if battle.label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        _record_deck(decks, battle.deck_a_hash, _presence_indices(index, battle.side_a))
        _record_deck(decks, battle.deck_b_hash, _presence_indices(index, battle.side_b))
        seen += 1
    if seen != prefix_rows:
        raise ValueError("fit prefix ended early")
    return decks


def _fit_centroids(
    decks: Mapping[str, tuple[int, ...]],
    identity_count: int,
    *,
    cluster_count: int,
    seed: int,
    batch_size: int,
    max_iter: int,
    n_init: int,
) -> tuple[np.ndarray, dict[str, int]]:
    populated = {deck_hash: indices for deck_hash, indices in decks.items() if indices}
    if len(populated) < cluster_count:
        raise ValueError("cluster count exceeds distinct prefix decks")
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-not-found]
    except ImportError as error:
        raise ValueError(
            "scikit-learn is required for this model; install the dev dependency group"
        ) from error
    ordered = sorted(populated)
    matrix = np.zeros((len(ordered), identity_count), dtype=np.float64)
    for row, deck_hash in enumerate(ordered):
        matrix[row, list(populated[deck_hash])] = 1.0
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        batch_size=batch_size,
        max_iter=max_iter,
        init="k-means++",
        n_init=n_init,  # type: ignore[arg-type]
        reassignment_ratio=_REASSIGNMENT_RATIO,
    )
    model.fit(matrix)  # type: ignore[no-untyped-call]
    centroids = np.asarray(model.cluster_centers_, dtype=np.float64)  # type: ignore[arg-type]
    centroid_sq = np.sum(centroids * centroids, axis=1)
    assignment = {
        deck_hash: _nearest_cluster(centroids, centroid_sq, populated[deck_hash])
        for deck_hash in ordered
    }
    return centroids, assignment


def _count_wins(
    battles: Iterable[ClusterBattle],
    decks: Mapping[str, tuple[int, ...]],
    assignment: Mapping[str, int],
    *,
    prefix_rows: int,
    cluster_count: int,
) -> np.ndarray:
    wins = np.zeros((cluster_count, cluster_count), dtype=np.int64)
    seen = 0
    for battle in battles:
        if seen >= prefix_rows:
            break
        if decks[battle.deck_a_hash] and decks[battle.deck_b_hash]:
            cluster_a = assignment[battle.deck_a_hash]
            cluster_b = assignment[battle.deck_b_hash]
            if battle.label == 1:
                wins[cluster_a, cluster_b] += 1
            else:
                wins[cluster_b, cluster_a] += 1
        seen += 1
    if seen != prefix_rows:
        raise ValueError("fit prefix ended early")
    return wins


def fit_prefix_matchups(
    battles: Callable[[], Iterable[ClusterBattle]],
    *,
    prefix_rows: int,
    identities: Sequence[str],
    cluster_count: int = CLUSTER_COUNT,
    alpha: float = 1.0,
    seed: int = 0,
    batch_size: int = CLUSTER_BATCH_SIZE,
    max_iter: int = CLUSTER_MAX_ITER,
    n_init: int = CLUSTER_N_INIT,
) -> ClusterMatchupTable:
    """Fit centroids and prefix win counts. The callable is read twice, in order.

    The first prefix_rows battles define both the clusters and the rates.
    Decks are sorted by hash before k-means. Assignment is nearest centroid.
    """
    if prefix_rows < 1:
        raise ValueError("fit prefix requires at least one row")
    if cluster_count < 1 or batch_size < 1 or max_iter < 1 or n_init < 1:
        raise ValueError("cluster settings must be positive")
    keys = tuple(identities)
    index = {key: slot for slot, key in enumerate(keys)}
    decks = _collect_prefix(battles(), index, prefix_rows)
    centroids, assignment = _fit_centroids(
        decks,
        len(keys),
        cluster_count=cluster_count,
        seed=seed,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
    )
    wins = _count_wins(
        battles(),
        decks,
        assignment,
        prefix_rows=prefix_rows,
        cluster_count=cluster_count,
    )
    table = ClusterMatchupTable(
        keys,
        centroids,
        wins,
        alpha=alpha,
        seed=seed,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
    )
    for deck_hash, cluster in assignment.items():
        slots = decks[deck_hash]
        side = tuple(keys[slot] for slot in slots)
        table.remember(side, cluster)
    return table
