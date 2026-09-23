"""Build training-support slices from the aligned fixture sidecars."""

from pathlib import Path

import pytest
from test_attention_cache_read import write_cache

from clash_sos.application.attention_support import (
    AttentionSupportIndex,
    iter_attention_sidecars,
)
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import load_attention_cache


def test_support_index_reads_refit_rows_only_and_counts_unordered_pairs(tmp_path: Path) -> None:
    dataset, directory, schema, protocol = write_cache(tmp_path)
    cache = load_attention_cache(dataset, directory, protocol, schema, chunk_size=1024)
    train = list(iter_attention_sidecars(cache, "refit"))
    development = list(iter_attention_sidecars(cache, "development"))
    assert len(train) == 2
    assert len(development) == 2
    assert [row["row_ordinal"] for row in development] == [0, 1]
    a, b = train[0]["deck_a_hash"], train[0]["deck_b_hash"]
    assert isinstance(a, str) and isinstance(b, str)
    index = AttentionSupportIndex.build(cache, tmp_path / "support.sqlite")
    try:
        assert index.lookup(a, b) == (2, 2, 2)
        assert index.lookup(b, a) == (2, 2, 2)
        assert index.lookup(a, "unseen") == (2, 0, 0)
        assert index.lookup("unseen", "other") == (0, 0, 0)
    finally:
        index.close()
    with pytest.raises(FileExistsError, match="already exists"):
        AttentionSupportIndex.build(cache, tmp_path / "support.sqlite")
