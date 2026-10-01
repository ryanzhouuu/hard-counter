from collections import Counter

import pytest
from experiments.common.contracts import Role
from experiments.common.data_access import Operation
from experiments.common.synthetic import synthetic_smoke_population
from experiments.higher_order.features import eligibility, extract
from experiments.mechanics.contracts import decode_pair


def test_native_schema_tokens_mechanics_and_all_roles_align() -> None:
    access, catalog, schema = synthetic_smoke_population()
    assert not schema.network.neural_component
    assert schema.input_size == 9
    assert schema.identity_vocab == tuple(sorted(schema.identity_vocab))
    assert tuple(
        catalog.entries[token].identity for token in range(len(schema.identity_vocab))
    ) == (schema.identity_vocab)
    tower_tokens: set[int] = set()
    roles: tuple[tuple[Role, Operation], ...] = (
        ("selection_fit", "fit"),
        ("watch", "fit"),
        ("refit", "fit"),
        ("calibration", "calibrate"),
        ("development", "compare"),
    )
    for role, operation in roles:
        rows = access.read(role, operation)
        assert len(rows) == (64 if role == "refit" else 32)
        assert Counter(r.label for r in rows) == {0: len(rows) // 2, 1: len(rows) // 2}
        for row in rows:
            left, right = decode_pair(row.tokens, catalog)
            assert left[-1].kind == right[-1].kind == "tower"
            assert row.tokens[0][-1] in schema.tower_indices
            tower_tokens.update((row.tokens[0][-1], row.tokens[1][-1]))
    assert tower_tokens == schema.tower_indices
    again, same_catalog, same_schema = synthetic_smoke_population()
    assert again.protocol == access.protocol
    assert same_catalog.digest == catalog.digest
    assert same_schema.fingerprint() == schema.fingerprint()


def test_synthetic_boundaries_preserve_ties_and_pattern_support() -> None:
    access, catalog, _ = synthetic_smoke_population(64)
    fit = access.read("selection_fit", "fit")
    watch = access.read("watch", "fit")
    assert fit[-1].key[0] < watch[0].key[0]
    assert fit[-1].key[0] == fit[-2].key[0]
    eligibility_result = eligibility(
        [extract(row.tokens, catalog) for row in fit],
        role="selection_fit",
        minimum_support=2,
    )
    assert all(eligibility_result.active)
    assert eligibility_result.support == (8, 16, 4)


@pytest.mark.parametrize("count", [32, 65, 0])
def test_invalid_synthetic_population_size_rejected(count: int) -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        synthetic_smoke_population(count)
