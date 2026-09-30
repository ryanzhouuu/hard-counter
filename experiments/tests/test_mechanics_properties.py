from experiments.form_mechanics.features import extract as form_extract
from experiments.higher_order.features import extract as pattern_extract
from experiments.matchup_features.features import extract as matchup_extract
from experiments.mechanics.load import synthetic_catalog
from experiments.tower_mechanics.features import extract as tower_extract
from hypothesis import given
from hypothesis import strategies as st


@given(st.permutations(range(8)), st.permutations(range(8)), st.booleans())
def test_every_registered_feature_family_preserves_permutation_and_swap(
    left_order: list[int],
    right_order: list[int],
    swap: bool,
) -> None:
    catalog = synthetic_catalog()
    original = [[0, 1, 2, 3, 4, 5, 6, 7, 13], [1, 2, 3, 4, 5, 6, 7, 10, 16]]
    permuted = [
        [*[original[0][i] for i in left_order], 13],
        [*[original[1][i] for i in right_order], 16],
    ]
    if swap:
        permuted.reverse()
    for extractor in (matchup_extract, form_extract, tower_extract, pattern_extract):
        expected = extractor(original, catalog)
        actual = extractor(permuted, catalog)
        assert actual.names == expected.names
        assert actual.values == tuple((-1 if swap else 1) * v for v in expected.values)
