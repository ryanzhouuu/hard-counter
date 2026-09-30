"""Frozen response/cost groups in a side-swap antisymmetric registry."""

from collections.abc import Sequence

from experiments.matchup_features import cycle, responses
from experiments.mechanics.contracts import FeatureResult, MechanicsCatalog, decode_pair, difference


def extract(
    tokens: Sequence[Sequence[int]],
    catalog: MechanicsCatalog,
    *,
    response: bool = True,
    cost: bool = True,
) -> FeatureResult:
    left, right = decode_pair(tokens, catalog)
    names: list[str] = []
    values: list[float] = []
    formulas: list[str] = []
    for enabled, extractor in ((response, responses.directed), (cost, cycle.directed)):
        if enabled:
            order, forward, definitions = extractor(left, right)
            _, reverse, _ = extractor(right, left)
            result = difference(order, forward, reverse, definitions)
            names.extend(result.names)
            values.extend(result.values)
            formulas.extend(result.formulas)
    return FeatureResult(tuple(names), tuple(values), tuple(formulas))
