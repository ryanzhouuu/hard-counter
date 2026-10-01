"""Support bins are configuration inputs fixed before controlled comparisons."""

from typing import Self

from pydantic import model_validator

from clash_sos.domain.manifests import ManifestModel


class SupportBins(ManifestModel):
    prior_history_edges: tuple[int, ...]
    deck_switching_edges: tuple[int, ...]

    @model_validator(mode="after")
    def validate_edges(self) -> Self:
        for edges in (self.prior_history_edges, self.deck_switching_edges):
            if not edges or any(edge < 0 for edge in edges) or tuple(sorted(set(edges))) != edges:
                raise ValueError("support boundaries must be nonnegative, unique, and increasing")
        return self
