"""Selected card files are validated before any cache or training output is created."""

from pathlib import Path

import pytest
from catalog_fixture import expanded_catalog

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.card_inputs import (
    CardInputError,
    load_card_attributes,
    load_card_catalog,
)
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def test_card_inputs_default_to_packaged_snapshots_and_load_expanded_files(tmp_path: Path) -> None:
    assert load_card_catalog() is KAGGLE_V6_CARDS
    assert load_card_attributes() is CARD_ATTRIBUTES
    catalog, attributes = expanded_catalog(1)
    path = tmp_path / "cards.json"
    path.write_bytes(catalog.serialize())
    assert load_card_catalog(path) == catalog
    path.write_bytes(canonical_json_bytes(attributes.to_payload()))
    assert load_card_attributes(path).to_payload() == attributes.to_payload()


@pytest.mark.parametrize("content", [None, "{", "{}"])
def test_card_input_errors_identify_the_selected_file(tmp_path: Path, content: str | None) -> None:
    path = tmp_path / "input.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(CardInputError, match="invalid card catalog"):
        load_card_catalog(path)
    with pytest.raises(CardInputError, match="invalid card attributes"):
        load_card_attributes(path)
