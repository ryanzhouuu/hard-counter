"""Current official deck identities, separate from frozen Kaggle inputs."""

from importlib.resources import files
from json import loads

from clash_sos.domain.card_attributes import CardAttributeTable
from clash_sos.domain.card_catalog import CardCatalog

CURRENT_CARD_CATALOG = CardCatalog.from_payload(
    loads(files(__package__).joinpath("cards.json").read_text(encoding="utf-8"))
)
CURRENT_CARD_ATTRIBUTES = CardAttributeTable.from_payload(
    loads(files(__package__).joinpath("attributes.json").read_text(encoding="utf-8"))
)
