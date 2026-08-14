"""Executable schema contract for Kaggle version 6 Parquet members."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from json import dumps


@dataclass(frozen=True)
class KaggleSchemaColumn:
    name: str
    physical_type: str
    nullable: bool


PLAYER_COLUMNS = ("winner_id", "loser_id")
CARD_COLUMNS = tuple(f"{side}_card_{index}" for side in ("winner", "loser") for index in range(8))
LEVEL_COLUMNS = tuple(f"{name}_level" for name in CARD_COLUMNS)
ROW_COLUMNS = (*PLAYER_COLUMNS, "time", "game_mode", *CARD_COLUMNS, *LEVEL_COLUMNS)

KAGGLE_V6_SCHEMA = tuple(
    KaggleSchemaColumn(
        name=name,
        physical_type=(
            "TIMESTAMP WITH TIME ZONE"
            if name == "time"
            else "VARCHAR"
            if name in {*PLAYER_COLUMNS, "game_mode"}
            else "UTINYINT"
        ),
        nullable=True,
    )
    for name in ROW_COLUMNS
)


class IncompatibleKaggleSchemaError(ValueError):
    pass


def validate_kaggle_v6_schema(columns: Sequence[KaggleSchemaColumn]) -> str:
    observed = tuple(columns)
    if observed != KAGGLE_V6_SCHEMA:
        raise IncompatibleKaggleSchemaError("Parquet schema does not match Kaggle v6")
    payload = [
        {"name": column.name, "physical_type": column.physical_type, "nullable": column.nullable}
        for column in observed
    ]
    return sha256(dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
