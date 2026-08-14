"""Streaming archive operations for the Kaggle version 6 audit."""

from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from json import loads
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_CARD_MAPPING_NAME


class KaggleV6AuditError(ValueError):
    pass


def validate_archive_members(archive: ZipFile) -> list[ZipInfo]:
    members = sorted(archive.infolist(), key=lambda member: member.filename)
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise KaggleV6AuditError("archive member paths must be unique")
    for member in members:
        path = PurePosixPath(member.filename)
        if member.is_dir() or path.name != member.filename or path.name in {"", ".", ".."}:
            raise KaggleV6AuditError(f"unsafe archive member path: {member.filename}")
    mapping_count = names.count(KAGGLE_V6_CARD_MAPPING_NAME)
    if mapping_count != 1 or any(
        name != KAGGLE_V6_CARD_MAPPING_NAME and not name.endswith(".parquet") for name in names
    ):
        raise KaggleV6AuditError("archive must contain only Parquet files and cardToID.json")
    return members


@contextmanager
def extracted_member(
    archive: ZipFile, member: ZipInfo, work_directory: Path, chunk_size: int
) -> Generator[tuple[Path, str]]:
    destination = work_directory / member.filename
    digest = sha256()
    with archive.open(member) as source, destination.open("wb") as target:
        copied = 0
        while chunk := source.read(chunk_size):
            target.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
    if copied != member.file_size:
        raise KaggleV6AuditError(f"member size mismatch: {member.filename}")
    try:
        yield destination, digest.hexdigest()
    finally:
        destination.unlink(missing_ok=True)


def hash_file(path: Path, chunk_size: int) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def validate_card_mapping(path: Path) -> None:
    observed = loads(path.read_text(encoding="utf-8"))
    expected = {entry.source_name: entry.source_id for entry in KAGGLE_V6_CARDS.entries}
    if observed != expected:
        raise KaggleV6AuditError("cardToID.json does not match the versioned card catalog")
