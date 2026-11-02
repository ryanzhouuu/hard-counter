"""Fetch a checksum-pinned private model archive during deployment builds."""

import os
import tarfile
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MODEL_BYTES = 64 * 1024 * 1024


class ModelRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("url")
    @classmethod
    def private_blob_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not (parsed.hostname or "").endswith(".private.blob.vercel-storage.com")
            or parsed.netloc != parsed.hostname
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Model release must use an HTTPS private Vercel Blob URL.")
        return value


def _download(release: ModelRelease, archive: Path) -> None:
    token = os.environ.get("VERCEL_OIDC_TOKEN") or os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token:
        raise RuntimeError("Model download requires VERCEL_OIDC_TOKEN or BLOB_READ_WRITE_TOKEN.")
    digest, size = sha256(), 0
    with (
        httpx.Client(timeout=60) as client,
        client.stream("GET", release.url, headers={"Authorization": f"Bearer {token}"}) as response,
    ):
        response.raise_for_status()
        with archive.open("wb") as output:
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_MODEL_BYTES:
                    raise ValueError("Model archive exceeds the download size limit.")
                digest.update(chunk)
                output.write(chunk)
    if digest.hexdigest() != release.sha256:
        raise ValueError("Model archive checksum does not match the pinned release.")


def _extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:gz") as bundle:
        members = bundle.getmembers()
        if sum(member.size for member in members) > MAX_MODEL_BYTES:
            raise ValueError("Expanded model archive exceeds the size limit.")
        for member in members:
            target = (destination / member.name).resolve()
            if (
                not (member.isfile() or member.isdir())
                or Path(member.name).is_absolute()
                or not target.is_relative_to(destination.resolve())
            ):
                raise ValueError("Unsafe member in model archive.")
        bundle.extractall(destination, members=members, filter="data")


def _restore_archive(source: Path, destination: Path, expected_digest: str) -> bool:
    """Snapshot a bounded cache entry; a missing or mismatched archive is a cache miss."""
    digest, size = sha256(), 0
    try:
        with source.open("rb") as cached, destination.open("wb") as output:
            while chunk := cached.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MODEL_BYTES:
                    return False
                digest.update(chunk)
                output.write(chunk)
    except FileNotFoundError:
        return False
    return digest.hexdigest() == expected_digest


@contextmanager
def downloaded_model(release_file: Path, cache: Path | None = None) -> Generator[Path, None, None]:
    """Reverify cached bytes; atomically cache downloads only after caller validation succeeds."""
    release = ModelRelease.model_validate_json(release_file.read_bytes())
    cached = cache / f"{release.sha256}.tar.gz" if cache is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="vercel-model-", dir=cache) as workspace:
        archive, extracted = Path(workspace) / "model.tar.gz", Path(workspace) / "model"
        cache_hit = cached is not None and _restore_archive(cached, archive, release.sha256)
        if not cache_hit:
            _download(release, archive)
        _extract(archive, extracted)
        yield extracted
        if cached is not None and not cache_hit:
            archive.replace(cached)
