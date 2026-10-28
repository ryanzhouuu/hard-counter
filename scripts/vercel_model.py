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


@contextmanager
def downloaded_model(release_file: Path) -> Generator[Path, None, None]:
    """Yield extracted bytes only after checksum verification; remove them on exit."""
    release = ModelRelease.model_validate_json(release_file.read_bytes())
    with TemporaryDirectory(prefix="vercel-model-") as workspace:
        archive, extracted = Path(workspace) / "model.tar.gz", Path(workspace) / "model"
        _download(release, archive)
        _extract(archive, extracted)
        yield extracted
