"""Archive reuse must preserve the pinned release and publish only validated cache entries."""

import json
import tarfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from scripts import vercel_model


def model_archive(name: str = "model.txt") -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        content = b"model bytes"
        member = tarfile.TarInfo(name)
        member.size = len(content)
        archive.addfile(member, BytesIO(content))
    return buffer.getvalue()


def release_file(directory: Path, body: bytes, digest: str | None = None) -> Path:
    release = directory / "release.json"
    release.write_text(
        json.dumps(
            {
                "url": "https://example.private.blob.vercel-storage.com/model.tar.gz",
                "sha256": digest or sha256(body).hexdigest(),
            }
        )
    )
    return release


def mock_blob(
    monkeypatch: pytest.MonkeyPatch, body: bytes, status: int = 200
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []
    original_client = httpx.Client

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, content=body)

    def client(*, timeout: float) -> httpx.Client:
        return original_client(transport=httpx.MockTransport(respond), timeout=timeout)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
    monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "test-credential")
    return requests


def test_validated_archive_is_published_after_the_context_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = model_archive()
    cache = tmp_path / "cache"
    cached = cache / f"{sha256(body).hexdigest()}.tar.gz"
    requests = mock_blob(monkeypatch, body)

    with vercel_model.downloaded_model(release_file(tmp_path, body), cache=cache) as extracted:
        assert (extracted / "model.txt").read_bytes() == b"model bytes"
        assert not cached.exists()

    assert not extracted.exists()
    assert cached.read_bytes() == body
    assert list(cache.iterdir()) == [cached]
    assert len(requests) == 1


def test_verified_cache_hit_needs_no_network_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = model_archive()
    release = release_file(tmp_path, body)
    cache = tmp_path / "cache"
    requests = mock_blob(monkeypatch, body)
    with vercel_model.downloaded_model(release, cache=cache):
        pass
    monkeypatch.delenv("BLOB_READ_WRITE_TOKEN")

    with vercel_model.downloaded_model(release, cache=cache) as extracted:
        assert (extracted / "model.txt").read_bytes() == b"model bytes"

    assert len(requests) == 1


@pytest.mark.parametrize("damage", ["checksum", "oversized"])
def test_damaged_cached_bytes_are_rechecked_and_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    body = model_archive()
    release = release_file(tmp_path, body)
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = cache / f"{sha256(body).hexdigest()}.tar.gz"
    cached.write_bytes(b"corrupt" if damage == "checksum" else body + b"extra")
    monkeypatch.setattr(vercel_model, "MAX_MODEL_BYTES", len(body))
    requests = mock_blob(monkeypatch, body)

    with vercel_model.downloaded_model(release, cache=cache) as extracted:
        assert (extracted / "model.txt").read_bytes() == b"model bytes"

    assert cached.read_bytes() == body
    assert len(requests) == 1


@pytest.mark.parametrize("failure", ["checksum", "http", "credentials"])
def test_a_different_cached_release_cannot_mask_a_failed_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    body = model_archive()
    cache = tmp_path / "cache"
    cache.mkdir()
    previous = cache / f"{sha256(body).hexdigest()}.tar.gz"
    previous.write_bytes(body)
    requested_digest = "0" * 64
    release = release_file(tmp_path, body, requested_digest)
    mock_blob(monkeypatch, body, status=503 if failure == "http" else 200)
    if failure == "credentials":
        monkeypatch.delenv("BLOB_READ_WRITE_TOKEN")
    error = {
        "checksum": ValueError,
        "http": httpx.HTTPStatusError,
        "credentials": RuntimeError,
    }[failure]

    with pytest.raises(error), vercel_model.downloaded_model(release, cache=cache):
        pytest.fail("A different cached model must not satisfy the requested release")

    assert list(cache.iterdir()) == [previous]
    assert previous.read_bytes() == body


def test_caller_validation_failure_does_not_publish_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = model_archive()
    cache = tmp_path / "cache"
    mock_blob(monkeypatch, body)

    with (
        pytest.raises(ValueError, match="Model manifest invalid"),
        vercel_model.downloaded_model(release_file(tmp_path, body), cache=cache) as extracted,
    ):
        assert (extracted / "model.txt").is_file()
        raise ValueError("Model manifest invalid")

    assert not extracted.exists()
    assert list(cache.iterdir()) == []


def test_unsafe_archive_is_not_cached_even_with_the_expected_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = model_archive("../escape")
    cache = tmp_path / "cache"
    mock_blob(monkeypatch, body)

    with (
        pytest.raises(ValueError, match="Unsafe member"),
        vercel_model.downloaded_model(release_file(tmp_path, body), cache=cache),
    ):
        pytest.fail("An unsafe archive must not be extracted or cached")

    assert list(cache.iterdir()) == []
    assert not (tmp_path / "escape").exists()
