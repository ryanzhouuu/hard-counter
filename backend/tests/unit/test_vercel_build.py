"""Deployment staging preserves verified model bytes and separates them from public assets."""

import json
import sys
import tarfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from scripts.vercel import main, prepare_public, stage_model

from clash_sos.application.live_model import LiveModelError
from clash_sos.domain.model_artifact import ModelArtifactManifest, ModelOutputFile
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


@pytest.fixture
def model(tmp_path: Path) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    members = {
        "card_catalog": ("catalog.json", KAGGLE_V6_CARDS.serialize()),
        "evaluation": ("evaluation.json", b"{}"),
        "feature_schema": ("feature-schema.json", b'{"feature_schema_version":"card-log-odds:v1"}'),
        "predictor": ("predictor.json", b'{"effects":{}}'),
    }
    files: list[ModelOutputFile] = []
    for kind, (name, body) in members.items():
        (directory / name).write_bytes(body)
        files.append(
            ModelOutputFile.model_validate(
                {
                    "path": name,
                    "kind": kind,
                    "size_bytes": len(body),
                    "sha256": sha256(body).hexdigest(),
                }
            )
        )
    manifest = ModelArtifactManifest(
        model_version="deployment-test",
        dataset_version="data",
        catalog_version=KAGGLE_V6_CARDS.version,
        balance_era_id="2026-06",
        files=tuple(sorted(files, key=lambda item: item.path)),
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json())
    return directory


def snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_staging_copies_the_complete_selected_artifact(tmp_path: Path, model: Path) -> None:
    destination = tmp_path / "staged"

    assert stage_model(model, destination) == destination
    assert snapshot(destination) == snapshot(model)


def test_invalid_replacement_preserves_the_staged_model(tmp_path: Path, model: Path) -> None:
    destination = stage_model(model, tmp_path / "staged")
    previous = snapshot(destination)
    (model / "catalog.json").write_bytes(b"corrupt")

    with pytest.raises(LiveModelError, match="model_unavailable"):
        stage_model(model, destination)

    assert snapshot(destination) == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source", "staged"]


@pytest.mark.parametrize("relationship", ["same", "parent", "child"])
def test_staging_rejects_overlapping_directories(model: Path, relationship: str) -> None:
    destination = {"same": model, "parent": model.parent, "child": model / "nested"}[relationship]
    previous = snapshot(model)

    with pytest.raises(ValueError, match="outside the staging directory"):
        stage_model(model, destination)

    assert snapshot(model) == previous


def test_public_output_contains_only_current_frontend_assets(tmp_path: Path, model: Path) -> None:
    frontend, public = tmp_path / "frontend", tmp_path / "public"
    (frontend / "assets").mkdir(parents=True)
    (frontend / "index.html").write_text("<html>app</html>")
    (frontend / "assets" / "app-hash.js").write_text("console.log('app')")
    public.mkdir()
    (public / "old.js").write_text("stale")

    prepare_public(frontend, model, public)

    assert snapshot(public) == snapshot(frontend)
    assert not (public / "manifest.json").exists()


def test_build_fails_without_a_cli_uploaded_model(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "index.html").write_text("previous")

    with pytest.raises(FileNotFoundError, match="pnpm vercel:stage"):
        prepare_public(tmp_path / "frontend", tmp_path / "missing-model", public)

    assert (public / "index.html").read_text() == "previous"


def test_build_rejects_a_corrupt_model_before_publishing_assets(
    tmp_path: Path, model: Path
) -> None:
    (model / "catalog.json").write_bytes(b"corrupt")
    public = tmp_path / "public"

    with pytest.raises(LiveModelError, match="model_unavailable"):
        prepare_public(tmp_path / "frontend", model, public)

    assert not public.exists()


def test_build_requires_the_compiled_frontend(tmp_path: Path, model: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pnpm web:build"):
        prepare_public(tmp_path / "frontend", model, tmp_path / "public")


def archive_model(model: Path) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(model.rglob("*")):
            archive.add(path, arcname=path.relative_to(model), recursive=False)
    return buffer.getvalue()


def write_release(tmp_path: Path, body: bytes) -> Path:
    release = tmp_path / "model-release.json"
    release.write_text(
        json.dumps(
            {
                "url": "https://example.private.blob.vercel-storage.com/model.tar.gz",
                "sha256": sha256(body).hexdigest(),
            }
        )
    )
    return release


def mock_blob(monkeypatch: pytest.MonkeyPatch, body: bytes, status: int = 200) -> None:
    original_client = httpx.Client

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-credential"
        return httpx.Response(status, content=body)

    def client(*, timeout: float) -> httpx.Client:
        return original_client(transport=httpx.MockTransport(respond), timeout=timeout)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
    monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "test-credential")


@pytest.mark.parametrize("existing", [False, True])
def test_git_build_provisions_the_pinned_model(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    body = archive_model(model)
    release = write_release(tmp_path, body)
    mock_blob(monkeypatch, body)
    staged, frontend, public = (tmp_path / name for name in ("staged", "frontend", "public"))
    if existing:
        staged.mkdir()
        (staged / "manifest.json").write_text("stale cached model")
    frontend.mkdir()
    (frontend / "index.html").write_text("frontend")

    prepare_public(frontend, staged, public, release=release)

    assert snapshot(staged) == snapshot(model)
    assert snapshot(public) == snapshot(frontend)


def test_download_prefers_build_oidc_over_the_static_token(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.vercel_model import downloaded_model

    body = archive_model(model)
    mock_blob(monkeypatch, body)
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "test-credential")
    monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "unused-static-token")

    with downloaded_model(write_release(tmp_path, body)) as extracted:
        assert snapshot(extracted) == snapshot(model)
    assert not extracted.exists()


def test_build_command_downloads_without_a_staged_model(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = archive_model(model)
    release = write_release(tmp_path, body)
    mock_blob(monkeypatch, body)
    config = tmp_path / "infra/vercel/model-release.json"
    config.parent.mkdir(parents=True)
    release.replace(config)
    frontend = tmp_path / "frontend/dist"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text("frontend")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vercel", "build"])

    main()

    assert snapshot(tmp_path / "infra/vercel/model") == snapshot(model)
    assert snapshot(tmp_path / "public") == snapshot(frontend)


@pytest.mark.parametrize("failure", ["checksum", "http", "manifest", "credentials"])
def test_failed_download_preserves_previous_model_and_frontend(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    staged = stage_model(model, tmp_path / "staged")
    previous = snapshot(staged)
    if failure == "manifest":
        (model / "catalog.json").write_text("corrupt")
    body = archive_model(model)
    release = write_release(tmp_path, body)
    mock_blob(monkeypatch, body, status=403 if failure == "http" else 200)
    if failure == "checksum":
        release.write_text(release.read_text().replace(sha256(body).hexdigest(), "0" * 64))
    if failure == "credentials":
        monkeypatch.delenv("BLOB_READ_WRITE_TOKEN")
    public = tmp_path / "public"
    public.mkdir()
    (public / "index.html").write_text("previous")
    error = {
        "checksum": ValueError,
        "http": httpx.HTTPStatusError,
        "manifest": LiveModelError,
        "credentials": RuntimeError,
    }[failure]

    with pytest.raises(error):
        prepare_public(tmp_path / "frontend", staged, public, release=release)

    assert snapshot(staged) == previous
    assert (public / "index.html").read_text() == "previous"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.private.blob.vercel-storage.com/model.tar.gz",
        "https://example.com/model.tar.gz",
        "https://user@example.private.blob.vercel-storage.com/model.tar.gz",
        "https://example.private.blob.vercel-storage.com/model.tar.gz?token=secret",
    ],
)
def test_download_rejects_untrusted_urls(tmp_path: Path, url: str) -> None:
    from scripts.vercel_model import downloaded_model

    release = write_release(tmp_path, b"")
    release.write_text(json.dumps({"url": url, "sha256": "0" * 64}))

    with (
        pytest.raises(ValueError, match="HTTPS private Vercel Blob URL"),
        downloaded_model(release),
    ):
        pytest.fail("Untrusted release was accepted")


@pytest.mark.parametrize("kind", ["parent", "absolute", "symlink", "hardlink"])
def test_archive_rejects_escaping_paths_and_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from scripts.vercel_model import downloaded_model

    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        name = {"parent": "../escaped", "absolute": str(tmp_path / "escaped")}.get(kind, "link")
        member = tarfile.TarInfo(name)
        if kind in {"symlink", "hardlink"}:
            member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "../escaped"
        archive.addfile(member)
    body = buffer.getvalue()
    mock_blob(monkeypatch, body)

    with (
        pytest.raises(ValueError, match="Unsafe member"),
        downloaded_model(write_release(tmp_path, body)),
    ):
        pytest.fail("Unsafe archive was accepted")
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("limit", [1, 1024])
def test_download_bounds_compressed_and_expanded_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    from scripts import vercel_model

    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("large")
        member.size = 4096
        archive.addfile(member, BytesIO(b"0" * member.size))
    body = buffer.getvalue()
    mock_blob(monkeypatch, body)
    monkeypatch.setattr(vercel_model, "MAX_MODEL_BYTES", limit)

    with (
        pytest.raises(ValueError, match="size limit"),
        vercel_model.downloaded_model(write_release(tmp_path, body)),
    ):
        pytest.fail("Oversized archive was accepted")
