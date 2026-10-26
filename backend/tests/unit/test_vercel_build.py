"""Deployment staging preserves verified model bytes and separates them from public assets."""

from hashlib import sha256
from pathlib import Path

import pytest
from scripts.vercel import prepare_public, stage_model

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
