import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.contracts import FileRecord, RunManifest


def file_record(path: Path, root: Path, *, row_count: int | None = None) -> FileRecord:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return FileRecord(
        path=path.relative_to(root).as_posix(),
        sha256=digest.hexdigest(),
        size_bytes=path.stat().st_size,
        row_count=row_count,
    )


def verify_files(root: Path, files: tuple[FileRecord, ...]) -> None:
    for expected in files:
        path = root / expected.path
        if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
            raise ValueError(f"missing or unsafe member: {expected.path}")
        if file_record(path, root, row_count=expected.row_count) != expected:
            raise ValueError(f"member hash mismatch: {expected.path}")


@contextmanager
def publication(destination: Path) -> Generator[Path]:
    """Own the staging directory and serialize publishers before an atomic rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    staging: Path | None = None
    try:
        if destination.exists():
            raise FileExistsError("research destination already exists")
        staging = Path(tempfile.mkdtemp(prefix=".research-", dir=destination.parent))
        yield staging
        manifest = load_run(staging)
        if not manifest.outputs:
            raise ValueError("published runs require inventoried output")
        if destination.exists():
            raise FileExistsError("research destination already exists")
        staging.rename(destination)
        staging = None
    finally:
        os.close(descriptor)
        lock.unlink()
        if staging is not None:
            shutil.rmtree(staging)


def finish_run(directory: Path, manifest: RunManifest) -> None:
    verify_files(directory, manifest.outputs)
    members = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if members != {f.path for f in manifest.outputs}:
        raise ValueError("all output members must be inventoried before completion")
    path = directory / "manifest.json"
    with path.open("xb") as output:
        output.write(canonical_json_bytes(manifest.model_dump(mode="python")))
        output.flush()
        os.fsync(output.fileno())


def load_run(directory: Path) -> RunManifest:
    manifest = RunManifest.model_validate_json((directory / "manifest.json").read_bytes())
    verify_files(directory, manifest.outputs)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != {"manifest.json", *(f.path for f in manifest.outputs)}:
        raise ValueError("run contains uninventoried members")
    return manifest


def resume_matches(
    directory: Path, config_sha256: str, inputs: tuple[FileRecord, ...], input_root: Path
) -> bool:
    if not directory.exists():
        return False
    run = load_run(directory)
    verify_files(input_root, inputs)
    return run.status == "complete" and run.config_sha256 == config_sha256 and run.inputs == inputs
