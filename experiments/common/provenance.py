import platform
import subprocess
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from experiments.common.artifacts import file_record


def revision(root: Path) -> tuple[str, str | None, str]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=root, check=True, capture_output=True
    ).stdout
    return (
        sha,
        sha256(changed).hexdigest() if changed else None,
        file_record(root / "uv.lock", root).sha256,
    )


def versions() -> tuple[tuple[str, str], ...]:
    return (
        ("python", platform.python_version()),
        *((name, version(name)) for name in ("numpy", "torch", "scipy", "duckdb", "polars")),
    )


def code_digest(root: Path) -> str:
    digest = sha256()
    paths = sorted(
        (
            *root.glob("backend/src/**/*.py"),
            *root.glob("experiments/**/*.py"),
            *root.glob("experiments/configs/*.json"),
            *root.glob("experiments/mechanics/inputs/*.json"),
            *root.glob("experiments/mechanics/evidence/*.json"),
        )
    )
    for path in paths:
        if "tests" not in path.relative_to(root).parts:
            digest.update(
                path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
            )
    return digest.hexdigest()
