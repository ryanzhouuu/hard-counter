"""Atomic same-filesystem publication of a processed dataset version."""

import shutil
from pathlib import Path


class KaggleV6PublishError(ValueError):
    pass


def require_same_filesystem(left: Path, right: Path) -> None:
    """Fail unless both paths exist on the same device; never copy across filesystems."""
    if left.stat().st_dev != right.stat().st_dev:
        raise KaggleV6PublishError("output workspace must be on the same filesystem as destination")


def check_prepare_preconditions(
    *,
    destination: Path,
    output_workspace: Path,
    staging_workspace: Path,
) -> None:
    """Refuse to start if the published version or workspaces already exist."""
    if destination.exists():
        raise KaggleV6PublishError("published dataset version already exists")
    if output_workspace.exists():
        raise KaggleV6PublishError("output workspace already exists")
    if staging_workspace.exists():
        raise KaggleV6PublishError("staging workspace already exists")
    require_same_filesystem(output_workspace.parent, destination.parent)


def publish_processed_version(output_workspace: Path, destination: Path) -> Path:
    """Rename a completed output workspace onto destination. Never overwrite."""
    if destination.exists():
        raise KaggleV6PublishError("published dataset version already exists")
    if not output_workspace.is_dir():
        raise KaggleV6PublishError("output workspace is required")
    require_same_filesystem(output_workspace.parent, destination.parent)
    output_workspace.replace(destination)
    return destination


def cleanup_prepare_workspaces(output_workspace: Path, staging_workspace: Path) -> None:
    """Delete prepare workspaces after failure. Never delete destination."""
    shutil.rmtree(output_workspace, ignore_errors=True)
    shutil.rmtree(staging_workspace, ignore_errors=True)
