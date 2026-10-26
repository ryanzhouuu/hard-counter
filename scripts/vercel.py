"""Stage an ignored model for CLI uploads and publish Vite output to Vercel's CDN."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from clash_sos.application.live_model import score_live_decks
from clash_sos.infrastructure.settings import Settings

MODEL_DIRECTORY = Path("infra/vercel/model")


def stage_model(source: Path, destination: Path = MODEL_DIRECTORY) -> Path:
    """Validate an independent copy before replacing the previous deployment artifact."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Choose a model outside the staging directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-", dir=destination.parent) as workspace:
        staged = Path(workspace) / "model"
        shutil.copytree(source, staged, symlinks=True)
        score_live_decks(staged, [])
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)
    return destination


def prepare_public(
    frontend: Path = Path("frontend/dist"),
    model: Path = MODEL_DIRECTORY,
    destination: Path = Path("public"),
) -> None:
    """Fail before publishing assets when the CLI-uploaded model is missing or invalid."""
    if not (model / "manifest.json").is_file():
        raise FileNotFoundError(
            "Model missing: run pnpm vercel:stage before deploying with the CLI."
        )
    score_live_decks(model, [])
    if not (frontend / "index.html").is_file():
        raise FileNotFoundError("Frontend missing: run pnpm web:build first.")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(frontend, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage-model")
    stage.add_argument("source", type=Path, nargs="?")
    commands.add_parser("build")
    arguments = parser.parse_args()
    if arguments.command == "stage-model":
        source = arguments.source or Settings().active_model_path
        print(f"Staged model at {stage_model(source)}")
    else:
        prepare_public()


if __name__ == "__main__":
    main()
