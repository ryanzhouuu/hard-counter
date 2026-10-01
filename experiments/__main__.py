import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    if importlib.util.find_spec("clash_sos") is None:
        raise RuntimeError("install the backend before running experiments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/experiments"))
    parser.add_argument("--models", type=Path, default=Path("models/experiments"))
    args = parser.parse_args()
    if args.synthetic:
        from experiments.common.cli import STUDIES
        from experiments.common.cli import main as study_main

        for study in STUDIES.values():
            study_main(
                study,
                [
                    "smoke",
                    "--synthetic",
                    "--output",
                    str(args.output),
                    "--models",
                    str(args.models),
                ],
            )
    else:
        print("Experiment workspace imports successfully")


if __name__ == "__main__":
    main()
