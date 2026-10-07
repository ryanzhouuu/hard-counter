"""Run local diagnostics on an explicitly bounded immutable search snapshot."""

import argparse
import json
from pathlib import Path

import torch

from experiments.common.contracts import Role, StudyConfig, Variant
from experiments.common.data_access import Operation
from experiments.common.session import prepare_session
from experiments.model_diagnostics.batch import run_batch
from experiments.model_diagnostics.report import write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mechanics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-cap", type=int, required=True)
    parser.add_argument(
        "--check", action="store_true", help="Verify inputs and sizes without fitting"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = StudyConfig(study_id="diagnostic", variants=(Variant(variant_id="baseline"),))
    session = prepare_session(
        config,
        args.output / "inputs",
        synthetic=False,
        dataset=args.dataset,
        protocol=args.protocol,
        schema=args.schema,
        cache=args.cache,
        mechanics=args.mechanics,
        row_cap=args.row_cap,
    )
    roles: tuple[tuple[Role, Operation], ...] = (
        ("selection_fit", "fit"),
        ("watch", "fit"),
        ("refit", "fit"),
        ("calibration", "calibrate"),
        ("development", "compare"),
    )
    if args.check:
        print(
            json.dumps(
                {
                    "status": "verified",
                    "population": session.population.model_dump(mode="json"),
                    "roles": {
                        role: len(session.access.read(role, operation)) for role, operation in roles
                    },
                    "maximum_fits": 26,
                    "budget_hours": 8,
                },
                indent=2,
            )
        )
        return
    result = run_batch(args.output, session, Path(__file__).resolve().parents[2])
    print(json.dumps(result, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)
    print(write_summary(args.output))


if __name__ == "__main__":
    main()
