"""Initialize the optional neural runtime without burdening legacy commands.

PyTorch is imported only when attention work begins. Callers can therefore load
the CLI, API, and existing predictors without installing the `ml` extra.
"""

import platform
import random
from dataclasses import dataclass
from typing import Literal

MAX_SEED = 2**32 - 1
ML_INSTALL_COMMAND = "uv sync --locked --dev --extra ml"
SelectedDevice = Literal["cpu", "mps"]


class NeuralRuntimeUnavailable(RuntimeError):
    """Report a missing dependency or explicitly requested accelerator."""


@dataclass(frozen=True)
class AttentionRuntime:
    """Capture initialized runtime facts for later artifact provenance."""

    seed: int
    device: SelectedDevice
    python_version: str
    torch_version: str


def initialize_attention_runtime(
    seed: int,
    requested_device: str = "auto",
) -> AttentionRuntime:
    """Seed neural libraries and select CPU or MPS without silent fallback.

    Seeds must fit NumPy's unsigned 32-bit global seed. An explicit MPS request
    raises `NeuralRuntimeUnavailable` when the backend cannot be used.
    """
    if isinstance(seed, bool) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer from 0 through {MAX_SEED}")
    if requested_device not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")

    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise NeuralRuntimeUnavailable(
            f"attention models require the 'ml' extra; run `{ML_INSTALL_COMMAND}`"
        ) from error

    mps_available = torch.backends.mps.is_available()
    if requested_device == "mps" and not mps_available:
        raise NeuralRuntimeUnavailable("MPS was requested but is not available")

    selected_device: SelectedDevice
    if requested_device == "mps" or (requested_device == "auto" and mps_available):
        selected_device = "mps"
    else:
        selected_device = "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore[reportUnknownMemberType]
    return AttentionRuntime(
        seed=seed,
        device=selected_device,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
    )
