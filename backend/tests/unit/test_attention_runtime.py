"""Exercise the neural runtime without loading PyTorch beside legacy native libraries."""

import os
import subprocess
import sys
from pathlib import Path


def run_runtime_script(script: str) -> subprocess.CompletedProcess[str]:
    """Run one required neural check in a clean interpreter with project imports."""
    backend_src = Path(__file__).parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(backend_src), environment.get("PYTHONPATH")) if value
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_cpu_runtime_trains_repeatedly_and_records_provenance() -> None:
    completed = run_runtime_script(
        """
import math
import platform
import random

import numpy as np
import torch
from torch import nn

from clash_sos.infrastructure.ml.attention_runtime import initialize_attention_runtime

runtime = initialize_attention_runtime(7, "cpu")
model = nn.Linear(4, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
inputs = torch.randn(8, 4)
targets = torch.arange(8, dtype=torch.float32).remainder(2).unsqueeze(1)
before = model.weight.detach().clone()

optimizer.zero_grad(set_to_none=True)
loss = nn.functional.binary_cross_entropy_with_logits(model(inputs), targets)
loss.backward()

assert runtime.seed == 7
assert runtime.device == "cpu"
assert runtime.python_version == platform.python_version()
assert runtime.torch_version == str(torch.__version__)
assert math.isfinite(loss.item())
assert all(
    parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for parameter in model.parameters()
)
optimizer.step()
assert not torch.equal(before, model.weight)

initialize_attention_runtime(19, "cpu")
first_python_value = random.random()
first_numpy_value = np.random.random()
first_model = nn.Linear(3, 2)
first_inputs = torch.randn(4, 3)

initialize_attention_runtime(19, "cpu")
second_python_value = random.random()
second_numpy_value = np.random.random()
second_model = nn.Linear(3, 2)
second_inputs = torch.randn(4, 3)

assert first_python_value == second_python_value
assert first_numpy_value == second_numpy_value
torch.testing.assert_close(first_model.weight, second_model.weight)
torch.testing.assert_close(first_model.bias, second_model.bias)
torch.testing.assert_close(first_inputs, second_inputs)
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_runtime_validates_device_and_seed_policy() -> None:
    completed = run_runtime_script(
        """
import torch

from clash_sos.infrastructure.ml.attention_runtime import (
    MAX_SEED,
    NeuralRuntimeUnavailable,
    initialize_attention_runtime,
)

torch.backends.mps.is_available = lambda: True
assert initialize_attention_runtime(29).device == "mps"

torch.backends.mps.is_available = lambda: False
assert initialize_attention_runtime(31).device == "cpu"

try:
    initialize_attention_runtime(37, "mps")
except NeuralRuntimeUnavailable as error:
    assert "MPS" in str(error) and "not available" in str(error)
else:
    raise AssertionError("unavailable MPS did not fail")

for invalid_seed in (-1, MAX_SEED + 1, True):
    try:
        initialize_attention_runtime(invalid_seed, "cpu")
    except ValueError as error:
        assert "seed" in str(error)
    else:
        raise AssertionError(f"invalid seed accepted: {invalid_seed!r}")

try:
    initialize_attention_runtime(41, "cuda")
except ValueError as error:
    assert "device" in str(error)
else:
    raise AssertionError("unknown device did not fail")
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_legacy_imports_work_when_torch_is_unavailable() -> None:
    completed = run_runtime_script(
        """
import builtins

real_import = builtins.__import__

def reject_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch blocked for import-isolation test", name="torch")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_torch

from clash_sos.infrastructure.ml.attention_runtime import (
    ML_INSTALL_COMMAND,
    NeuralRuntimeUnavailable,
    initialize_attention_runtime,
)

try:
    initialize_attention_runtime(43, "cpu")
except NeuralRuntimeUnavailable as error:
    assert ML_INSTALL_COMMAND in str(error)
else:
    raise AssertionError("missing PyTorch did not fail")

import clash_sos.application.model_train
import clash_sos.interfaces.cli.main
"""
    )

    assert completed.returncode == 0, completed.stderr
