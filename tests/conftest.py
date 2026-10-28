# File: tests/conftest.py
import os
import sys

import pytest
import torch

# Make tests/models importable as `models`.
sys.path.insert(0, os.path.dirname(__file__))

# GPU-backend tests run on a real CUDA GPU when there is one. Without one,
# Triton's interpreter runs the same generated kernels on the CPU. It must be
# switched on before anything imports triton.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield