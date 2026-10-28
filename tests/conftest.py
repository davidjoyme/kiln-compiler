# File: tests/conftest.py
import os
import sys

import pytest
import torch

# Make tests/models importable as `models`.
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield