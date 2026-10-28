# File: tests/test_lit.py
"""Run the FileCheck tests in tests/lit as part of pytest (needs `lit` and FileCheck)."""

import os
import shutil
import subprocess
import sys

import pytest

LIT_DIR = os.path.join(os.path.dirname(__file__), "lit")


@pytest.mark.skipif(shutil.which("lit") is None, reason="lit not installed (pip install lit filecheck)")
def test_filecheck_suite():
    result = subprocess.run([shutil.which("lit"), "-q", LIT_DIR], capture_output=True, text=True,
                            env={**os.environ, "PYTHON": sys.executable})
    assert result.returncode == 0, result.stdout + result.stderr