# File: tests/lit/lit.cfg.py
# Configuration for the FileCheck tests. Run with:  lit -v tests/lit
import os
import shutil
import sys

import lit.formats

config.name = "Kiln"
config.test_format = lit.formats.ShTest()
config.suffixes = [".kir"]
config.test_source_root = os.path.dirname(__file__)

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config.environment["PYTHONPATH"] = os.pathsep.join(p for p in [root, os.environ.get("PYTHONPATH")] if p)

config.substitutions.append(("%kiln-opt", f"{sys.executable} -m kiln.tools.opt"))
# LLVM's FileCheck if it's on PATH (e.g. build/bin), else the `filecheck` pip package.
filecheck = shutil.which("FileCheck") or shutil.which("filecheck")
if filecheck is None:
    raise RuntimeError("FileCheck not found: add build/bin to PATH or `pip install filecheck`")
config.substitutions.append(("FileCheck", filecheck))