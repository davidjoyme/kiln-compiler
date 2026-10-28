# File: kiln/__init__.py
"""Kiln: an MLIR-based inference compiler for PyTorch models.

    import kiln
    compiled = kiln.compile(model, (example_input,))   # export, freeze, compile
    y = compiled(x)

    compiled = torch.compile(model, backend=kiln.backend, dynamic=False)
"""

from kiln.api import CompiledModule, Options, compile, compile_graph
from kiln.backend import backend, stats

__all__ = ["compile", "compile_graph", "Options", "CompiledModule", "backend", "stats"]