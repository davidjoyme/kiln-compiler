# File: kiln/backend.py
"""torch.compile backend.

    compiled = torch.compile(model, backend=kiln.backend, dynamic=False)

This path compiles whatever graphs TorchDynamo captures, with parameters as
runtime inputs (Dynamo does not tell a backend which inputs are weights), so
freezing is off and weights are read in their original layout. For inference
deployments prefer ``kiln.compile``, which exports the model and freezes it.

Environment variables:
    KILN_STRICT=1   raise on unsupported graphs instead of falling back to eager
"""

from __future__ import annotations

import os
import warnings

import torch
from torch._dynamo.backends.common import aot_autograd

stats = {"compiled": 0, "fallback": 0}


def _strict():
    return os.environ.get("KILN_STRICT", "0") not in ("", "0", "false", "False")


def kiln_compiler(gm: torch.fx.GraphModule, example_inputs):
    from kiln.api import Options, compile_graph
    from kiln.frontend.fx_import import Unsupported, import_fx

    try:
        if any(not isinstance(t, torch.Tensor) for t in example_inputs):
            raise Unsupported("non-tensor graph inputs (symbolic shapes?)")
        graph = import_fx(gm)
        exe, _ = compile_graph(graph, Options(freeze=False))
    except NotImplementedError as err:
        if _strict():
            raise
        warnings.warn(f"Kiln: falling back to eager PyTorch ({err})")
        stats["fallback"] += 1
        return gm.forward

    stats["compiled"] += 1

    def run(*args):
        return exe(*args)

    return run


# aot_autograd traces down to ATen ops and lifts parameters into graph inputs.
backend = aot_autograd(fw_compiler=kiln_compiler)