"""torch.compile entry point: FX graph -> MLIR -> LLVM -> native code.

Usage:
    import kiln
    compiled = torch.compile(model, backend=kiln.backend)

Environment variables:
    KILN_DUMP_IR=1   print the MLIR after each pipeline stage
    KILN_STRICT=1    raise on unsupported graphs instead of falling back to eager
"""

import os
import warnings

from torch._dynamo.backends.common import aot_autograd

from kiln.frontend.fx_to_mlir import fx_to_mlir
from kiln.runtime.executor import KilnExecutable

# Simple counters so tests can check that graphs really went through Kiln.
stats = {"compiled": 0, "fallback": 0}


def _flag(name):
    return os.environ.get(name, "0") not in ("", "0", "false", "False")


def kiln_compiler(gm, example_inputs):
    dump = _flag("KILN_DUMP_IR")
    try:
        program = fx_to_mlir(gm)
        if dump:
            print("// ----- Kiln input IR -----")
            print(program.mlir)
        executable = KilnExecutable(program, dump_ir=dump)
    except NotImplementedError as err:
        if _flag("KILN_STRICT"):
            raise
        warnings.warn(f"Kiln: falling back to eager PyTorch ({err})")
        stats["fallback"] += 1
        return gm.forward

    stats["compiled"] += 1
    return executable


# aot_autograd traces the model down to ATen ops and lifts parameters into
# graph inputs, so kiln_compiler only ever sees a small, stable op set.
backend = aot_autograd(fw_compiler=kiln_compiler)