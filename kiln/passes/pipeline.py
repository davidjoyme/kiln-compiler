"""Lowering pipeline: tensor/linalg/arith  ->  LLVM dialect.

Each stage is a named list of passes so it is easy to insert new ones (e.g. a
fusion or tiling stage) and to dump the IR between stages when debugging.
"""

from mlir.ir import Context, Module
from mlir.passmanager import PassManager

# Optimisation passes go here as Kiln grows (fusion, tiling, vectorisation...).
OPTIMIZE = []

# Tensors -> buffers. Function-boundary bufferization turns tensor arguments
# into plain memrefs; the dealloc pipeline frees intermediate buffers.
BUFFERIZE = [
    "one-shot-bufferize{bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map}",
    "buffer-deallocation-pipeline",
    "convert-bufferization-to-memref",
]

# Structured ops -> plain loops -> LLVM.
TO_LLVM = [
    "convert-linalg-to-loops",
    "expand-strided-metadata",
    "lower-affine",
    "convert-scf-to-cf",
    "finalize-memref-to-llvm",
    "convert-arith-to-llvm",
    "convert-cf-to-llvm",
    "convert-func-to-llvm",
    "reconcile-unrealized-casts",
]

STAGES = [("optimize", OPTIMIZE), ("bufferize", BUFFERIZE), ("to-llvm", TO_LLVM)]


def run_passes(module: Module, passes):
    if not passes:
        return
    pm = PassManager.parse(f"builtin.module({','.join(passes)})", context=module.context)
    pm.run(module.operation)


def lower_to_llvm(mlir_text: str, ctx: Context, dump_stages: bool = False) -> Module:
    """Parse MLIR text and lower it all the way to the LLVM dialect."""
    module = Module.parse(mlir_text, context=ctx)
    for name, passes in STAGES:
        run_passes(module, passes)
        if dump_stages and passes:
            print(f"// ----- after stage: {name} -----")
            print(module)
    return module