# File: kiln/codegen/lowering.py
"""Lower the emitted MLIR module to the LLVM dialect.

Stage 1 compiles every kernel to plain scalar loops (convert-linalg-to-loops).
Tiling and vectorisation schedules are added in a later stage, between
emission and bufferization.
"""

from __future__ import annotations

from mlir.ir import Context, Module
from mlir.passmanager import PassManager

from kiln.codegen.emitter import ModuleCode

# Tensors -> buffers. Kernel and entry-point arguments are already memrefs, so
# only kernel-internal temporaries (if any) are allocated here.
BUFFERIZE = [
    "one-shot-bufferize",
    "buffer-deallocation-pipeline",
    "convert-bufferization-to-memref",
    "fold-memref-alias-ops",
    "func.func(canonicalize,cse,loop-invariant-code-motion)",
]

TO_LLVM = [
    "convert-linalg-to-loops",
    "expand-strided-metadata",
    "lower-affine",
    "convert-scf-to-cf",
    "finalize-memref-to-llvm",
    "convert-math-to-llvm",
    "convert-arith-to-llvm",
    "convert-index-to-llvm",
    "convert-cf-to-llvm",
    "convert-func-to-llvm",
    "convert-ub-to-llvm",
    "reconcile-unrealized-casts",
]


def run_pipeline(module: Module, passes):
    PassManager.parse(f"builtin.module({','.join(passes)})", context=module.context).run(module.operation)


def lower(code: ModuleCode, dump=None):
    """Returns (llvm-dialect module, context, report)."""
    ctx = Context()
    module = Module.parse(code.text, context=ctx)
    run_pipeline(module, ["canonicalize", "cse"])
    run_pipeline(module, BUFFERIZE)
    text = str(module)
    report = {"heap_allocs": text.count("memref.alloc(") + text.count("memref.alloc()")}
    if dump is not None:
        dump("after bufferization", text)
    run_pipeline(module, TO_LLVM)
    return module, ctx, report