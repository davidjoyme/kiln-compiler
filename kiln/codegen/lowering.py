# File: kiln/codegen/lowering.py
"""Apply loop fusion to each kernel, then lower the module to the LLVM dialect.

Each kernel's transform script (kiln/codegen/transforms.py) is applied on its
own. If MLIR rejects one, that kernel is restored from a snapshot and compiled
without loop fusion, with a warning: a fusion bug costs performance, never
correctness or a failed compile.

Kernels are still compiled to plain scalar loops (convert-linalg-to-loops);
tiling for cache and vectorisation come in a later stage.
"""

from __future__ import annotations

import warnings

from mlir.ir import Context, MLIRError, Module
from mlir.passmanager import PassManager

from kiln.codegen.emitter import ModuleCode
from kiln.codegen.transforms import build_script

# Tensors -> buffers. Kernel and entry-point arguments are already memrefs, so
# only kernel-internal temporaries are allocated here. After loop fusion they
# are one row each: hoist them out of the row loop (one buffer reused by every
# iteration) and move them to the stack, so a call does no heap allocation.
BUFFERIZE = [
    "one-shot-bufferize",
    "func.func(buffer-loop-hoisting,promote-buffers-to-stack{max-alloc-size-in-bytes=65536})",
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


def apply_script(module: Module, script: str):
    """Run one transform script against ``module`` (modifies it in place)."""
    ctx = module.context
    sched = Module.parse(script, context=ctx)
    module.operation.attributes["transform.with_named_sequence"] = \
        sched.operation.attributes["transform.with_named_sequence"]
    for op in sched.body.operations:
        module.body.append(op.clone())
    try:
        run_pipeline(module, ["transform-interpreter"])
    finally:
        for op in list(module.body.operations):
            if op.operation.name == "transform.named_sequence":
                op.operation.erase()
        del module.operation.attributes["transform.with_named_sequence"]


def apply_loop_fusion(code: ModuleCode, ctx: Context, report: dict, dump=None) -> Module:
    module = Module.parse(code.text, context=ctx)
    for kc in code.kernels:
        script = build_script(kc)
        if script is None:
            continue
        if dump is not None:
            dump(f"loop-fusion script for {kc.kernel.name}", script)
        snapshot = str(module)
        try:
            apply_script(module, script)
            report["loop_fused"].append(kc.kernel.name)
        except MLIRError as err:
            module = Module.parse(snapshot, context=ctx)
            first = str(err).splitlines()[0] if str(err) else "transform failed"
            warnings.warn(f"Kiln: loop fusion for {kc.kernel.name} failed ({first}); "
                          f"compiling it unfused")
            report["fallback"].append(kc.kernel.name)
    return module


def lower(code: ModuleCode, dump=None, loop_fusion=True):
    """Returns (llvm-dialect module, context, report)."""
    ctx = Context()
    report = {"loop_fused": [], "fallback": []}
    if loop_fusion:
        module = apply_loop_fusion(code, ctx, report, dump)
    else:
        module = Module.parse(code.text, context=ctx)
    run_pipeline(module, ["canonicalize", "cse"])
    if dump is not None:
        dump("after loop fusion", str(module))
    run_pipeline(module, BUFFERIZE)
    text = str(module)
    report["heap_allocs"] = text.count("memref.alloc(") + text.count("memref.alloc()")
    report["stack_allocs"] = text.count("memref.alloca(")
    if dump is not None:
        dump("after bufferization", text)
    run_pipeline(module, TO_LLVM)
    return module, ctx, report