# File: kiln/codegen/lowering.py
"""Apply each kernel's transform script, then lower the module to the LLVM dialect.

Each kernel's script (kiln/codegen/transforms.py: loop fusion, and with a
schedule also tiling and vectorisation) is applied on its own. If MLIR
rejects one, that kernel is restored from a snapshot and compiled without
it, as plain loops, with a warning: a scheduling bug costs performance,
never correctness or a failed compile.
"""

from __future__ import annotations

import warnings

from mlir.dialects import linalg
from mlir.ir import Context, InsertionPoint, Location, MLIRError, Module, WalkResult
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
    "convert-linalg-to-loops",       # anything left unscheduled: scalar loops
    "convert-vector-to-scf",         # vector transfers that need loops (e.g. strided)
    "expand-strided-metadata",
    "lower-affine",
    "convert-scf-to-cf",
    "convert-vector-to-llvm",        # vector ops -> LLVM vector IR (AVX / NEON registers)
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


def apply_scripts(code: ModuleCode, ctx: Context, report: dict, dump=None) -> Module:
    module = Module.parse(code.text, context=ctx)
    for kc in code.kernels:
        script = build_script(kc)
        if script is None:
            continue
        if dump is not None:
            dump(f"transform script for {kc.kernel.name}", script)
        snapshot = str(module)
        try:
            apply_script(module, script)
            report["transformed"].append(kc.kernel.name)
        except MLIRError as err:
            module = Module.parse(snapshot, context=ctx)
            first = str(err).splitlines()[0] if str(err) else "transform failed"
            warnings.warn(f"Kiln: transform script for {kc.kernel.name} failed ({first}); "
                          f"compiling it as plain loops")
            report["fallback"].append(kc.kernel.name)
    return module


def copies_to_loops(module: Module) -> int:
    """Rewrite every memref.copy left after bufferization into a linalg.copy.

    Bufferization inserts memref.copy when a kernel's result is a copy of
    one of its inputs, for example when canonicalization folds maximum(x, x)
    to x. For a strided source, MLIR lowers memref.copy to a call into its
    C runner-utils library (memrefCopy), which Kiln doesn't link. linalg.copy
    lowers to plain loops like every other kernel. Returns the count.
    """
    copies = []

    def visit(op):
        if op.name == "memref.copy":
            copies.append(op)
        return WalkResult.ADVANCE

    module.operation.walk(visit)
    with module.context, Location.unknown():
        for op in copies:
            with InsertionPoint(op):
                linalg.copy(op.operands[0], outs=[op.operands[1]])
            op.erase()
    return len(copies)


def lower(code: ModuleCode, dump=None, transform=True):
    """Returns (llvm-dialect module, context, report)."""
    ctx = Context()
    report = {"transformed": [], "fallback": []}
    if transform:
        module = apply_scripts(code, ctx, report, dump)
    else:
        module = Module.parse(code.text, context=ctx)
    run_pipeline(module, ["canonicalize", "cse"])
    if dump is not None:
        dump("after transform scripts", str(module))
    run_pipeline(module, BUFFERIZE)
    report["copies_to_loops"] = copies_to_loops(module)
    text = str(module)
    report["heap_allocs"] = text.count("memref.alloc(") + text.count("memref.alloc()")
    report["stack_allocs"] = text.count("memref.alloca(")
    # Vector ops in the kernels after bufferization (0 when nothing vectorised).
    report["vector_ops"] = sum(text.count(f"vector.{op} ") for op in
                               ("fma", "transfer_read", "transfer_write", "reduction"))
    if dump is not None:
        dump("after bufferization", text)
    run_pipeline(module, TO_LLVM)
    return module, ctx, report