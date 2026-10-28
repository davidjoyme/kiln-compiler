# File: kiln/tools/opt.py
"""kiln-opt: run Kiln passes on textual IR (the analogue of mlir-opt).

    kiln-opt input.kir --passes=freeze,const-fold,canonicalize
    kiln-opt input.kir --passes=fusion,buffer-assign,memory-plan --emit=fusion-scripts
    kiln-opt input.kir --pipeline=default --emit=mlir
    kiln-opt input.kir --passes=freeze,const-fold --print-ir-after-all

Reads from a file or stdin ("-") and writes the resulting IR to stdout. The
FileCheck tests in tests/lit are written against this tool.
"""

from __future__ import annotations

import argparse
import sys


def make_pass(name):
    from kiln.passes.buffers import BufferAssignment, MemoryPlan
    from kiln.passes.freeze import ConstantFold, FreezeParameters
    from kiln.passes.fusion import Fusion
    from kiln.passes.partition import OneOpPerKernel
    from kiln.passes.simplify import CSE, DCE, Canonicalize

    table = {
        "canonicalize": Canonicalize,
        "cse": CSE,
        "dce": DCE,
        "freeze": FreezeParameters,
        "const-fold": ConstantFold,
        "partition": OneOpPerKernel,
        "fusion": Fusion,
        "buffer-assign": BufferAssignment,
        "memory-plan": MemoryPlan,
    }
    if name not in table:
        raise SystemExit(f"kiln-opt: unknown pass '{name}' (known: {', '.join(table)})")
    return table[name]()


def main(argv=None):
    from kiln.api import Options, build_passes
    from kiln.ir.parser import ParseError, parse
    from kiln.passes.manager import PassManager

    p = argparse.ArgumentParser(prog="kiln-opt", description=__doc__.splitlines()[0])
    p.add_argument("input", nargs="?", default="-")
    p.add_argument("--passes", default="", help="comma-separated list of passes")
    p.add_argument("--pipeline", choices=["default", "none"], help="run the full compile pipeline")
    p.add_argument("--emit", choices=["kir", "mlir", "fusion-scripts"], default="kir",
                   help="fusion-scripts: the MLIR loop-fusion script for each kernel")
    p.add_argument("--print-ir-after-all", action="store_true")
    args = p.parse_args(argv)

    text = sys.stdin.read() if args.input == "-" else open(args.input).read()
    try:
        graph = parse(text)
    except ParseError as err:
        print(f"kiln-opt: {err}", file=sys.stderr)
        return 1

    if args.pipeline == "default":
        passes = build_passes(Options())
    elif args.pipeline == "none":
        passes = build_passes(Options.none())
    else:
        passes = [make_pass(n.strip()) for n in args.passes.split(",") if n.strip()]
    PassManager(passes, print_after_all=args.print_ir_after_all).run(graph)

    if args.emit != "kir" and "arena_bytes" not in graph.attrs:
        print("kiln-opt: --emit=mlir/fusion-scripts needs kernels and buffers: run the "
              "partition (or fusion), buffer-assign and memory-plan passes, or --pipeline=default",
              file=sys.stderr)
        return 1

    if args.emit == "mlir":
        from kiln.codegen.emitter import emit_module

        print(emit_module(graph).text)
    elif args.emit == "fusion-scripts":
        from kiln.codegen.emitter import emit_module
        from kiln.codegen.transforms import build_script

        for kc in emit_module(graph).kernels:
            script = build_script(kc)
            print(f"// ----- @{kc.kernel.name} ({kc.kernel.kind}): "
                  f"{'no loop fusion needed' if script is None else 'loop fusion'}")
            if script:
                print(script)
    else:
        print(graph)
    return 0


if __name__ == "__main__":
    sys.exit(main())