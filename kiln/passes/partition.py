# File: kiln/passes/partition.py
"""Kernel partitioning: decide which ops are compiled together into one kernel.

Every node that computes something (elementwise, reduction, matmul) belongs
to exactly one kernel, and each kernel becomes one function in the generated
MLIR. View ops (reshape, transpose, slice, expand) and constants are never in
a kernel; buffer assignment turns views into strided views of buffers.

This pass is the simplest possible partition: one op per kernel, in program
order. It is the baseline the fusion pass (next stage) improves on, and stays
available as the ``fusion=False`` setting for ablations.
"""

from __future__ import annotations

from kiln.ir import ops
from kiln.ir.core import Graph, Kernel
from kiln.passes.manager import Pass

COMPUTE_KINDS = ("elementwise", "reduction", "matmul")


class OneOpPerKernel(Pass):
    name = "partition"

    def run(self, graph: Graph):
        graph.kernels = []
        for n in graph.nodes:
            kind = ops.kind(n.op)
            if kind not in COMPUTE_KINDS:
                continue
            k = Kernel(f"k{len(graph.kernels)}", kind, [n])
            n.attrs["kernel"] = k.name
            graph.kernels.append(k)
        return {"kernels": len(graph.kernels)}