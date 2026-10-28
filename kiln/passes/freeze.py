# File: kiln/passes/freeze.py
"""Weight freezing and constant folding.

For inference, parameters never change between calls, so they can be treated
as compile-time constants. That alone does nothing; its value is what it
enables: any computation that depends only on parameters can be evaluated
once, at compile time, instead of on every call.

The motivating example is ``nn.Linear``. PyTorch stores the weight as
[out, in] and ATen lowers ``linear`` to ``addmm(bias, x, weight.t())``. The
matmul kernel wants the right-hand operand as [K, N] with N contiguous, so it
can load a vector of N outputs per k step. Without freezing, every call has to
read the weight transposed (strided, scalar loads) or copy it. With freezing,
the transpose is folded into a new constant stored in exactly the layout the
kernel wants: a simple form of weight prepacking.
"""

from __future__ import annotations

from kiln.ir import ops
from kiln.ir.core import Graph
from kiln.passes.manager import Pass


class FreezeParameters(Pass):
    """Turn parameter/buffer graph arguments into constants."""

    name = "freeze"

    def run(self, g: Graph):
        frozen = 0
        keep_args, keep_attrs = [], []
        for arg, attrs in zip(g.args, g.arg_attrs):
            if "param" in attrs:
                const = g.add_node("const", [], {"name": attrs["param"]}, arg.type, index=frozen)
                g.replace_all_uses(arg, const)
                frozen += 1
            else:
                keep_args.append(arg)
                keep_attrs.append(attrs)
        g.args, g.arg_attrs = keep_args, keep_attrs
        return {"frozen": frozen} if frozen else None


class ConstantFold(Pass):
    """Evaluate nodes whose operands are all constants.

    Folding is refused when it would grow the data (e.g. ``expand`` of a
    scalar to a large tensor): materialising a broadcast as a constant trades a
    free view for real memory and bandwidth, which is a pessimisation.
    """

    name = "const-fold"

    def run(self, g: Graph):
        folded, folded_bytes = 0, 0
        for n in list(g.nodes):
            if n.op == "const" or not n.inputs or not all(g.is_const(v) for v in n.inputs):
                continue
            largest_input = max(v.type.numel for v in n.inputs)
            if n.out.type.numel > max(largest_input, 1):
                continue
            data = ops.evaluate(n.op, [g.const_data(v) for v in n.inputs], n.attrs)
            index = g.nodes.index(n)
            if n.out.type.rank == 0:
                new = g.scalar(float(data), index=index)
            else:
                base = next((v.node.attrs["name"] for v in n.inputs if "name" in v.node.attrs), "folded")
                new = g.constant(data, f"{base}.{n.op}", index=index)
                folded_bytes += data.nbytes
            g.replace_all_uses(n.out, new)
            g.nodes.remove(n)
            folded += 1
        g.remove_dead()
        return {"folded": folded, "bytes": folded_bytes} if folded else None