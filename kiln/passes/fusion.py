# File: kiln/passes/fusion.py
"""Operator fusion: partition the graph into kernels.

Every node that computes something (elementwise, reduction or matmul) is
assigned to exactly one kernel. Each kernel becomes one function, so values
that stay inside a kernel never touch memory: fusion removes memory traffic,
intermediate buffers and per-op dispatch overhead.

Three kinds of kernel are formed, greedily, in topological order:

  matmul       a contraction followed by an elementwise *epilogue* over its
               output (bias add, activation, residual add, attention scaling
               and masking). The epilogue is applied to each output tile while
               it is still in registers / L1, instead of re-reading the whole
               matmul result from memory.

  reduction    a "row kernel": reductions over the innermost dimension plus
               the elementwise ops around them that share the same rows
               (LayerNorm, softmax). Codegen computes one row at a time with
               all intermediates in registers or a small stack buffer.

  elementwise  chains of elementwise ops, compiled to a single loop nest.

Legality. A node may join a producer's kernel only if:
  * it is compatible with the kernel's kind (see ``_can_join``);
  * the producer reaches it through a direct edge. View ops (reshape,
    transpose, slice, expand) are never fused; they are resolved later as
    strided views of materialised buffers by buffer assignment;
  * joining does not create a cycle between kernels. If the node also
    depends on kernel H, and H (transitively) depends on the target kernel K,
    then putting the node in K would make K both a producer and a consumer of
    H. This is the standard fusion legality check (XLA, TVM).

Matmul epilogues additionally require the matmul result to have no other
users: otherwise the full matmul output would have to be materialised anyway,
and tile-level fusion would compute it twice.
"""

from __future__ import annotations

from kiln.ir import ops
from kiln.ir.core import Graph, Kernel, Node
from kiln.passes.manager import Pass

_PRIORITY = {"matmul": 0, "reduction": 1, "elementwise": 2}


def is_row_reduction(n: Node) -> bool:
    """A keepdim reduction over (only) the innermost dimension."""
    rank = n.inputs[0].type.rank
    return rank >= 1 and list(n.attrs["dims"]) == [rank - 1]


def compute_kind(n: Node):
    k = ops.kind(n.op)
    return k if k in ("elementwise", "reduction", "matmul") else None


def resolve_view(v):
    """Follow a chain of view ops back to the value that owns the data."""
    while v.node is not None and ops.kind(v.node.op) == "view":
        v = v.node.inputs[0]
    return v


class _Group:
    def __init__(self, kind, first: Node):
        self.kind = kind
        self.nodes = [first]
        self.anchor = first if kind == "matmul" else None
        self.rows = None          # row-kernel iteration space (all but the last dim)
        self.inner = None
        self.fusible = True
        if kind == "reduction":
            if is_row_reduction(first):
                shape = first.inputs[0].shape
                self.rows, self.inner = shape[:-1], shape[-1]
            else:
                self.fusible = False


class Fusion(Pass):
    name = "fusion"

    # -- helpers -----------------------------------------------------------------

    def _reaches(self, src, dst, succ):
        """Is there a path src -> ... -> dst in the kernel DAG?"""
        stack, seen = [src], {src}
        while stack:
            g = stack.pop()
            if g is dst:
                return True
            for s in succ.get(g, ()):
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        return False

    def _row_compatible(self, n: Node, rows, inner):
        shape = n.out.shape
        return len(shape) >= 1 and tuple(shape[:-1]) == tuple(rows) and shape[-1] in (inner, 1)

    def _can_join(self, n: Node, g: _Group, group_of, users):
        if not g.fusible:
            return False
        kind = compute_kind(n)
        in_group = [v.node for v in n.inputs if v.node is not None and group_of.get(v.node) is g]

        if g.kind == "matmul":
            if kind != "elementwise" or n.out.shape != g.anchor.out.shape:
                return False
            # The matmul result must feed only its epilogue.
            if g.anchor in in_group and len(users.get(g.anchor.out, [])) != 1:
                return False
            return True

        if kind == "elementwise":
            if g.kind == "elementwise":
                return True
            return self._row_compatible(n, g.rows, g.inner)

        if kind == "reduction":
            if not is_row_reduction(n):
                return False
            shape = n.inputs[0].shape
            rows, inner = shape[:-1], shape[-1]
            if g.kind == "reduction":
                return (tuple(rows), inner) == (tuple(g.rows), g.inner)
            # Joining an elementwise group turns it into a row kernel, so every
            # node already in it must live in the same row space.
            return all(self._row_compatible(m, rows, inner) for m in g.nodes)
        return False

    # -- main ----------------------------------------------------------------------

    def run(self, graph: Graph):
        users = graph.users()
        group_of: dict[Node, _Group] = {}
        groups: list[_Group] = []
        succ: dict[_Group, set] = {}

        def producer_groups(n):
            out = []
            for v in n.inputs:
                src = resolve_view(v).node
                if src is not None and src in group_of and group_of[src] not in out:
                    out.append(group_of[src])
            return out

        for n in graph.nodes:
            kind = compute_kind(n)
            if kind is None:
                continue

            target = None
            if kind != "matmul":
                direct = []
                for v in n.inputs:
                    if v.node is not None and v.node in group_of and group_of[v.node] not in direct:
                        direct.append(group_of[v.node])
                direct.sort(key=lambda g: _PRIORITY[g.kind])
                all_producers = producer_groups(n)
                for g in direct:
                    if not self._can_join(n, g, group_of, users):
                        continue
                    if any(h is not g and self._reaches(g, h, succ) for h in all_producers):
                        continue  # would create a cycle between kernels
                    target = g
                    break

            if target is None:
                target = _Group(kind, n)
                groups.append(target)
            else:
                target.nodes.append(n)
                if kind == "reduction" and target.kind == "elementwise":
                    shape = n.inputs[0].shape
                    target.kind, target.rows, target.inner = "reduction", shape[:-1], shape[-1]
            group_of[n] = target
            for h in producer_groups(n):
                if h is not target:
                    succ.setdefault(h, set()).add(target)

        # Order kernels topologically (stable w.r.t. program order).
        order, state = [], {}
        first_index = {g: graph.nodes.index(g.nodes[0]) for g in groups}
        preds = {g: set() for g in groups}
        for h, ss in succ.items():
            for s in ss:
                preds[s].add(h)

        def visit(g):
            if state.get(g) == "done":
                return
            if state.get(g) == "active":
                raise AssertionError("fusion produced a cycle between kernels")
            state[g] = "active"
            for p in sorted(preds[g], key=first_index.get):
                visit(p)
            state[g] = "done"
            order.append(g)

        for g in sorted(groups, key=first_index.get):
            visit(g)

        graph.kernels = []
        for i, g in enumerate(order):
            k = Kernel(f"k{i}", g.kind, list(g.nodes))
            for n in g.nodes:
                n.attrs["kernel"] = k.name
            graph.kernels.append(k)

        fused_ops = sum(len(k.nodes) for k in graph.kernels)
        return {"kernels": len(graph.kernels), "ops": fused_ops}