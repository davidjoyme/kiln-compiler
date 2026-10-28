# File: kiln/passes/simplify.py
"""Canonicalization, common subexpression elimination and dead code elimination.

These are the classic cleanup passes. They run after the frontend (to remove
no-op views and identity arithmetic the ATen decompositions leave behind) and
again after freezing, when constant folding exposes more of them.
"""

from __future__ import annotations

from kiln.ir.core import Graph
from kiln.passes.manager import Pass


def _identity_perm(perm):
    return list(perm) == list(range(len(perm)))


class Canonicalize(Pass):
    """Local rewrites, iterated to a fixed point.

    * reshape/expand to the same shape, identity transpose, full slice -> operand
    * reshape(reshape(x)) -> reshape(x)
    * transpose(transpose(x)) -> transpose(x) with the composed permutation
    * x * 1, x / 1, x + 0, x - 0 (scalar constants) -> x
    """

    name = "canonicalize"

    def _simplify(self, g: Graph, n):
        if n.op in ("reshape", "expand") and n.out.shape == n.inputs[0].shape:
            return n.inputs[0]
        if n.op == "transpose" and _identity_perm(n.attrs["perm"]):
            return n.inputs[0]
        if n.op == "slice" and n.out.shape == n.inputs[0].shape:
            return n.inputs[0]

        src = n.inputs[0].node if n.inputs else None
        if n.op == "reshape" and src is not None and src.op == "reshape":
            return g.add_node("reshape", [src.inputs[0]], {"shape": list(n.out.shape)},
                              index=g.nodes.index(n))
        if n.op == "transpose" and src is not None and src.op == "transpose":
            inner, outer = src.attrs["perm"], n.attrs["perm"]
            perm = [inner[p] for p in outer]
            return g.add_node("transpose", [src.inputs[0]], {"perm": perm}, index=g.nodes.index(n))

        if n.op in ("mul", "div", "add", "sub") and len(n.inputs) == 2:
            a, b = n.inputs
            neutral = 1.0 if n.op in ("mul", "div") else 0.0
            if g.is_scalar_const(b) and g.const_data(b) == neutral and a.shape == n.out.shape:
                return a
            if n.op in ("mul", "add") and g.is_scalar_const(a) and g.const_data(a) == neutral \
                    and b.shape == n.out.shape:
                return b
        return None

    def run(self, g: Graph):
        rewrites = 0
        changed = True
        while changed:
            changed = False
            for n in list(g.nodes):
                if n not in g.nodes:
                    continue
                new = self._simplify(g, n)
                if new is not None and new is not n.out:
                    g.replace_all_uses(n.out, new)
                    g.nodes.remove(n)
                    rewrites += 1
                    changed = True
        return {"rewrites": rewrites} if rewrites else None


class CSE(Pass):
    """Common subexpression elimination over (op, operands, attributes)."""

    name = "cse"

    def run(self, g: Graph):
        seen, removed = {}, 0
        for n in list(g.nodes):
            key = (n.op, tuple(id(v) for v in n.inputs), repr(sorted(n.attrs.items())), n.out.type)
            if key in seen:
                g.replace_all_uses(n.out, seen[key])
                g.nodes.remove(n)
                removed += 1
            else:
                seen[key] = n.out
        return {"removed": removed} if removed else None


class DCE(Pass):
    name = "dce"

    def run(self, g: Graph):
        removed = g.remove_dead()
        return {"removed": removed} if removed else None