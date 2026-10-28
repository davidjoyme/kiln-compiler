# File: kiln/ir/core.py
"""Kiln graph IR: a small SSA dataflow graph over statically shaped f32 tensors.

This is Kiln's high-level IR. Graph-level optimisations (freezing, constant
folding, fusion, buffer assignment, memory planning, scheduling) all run here,
and the result is lowered to MLIR (linalg/vector) for loop-level code
generation. It plays the same role as XLA's HLO or Inductor's FX/scheduler IR.

Design choices:
  * Every node produces exactly one value. Multi-result ATen ops
    (native_layer_norm, split) are decomposed by the frontend.
  * Values are immutable and have no layout. Layout (strides, aliasing) is
    decided late, by the buffer-assignment pass, so that view ops can be
    resolved without copies wherever possible.
  * Passes annotate the IR with attributes (``kernel``, ``offset`` ...) rather
    than switching to a different data structure. That keeps one printer and
    one parser, so every stage can be dumped and FileCheck-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

import numpy as np


@dataclass(frozen=True)
class TensorType:
    shape: tuple
    dtype: str = "f32"

    @property
    def rank(self):
        return len(self.shape)

    @property
    def numel(self):
        return prod(self.shape)

    @property
    def nbytes(self):
        return self.numel * 4

    def __str__(self):
        return f"{self.dtype}[{', '.join(map(str, self.shape))}]"


class Value:
    """An SSA value: either a graph argument or the result of a node."""

    __slots__ = ("type", "node", "name")

    def __init__(self, type: TensorType, node: "Node | None" = None, name: str | None = None):
        self.type = type
        self.node = node
        self.name = name  # only used for graph arguments (and as a printing hint)

    @property
    def shape(self):
        return self.type.shape

    @property
    def is_arg(self):
        return self.node is None

    def __repr__(self):
        origin = self.node.op if self.node else "arg"
        return f"<Value {self.name or '?'}: {self.type} from {origin}>"


class Node:
    __slots__ = ("op", "inputs", "attrs", "out")

    def __init__(self, op: str, inputs: list, attrs: dict, out_type: TensorType):
        self.op = op
        self.inputs = list(inputs)
        self.attrs = dict(attrs)
        self.out = Value(out_type, self)

    def __repr__(self):
        return f"<Node {self.op} -> {self.out.type}>"


@dataclass(eq=False)
class Kernel:
    """A fusion group: a set of nodes that will be compiled into one function."""

    name: str
    kind: str                      # elementwise | reduction | matmul | copy
    nodes: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


class Graph:
    def __init__(self, name: str = "forward"):
        self.name = name
        self.args: list[Value] = []
        self.arg_attrs: list[dict] = []
        self.nodes: list[Node] = []
        self.results: list[Value] = []
        self.attrs: dict = {}
        self.kernels: list[Kernel] = []
        # Tensor data for constants and (unfrozen) parameters, keyed by name.
        self.constants: dict[str, np.ndarray] = {}

    # -- construction -------------------------------------------------------

    def add_arg(self, type: TensorType, name: str, **attrs) -> Value:
        v = Value(type, None, name)
        self.args.append(v)
        self.arg_attrs.append(dict(attrs))
        return v

    def add_node(self, op: str, inputs, attrs=None, out_type=None, index=None) -> Value:
        """Create a node. The output type is inferred unless given explicitly."""
        from kiln.ir.ops import infer_type

        attrs = attrs or {}
        if out_type is None:
            out_type = infer_type(op, [v.type for v in inputs], attrs)
        node = Node(op, inputs, attrs, out_type)
        if index is None:
            self.nodes.append(node)
        else:
            self.nodes.insert(index, node)
        return node.out

    def constant(self, array: np.ndarray, name: str, index=None) -> Value:
        """Add a tensor constant whose data lives in ``self.constants``."""
        array = np.ascontiguousarray(array, dtype=np.float32)
        base, i = name, 1
        while name in self.constants:
            name, i = f"{base}_{i}", i + 1
        self.constants[name] = array
        return self.add_node("const", [], {"name": name}, TensorType(tuple(array.shape)), index)

    def scalar(self, value: float, index=None) -> Value:
        return self.add_node("const", [], {"value": float(value)}, TensorType(()), index)

    # -- queries ------------------------------------------------------------

    def users(self) -> dict:
        """Map from Value to the list of nodes that use it (in program order)."""
        uses = {}
        for n in self.nodes:
            for v in n.inputs:
                uses.setdefault(v, []).append(n)
        return uses

    def const_data(self, v: Value) -> np.ndarray:
        n = v.node
        assert n is not None and n.op == "const"
        if "value" in n.attrs:
            return np.asarray(n.attrs["value"], dtype=np.float32)
        return self.constants[n.attrs["name"]]

    def is_const(self, v: Value) -> bool:
        return v.node is not None and v.node.op == "const"

    def is_scalar_const(self, v: Value) -> bool:
        return self.is_const(v) and v.type.rank == 0

    def kernel_of(self, node: Node) -> "Kernel | None":
        name = node.attrs.get("kernel")
        for k in self.kernels:
            if k.name == name:
                return k
        return None

    # -- mutation -----------------------------------------------------------

    def replace_all_uses(self, old: Value, new: Value):
        for n in self.nodes:
            n.inputs = [new if v is old else v for v in n.inputs]
        self.results = [new if v is old else v for v in self.results]

    def remove_dead(self) -> int:
        """Dead code elimination. Returns the number of nodes removed."""
        live = set(self.results)
        kept = []
        for n in reversed(self.nodes):
            if n.out in live:
                kept.append(n)
                live.update(n.inputs)
        removed = len(self.nodes) - len(kept)
        self.nodes = list(reversed(kept))
        used_names = {n.attrs["name"] for n in self.nodes if n.op == "const" and "name" in n.attrs}
        used_names |= {a.get("param") for a in self.arg_attrs}
        self.constants = {k: v for k, v in self.constants.items() if k in used_names}
        return removed

    def toposort(self):
        """Re-sort nodes topologically (stable with respect to current order)."""
        order, seen = [], set()
        index = {n: i for i, n in enumerate(self.nodes)}

        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for v in n.inputs:
                if v.node is not None and v.node in index:
                    visit(v.node)
            order.append(n)

        for n in self.nodes:
            visit(n)
        self.nodes = order

    def __str__(self):
        from kiln.ir.printer import print_graph

        return print_graph(self)


def verify(graph: Graph):
    """Check SSA dominance and that every node's type matches inference."""
    from kiln.ir.ops import OPS, infer_type

    defined = set(graph.args)
    for n in graph.nodes:
        if n.op not in OPS:
            raise ValueError(f"unknown op '{n.op}'")
        for v in n.inputs:
            if v not in defined:
                raise ValueError(f"{n.op}: operand used before definition: {v}")
        if n.op == "const":
            if "value" in n.attrs and n.out.type.rank != 0:
                raise ValueError("const with 'value' must be a scalar (f32[])")
            if "value" not in n.attrs and n.attrs.get("name") not in graph.constants:
                raise ValueError(f"const refers to missing data '{n.attrs.get('name')}'")
        else:
            expected = infer_type(n.op, [v.type for v in n.inputs], n.attrs)
            if expected != n.out.type:
                raise ValueError(f"{n.op}: type {n.out.type} does not match inferred {expected}")
        defined.add(n.out)
    for v in graph.results:
        if v not in defined:
            raise ValueError(f"result {v} is not defined in the graph")