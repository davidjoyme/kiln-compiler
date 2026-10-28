# File: kiln/ir/ops.py
"""Op definitions for the Kiln IR: kind, type inference and reference semantics.

The reference implementations (NumPy) serve two purposes: the constant
folder evaluates them at compile time, and ``interpret`` runs a whole graph,
which the tests use to check each pass in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf as _erf, prod
from typing import Callable

import numpy as np

from kiln.ir.core import Graph, TensorType


@dataclass(frozen=True)
class OpDef:
    kind: str                      # elementwise | reduction | matmul | view | const
    infer: Callable                # (input shapes, attrs) -> output shape
    ref: Callable                  # (numpy inputs, attrs) -> numpy output
    arity: int | None = None


OPS: dict[str, OpDef] = {}


def broadcast_shapes(*shapes):
    return tuple(np.broadcast_shapes(*shapes))


def _register(name, kind, infer, ref, arity=None):
    OPS[name] = OpDef(kind, infer, ref, arity)


# -- elementwise ----------------------------------------------------------------

def _ew_infer(shapes, attrs):
    return broadcast_shapes(*shapes)


_erf_vec = np.vectorize(_erf, otypes=[np.float64])


def _gelu(x, attrs):
    x = x.astype(np.float64)
    if attrs.get("approximate", "none") == "tanh":
        return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))
    return 0.5 * x * (1 + _erf_vec(x / np.sqrt(2.0)))


UNARY = {
    "neg": lambda x, a: -x,
    "exp": lambda x, a: np.exp(x),
    "tanh": lambda x, a: np.tanh(x),
    "sigmoid": lambda x, a: 1 / (1 + np.exp(-x)),
    "relu": lambda x, a: np.maximum(x, 0),
    "gelu": _gelu,
    "rsqrt": lambda x, a: 1 / np.sqrt(x),
    "sqrt": lambda x, a: np.sqrt(x),
    "erf": lambda x, a: _erf_vec(x),
    "copy": lambda x, a: x.copy(),
}
BINARY = {
    "add": lambda x, y, a: x + y,
    "sub": lambda x, y, a: x - y,
    "mul": lambda x, y, a: x * y,
    "div": lambda x, y, a: x / y,
    "maximum": lambda x, y, a: np.maximum(x, y),
    "minimum": lambda x, y, a: np.minimum(x, y),
}
for _name, _fn in UNARY.items():
    _register(_name, "elementwise", _ew_infer, (lambda f: lambda xs, a: f(xs[0], a))(_fn), 1)
for _name, _fn in BINARY.items():
    _register(_name, "elementwise", _ew_infer, (lambda f: lambda xs, a: f(xs[0], xs[1], a))(_fn), 2)


# -- reductions (always keepdim) ------------------------------------------------

def _red_infer(shapes, attrs):
    (shape,) = shapes
    dims = {d % len(shape) for d in attrs["dims"]}
    return tuple(1 if i in dims else s for i, s in enumerate(shape))


def _red_ref(fn):
    return lambda xs, a: fn(xs[0], axis=tuple(a["dims"]), keepdims=True)


_register("reduce_sum", "reduction", _red_infer, _red_ref(np.sum), 1)
_register("reduce_max", "reduction", _red_infer, _red_ref(np.max), 1)

REDUCTION_COMBINER = {"reduce_sum": "add", "reduce_max": "maximum"}
REDUCTION_INIT = {"reduce_sum": 0.0, "reduce_max": float("-inf")}


# -- matmul -----------------------------------------------------------------------

def _mm_infer(shapes, attrs):
    a, b = shapes
    if len(a) != len(b) or len(a) not in (2, 3) or a[-1] != b[-2] or a[:-2] != b[:-2]:
        raise ValueError(f"matmul: incompatible shapes {a} x {b}")
    return (*a[:-1], b[-1])


_register("matmul", "matmul", _mm_infer, lambda xs, a: np.matmul(xs[0], xs[1]), 2)


# -- views (no data movement once buffers are assigned) ----------------------------

def _transpose_infer(shapes, attrs):
    return tuple(shapes[0][p] for p in attrs["perm"])


def _reshape_infer(shapes, attrs):
    shape = tuple(attrs["shape"])
    if prod(shape) != prod(shapes[0]):
        raise ValueError(f"reshape: {shapes[0]} -> {shape} changes element count")
    return shape


def _slice_infer(shapes, attrs):
    shape = list(shapes[0])
    d = attrs["dim"] % len(shape)
    start, end, step = attrs["start"], min(attrs["end"], shape[d]), attrs.get("step", 1)
    shape[d] = max(0, (end - start + step - 1) // step)
    return tuple(shape)


def _slice_ref(xs, a):
    idx = [slice(None)] * xs[0].ndim
    idx[a["dim"]] = slice(a["start"], a["end"], a.get("step", 1))
    return xs[0][tuple(idx)]


def _expand_infer(shapes, attrs):
    return broadcast_shapes(shapes[0], tuple(attrs["shape"]))


_register("transpose", "view", _transpose_infer, lambda xs, a: np.transpose(xs[0], a["perm"]), 1)
_register("reshape", "view", _reshape_infer, lambda xs, a: xs[0].reshape(a["shape"]), 1)
_register("slice", "view", _slice_infer, _slice_ref, 1)
_register("expand", "view", _expand_infer, lambda xs, a: np.broadcast_to(xs[0], a["shape"]), 1)

# -- constants --------------------------------------------------------------------

_register("const", "const", lambda shapes, attrs: None, lambda xs, a: None, 0)


def kind(op: str) -> str:
    return OPS[op].kind


def infer_type(op: str, input_types, attrs) -> TensorType:
    if op not in OPS:
        raise ValueError(f"unknown op '{op}'")
    opdef = OPS[op]
    if opdef.arity is not None and len(input_types) != opdef.arity:
        raise ValueError(f"{op} expects {opdef.arity} operands, got {len(input_types)}")
    shape = opdef.infer([t.shape for t in input_types], attrs)
    if shape is None:
        raise ValueError(f"{op}: result type must be given explicitly")
    return TensorType(tuple(int(s) for s in shape))


def evaluate(op: str, inputs, attrs) -> np.ndarray:
    out = OPS[op].ref([np.asarray(x) for x in inputs], attrs)
    return np.array(out, dtype=np.float32, order="C")  # keeps 0-d arrays 0-d


def interpret(graph: Graph, *inputs) -> list:
    """Run a graph with the NumPy reference semantics (for testing)."""
    env = {}
    user_args = [a for a, attrs in zip(graph.args, graph.arg_attrs) if "param" not in attrs]
    if len(inputs) != len(user_args):
        raise ValueError(f"expected {len(user_args)} inputs, got {len(inputs)}")
    it = iter(inputs)
    for arg, attrs in zip(graph.args, graph.arg_attrs):
        env[arg] = graph.constants[attrs["param"]] if "param" in attrs else np.asarray(next(it), np.float32)
    for n in graph.nodes:
        if n.op == "const":
            env[n.out] = graph.const_data(n.out)
        else:
            env[n.out] = evaluate(n.op, [env[v] for v in n.inputs], n.attrs)
    return [env[v] for v in graph.results]