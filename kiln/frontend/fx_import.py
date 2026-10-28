# File: kiln/frontend/fx_import.py
"""Import ATen-level torch.fx graphs into the Kiln IR.

Two entry points:
  * ``import_exported(ep)`` for ``torch.export`` programs (the inference path
    used by ``kiln.compile``). The graph signature tells us which placeholders
    are parameters/buffers, so they become graph arguments tagged ``param``,
    which the freeze pass can later turn into constants.
  * ``import_fx(gm)`` for graphs from ``torch.compile`` + ``aot_autograd``,
    where every placeholder is a runtime input.

Composite ATen ops are decomposed here into Kiln's small core op set: addmm
becomes matmul + add, layer_norm becomes reductions + elementwise ops, softmax
becomes max / exp / sum / div. Recombining those pieces efficiently is then
the fusion pass's job, which keeps every later pass simple.
"""

from __future__ import annotations

import operator
from numbers import Number

import numpy as np
import torch
import torch.fx

from kiln.ir.core import Graph, TensorType, Value, verify

aten = torch.ops.aten


class Unsupported(NotImplementedError):
    """Raised for graphs Kiln cannot compile (callers fall back to eager)."""


def _shape_of(node):
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise Unsupported(f"{node.name}: no tensor metadata")
    if val.dtype != torch.float32:
        raise Unsupported(f"{node.name}: only float32 is supported (got {val.dtype})")
    if any(not isinstance(d, int) for d in val.shape):
        raise Unsupported(f"{node.name}: dynamic shapes are not supported")
    return tuple(val.shape)


class _Importer:
    def __init__(self, name="forward"):
        self.g = Graph(name)
        self.env = {}
        self.scalars = {}

    # -- helpers --------------------------------------------------------------

    def v(self, arg):
        """Kiln value for an FX argument (node or Python number)."""
        if isinstance(arg, torch.fx.Node):
            val = self.env[arg]
            if not isinstance(val, Value):
                raise Unsupported(f"{arg.name}: expected a single tensor")
            return val
        if isinstance(arg, bool) or not isinstance(arg, Number):
            raise Unsupported(f"unsupported operand {arg!r}")
        key = float(arg)
        if key not in self.scalars:
            self.scalars[key] = self.g.scalar(key)
        return self.scalars[key]

    def op(self, name, *inputs, **attrs):
        return self.g.add_node(name, list(inputs), attrs)

    def reduce(self, kind, x, dims):
        dims = sorted(d % x.type.rank for d in dims)
        return self.op(kind, x, dims=dims)

    def mean(self, x, dims, keepdim=True):
        dims = sorted(d % x.type.rank for d in dims)
        count = int(np.prod([x.shape[d] for d in dims]))
        out = self.op("mul", self.reduce("reduce_sum", x, dims), self.v(1.0 / count))
        return out if keepdim else self.squeeze(out, dims)

    def squeeze(self, x, dims):
        shape = [s for i, s in enumerate(x.shape) if i not in set(dims)]
        return self.op("reshape", x, shape=shape)

    def matmul_bias(self, bias, a, b):
        out = self.op("matmul", a, b)
        return self.op("add", out, bias) if bias is not None else out

    # -- per-op conversion -----------------------------------------------------

    def convert(self, node: torch.fx.Node):
        t, args, kw = node.target, node.args, node.kwargs

        if t is operator.getitem:
            return self.env[args[0]][args[1]]

        # Identity-like ops: values are immutable, so these are no-ops in Kiln.
        if t in (aten.clone.default, aten.alias.default, aten.detach.default,
                 aten.lift_fresh_copy.default, aten.contiguous.default):
            return self.v(args[0])

        binary = {
            aten.add.Tensor: "add", aten.add.Scalar: "add",
            aten.sub.Tensor: "sub", aten.sub.Scalar: "sub",
            aten.mul.Tensor: "mul", aten.mul.Scalar: "mul",
            aten.div.Tensor: "div", aten.div.Scalar: "div",
            aten.maximum.default: "maximum", aten.minimum.default: "minimum",
        }
        if t in binary:
            a, b = self.v(args[0]), self.v(args[1])
            alpha = kw.get("alpha", 1)
            if alpha != 1:
                b = self.op("mul", b, self.v(alpha))
            return self.op(binary[t], a, b)

        if t is aten.rsub.Scalar:
            return self.op("sub", self.v(args[1]), self.v(args[0]))

        unary = {
            aten.neg.default: "neg", aten.exp.default: "exp", aten.tanh.default: "tanh",
            aten.sigmoid.default: "sigmoid", aten.relu.default: "relu",
            aten.rsqrt.default: "rsqrt", aten.sqrt.default: "sqrt", aten.erf.default: "erf",
        }
        if t in unary:
            return self.op(unary[t], self.v(args[0]))

        if t is aten.gelu.default:
            return self.op("gelu", self.v(args[0]), approximate=kw.get("approximate", "none"))

        if t is aten.pow.Tensor_Scalar and args[1] == 2:
            x = self.v(args[0])
            return self.op("mul", x, x)

        # Linear algebra.
        if t is aten.mm.default:
            return self.op("matmul", self.v(args[0]), self.v(args[1]))
        if t is aten.bmm.default:
            return self.op("matmul", self.v(args[0]), self.v(args[1]))
        if t is aten.addmm.default:
            if kw.get("beta", 1) != 1 or kw.get("alpha", 1) != 1:
                raise Unsupported("addmm with alpha/beta != 1")
            return self.matmul_bias(self.v(args[0]), self.v(args[1]), self.v(args[2]))
        if t is aten.linear.default:
            x, w = self.v(args[0]), self.v(args[1])
            bias = self.v(args[2]) if len(args) > 2 and args[2] is not None else None
            wt = self.op("transpose", w, perm=[1, 0])
            if x.type.rank == 2:
                return self.matmul_bias(bias, x, wt)
            lead = x.shape[:-1]
            flat = self.op("reshape", x, shape=[int(np.prod(lead)), x.shape[-1]])
            out = self.matmul_bias(bias, flat, wt)
            return self.op("reshape", out, shape=[*lead, w.shape[0]])

        # Views.
        if t is aten.t.default:
            x = self.v(args[0])
            return x if x.type.rank < 2 else self.op("transpose", x, perm=[1, 0])
        if t is aten.permute.default:
            return self.op("transpose", self.v(args[0]), perm=[d % len(args[1]) for d in args[1]])
        if t is aten.transpose.int:
            x = self.v(args[0])
            perm = list(range(x.type.rank))
            d0, d1 = args[1] % x.type.rank, args[2] % x.type.rank
            perm[d0], perm[d1] = perm[d1], perm[d0]
            return self.op("transpose", x, perm=perm)
        if t in (aten.view.default, aten.reshape.default, aten._unsafe_view.default,
                 aten.unsqueeze.default, aten.squeeze.dim, aten.squeeze.dims,
                 aten.squeeze.default, aten.flatten.using_ints):
            return self.op("reshape", self.v(args[0]), shape=list(_shape_of(node)))
        if t is aten.expand.default:
            return self.op("expand", self.v(args[0]), shape=list(_shape_of(node)))
        if t is aten.slice.Tensor:
            x = self.v(args[0])
            dim = (args[1] if len(args) > 1 else 0) % x.type.rank
            size = x.shape[dim]
            start = args[2] if len(args) > 2 and args[2] is not None else 0
            end = args[3] if len(args) > 3 and args[3] is not None else size
            step = args[4] if len(args) > 4 else 1
            start, end = start + size if start < 0 else start, end + size if end < 0 else end
            return self.op("slice", x, dim=dim, start=max(0, start), end=min(end, size), step=step)
        if t in (aten.split_with_sizes.default, aten.split.Tensor):
            x = self.v(args[0])
            dim = (args[2] if len(args) > 2 else kw.get("dim", 0)) % x.type.rank
            sizes = args[1]
            if isinstance(sizes, int):
                n = x.shape[dim]
                sizes = [min(sizes, n - i) for i in range(0, n, sizes)]
            outs, start = [], 0
            for s in sizes:
                outs.append(self.op("slice", x, dim=dim, start=start, end=start + s, step=1))
                start += s
            return outs

        # Reductions.
        if t is aten.sum.dim_IntList:
            x = self.v(args[0])
            out = self.reduce("reduce_sum", x, args[1])
            keepdim = args[2] if len(args) > 2 else kw.get("keepdim", False)
            return out if keepdim else self.squeeze(out, [d % x.type.rank for d in args[1]])
        if t is aten.amax.default:
            x = self.v(args[0])
            out = self.reduce("reduce_max", x, args[1])
            keepdim = args[2] if len(args) > 2 else kw.get("keepdim", False)
            return out if keepdim else self.squeeze(out, [d % x.type.rank for d in args[1]])
        if t is aten.mean.dim:
            x = self.v(args[0])
            keepdim = args[2] if len(args) > 2 else kw.get("keepdim", False)
            return self.mean(x, args[1], keepdim)

        # Composite ops, decomposed into Kiln's core op set.
        if t is aten._softmax.default:
            x, dim = self.v(args[0]), args[1] % node.args[0].meta["val"].dim()
            m = self.reduce("reduce_max", x, [dim])
            e = self.op("exp", self.op("sub", x, m))
            return self.op("div", e, self.reduce("reduce_sum", e, [dim]))
        if t is aten.native_layer_norm.default:
            x, normalized_shape = self.v(args[0]), args[1]
            weight, bias, eps = args[2], args[3], args[4]
            dims = list(range(x.type.rank - len(normalized_shape), x.type.rank))
            mean = self.mean(x, dims)
            xc = self.op("sub", x, mean)
            var = self.mean(self.op("mul", xc, xc), dims)
            rstd = self.op("rsqrt", self.op("add", var, self.v(eps)))
            y = self.op("mul", xc, rstd)
            if weight is not None:
                y = self.op("mul", y, self.v(weight))
            if bias is not None:
                y = self.op("add", y, self.v(bias))
            return (y, mean, rstd)

        raise Unsupported(f"Kiln does not support {t} yet")

    # -- driver -----------------------------------------------------------------

    def run(self, gm: torch.fx.GraphModule, placeholder_params=None) -> Graph:
        placeholder_params = placeholder_params or {}
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                shape = _shape_of(node)
                param = placeholder_params.get(node.name)
                if param is not None:
                    name, data = param
                    self.g.constants[name] = data
                    self.env[node] = self.g.add_arg(TensorType(shape), name, param=name)
                else:
                    self.env[node] = self.g.add_arg(TensorType(shape), node.name)
            elif node.op == "call_function":
                out = self.convert(node)
                self.env[node] = out
                if isinstance(out, Value) and isinstance(node.meta.get("val"), torch.Tensor):
                    if tuple(out.shape) != _shape_of(node):
                        raise AssertionError(f"{node.name}: Kiln shape {out.shape} != {_shape_of(node)}")
            elif node.op == "output":
                outs = node.args[0]
                outs = list(outs) if isinstance(outs, (tuple, list)) else [outs]
                if any(not isinstance(o, torch.fx.Node) for o in outs):
                    raise Unsupported("graph returns non-tensor outputs")
                self.g.results = [self.v(o) for o in outs]
            else:
                raise Unsupported(f"unsupported FX node kind '{node.op}'")
        verify(self.g)
        return self.g


def import_fx(gm: torch.fx.GraphModule, name="forward") -> Graph:
    """Import an aot_autograd forward graph (all placeholders are inputs)."""
    return _Importer(name).run(gm)


def import_exported(ep, name="forward") -> Graph:
    """Import a decomposed ``torch.export.ExportedProgram``."""
    from torch.export.graph_signature import InputKind, OutputKind

    params = {}
    for spec in ep.graph_signature.input_specs:
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
            tensor = ep.state_dict.get(spec.target)
            if tensor is None:
                tensor = ep.constants[spec.target]
            data = tensor.detach().to(torch.float32).cpu().numpy().copy()
            params[spec.arg.name] = (spec.target, data)
        elif spec.kind != InputKind.USER_INPUT:
            raise Unsupported(f"unsupported graph input kind {spec.kind}")
    for spec in ep.graph_signature.output_specs:
        if spec.kind != OutputKind.USER_OUTPUT:
            raise Unsupported(f"unsupported graph output kind {spec.kind} (mutation?)")
    return _Importer(name).run(ep.graph_module, params)