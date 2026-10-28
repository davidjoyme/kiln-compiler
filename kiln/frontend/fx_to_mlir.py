"""Translate an ATen-level torch.fx graph into MLIR (tensor/linalg/arith dialects).

The emitted function uses destination-passing style for outputs:

    func.func @forward(%in0: tensor<...>, ..., %out0: memref<...>, ...)

Inputs arrive as tensors (value semantics, easy to optimize), while results are
written into caller-provided memrefs. That keeps the runtime simple: Python
allocates the output buffers, the compiled code fills them in.

Only float32 and static shapes are supported for now. Unsupported ops raise
NotImplementedError, which the backend catches to fall back to eager PyTorch.
"""

from dataclasses import dataclass, field
from numbers import Number
import torch
import torch.fx

aten = torch.ops.aten


@dataclass
class KilnProgram:
    """Everything the later pipeline stages need from the frontend."""

    mlir: str
    input_shapes: list = field(default_factory=list)
    output_shapes: list = field(default_factory=list)
    func_name: str = "forward"


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def _shape_str(shape):
    return "x".join(str(d) for d in shape) + "x" if len(shape) else ""


def tensor_type(shape):
    return f"tensor<{_shape_str(shape)}f32>"


def memref_type(shape):
    return f"memref<{_shape_str(shape)}f32>"


def _float_literal(x):
    # MLIR wants a float literal with a decimal point or exponent: 2 -> "2.0".
    return repr(float(x))


def _node_shape(node):
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise NotImplementedError(f"node {node.name} has no tensor metadata")
    if val.dtype != torch.float32:
        raise NotImplementedError(f"only float32 is supported, got {val.dtype} at {node.name}")
    if any(not isinstance(d, int) for d in val.shape):
        raise NotImplementedError(f"dynamic shapes are not supported yet ({node.name})")
    return list(val.shape)


def _broadcast_map(operand_shape, out_shape):
    """Affine map that reads an operand broadcast (NumPy-style) to out_shape."""
    rank = len(out_shape)
    dims = ", ".join(f"d{i}" for i in range(rank))
    offset = rank - len(operand_shape)
    results = []
    for i, size in enumerate(operand_shape):
        j = offset + i
        results.append("0" if size == 1 and out_shape[j] != 1 else f"d{j}")
    return f"affine_map<({dims}) -> ({', '.join(results)})>"


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

class MLIREmitter:
    def __init__(self, gm: torch.fx.GraphModule, func_name="forward"):
        self.gm = gm
        self.func_name = func_name
        self.lines = []
        self.values = {}   # fx node -> SSA name
        self.shapes = {}   # fx node -> shape
        self.counter = 0

    def fresh(self, prefix="v"):
        self.counter += 1
        return f"%{prefix}{self.counter}"

    def emit(self, line):
        self.lines.append("    " + line)

    # -- building blocks ----------------------------------------------------

    def empty(self, shape):
        name = self.fresh("init")
        self.emit(f"{name} = tensor.empty() : {tensor_type(shape)}")
        return name

    def elementwise(self, operands, out_shape, body_fn):
        """Emit a linalg.generic over out_shape.

        operands: list of fx Nodes or Python numbers. Tensors become generic
        inputs (with broadcasting); numbers become constants in the body.
        body_fn(args) -> list of (lines, result_name), where args are the scalar
        SSA names for each operand in order.
        """
        tensor_ops = [op for op in operands if isinstance(op, torch.fx.Node)]
        rank = len(out_shape)
        maps = [_broadcast_map(self.shapes[op], out_shape) for op in tensor_ops]
        maps.append(_broadcast_map(out_shape, out_shape))
        iters = ", ".join(['"parallel"'] * rank)

        init = self.empty(out_shape)
        result = self.fresh()
        ins = ", ".join(self.values[op] for op in tensor_ops)
        in_types = ", ".join(tensor_type(self.shapes[op]) for op in tensor_ops)
        ins_clause = f"ins({ins} : {in_types}) " if tensor_ops else ""

        block_args = [self.fresh("in") for _ in tensor_ops]
        out_arg = self.fresh("out")
        sig = ", ".join(f"{a}: f32" for a in block_args + [out_arg])

        self.emit(
            f"{result} = linalg.generic {{indexing_maps = [{', '.join(maps)}], "
            f"iterator_types = [{iters}]}} {ins_clause}"
            f"outs({init} : {tensor_type(out_shape)}) {{"
        )
        self.emit(f"^bb0({sig}):")

        # Map every operand to a scalar SSA value inside the body.
        scalar_args, it = [], iter(block_args)
        for op in operands:
            if isinstance(op, torch.fx.Node):
                scalar_args.append(next(it))
            else:
                c = self.fresh("cst")
                self.emit(f"  {c} = arith.constant {_float_literal(op)} : f32")
                scalar_args.append(c)

        body_result = body_fn(scalar_args)
        self.emit(f"  linalg.yield {body_result} : f32")
        self.emit(f"}} -> {tensor_type(out_shape)}")
        return result

    def binary(self, arith_op):
        def body(args):
            r = self.fresh()
            self.emit(f"  {r} = {arith_op} {args[0]}, {args[1]} : f32")
            return r
        return body

    def matmul(self, lhs, rhs, init, out_shape):
        result = self.fresh()
        self.emit(
            f"{result} = linalg.matmul ins({self.values[lhs]}, {self.values[rhs]} : "
            f"{tensor_type(self.shapes[lhs])}, {tensor_type(self.shapes[rhs])}) "
            f"outs({init} : {tensor_type(out_shape)}) -> {tensor_type(out_shape)}"
        )
        return result

    def zeros(self, shape):
        zero = self.fresh("cst")
        self.emit(f"{zero} = arith.constant 0.0 : f32")
        init = self.empty(shape)
        filled = self.fresh()
        self.emit(
            f"{filled} = linalg.fill ins({zero} : f32) outs({init} : {tensor_type(shape)}) "
            f"-> {tensor_type(shape)}"
        )
        return filled

    def transpose(self, node, src, perm):
        out_shape = self.shapes[node]
        init = self.empty(out_shape)
        result = self.fresh()
        self.emit(
            f"{result} = linalg.transpose ins({self.values[src]} : {tensor_type(self.shapes[src])}) "
            f"outs({init} : {tensor_type(out_shape)}) permutation = [{', '.join(map(str, perm))}]"
        )
        return result

    # -- op lowering ----------------------------------------------------------

    def lower_node(self, node):
        target, args, kwargs = node.target, node.args, node.kwargs
        out_shape = self.shapes[node]

        if target in (aten.t.default,):
            src = args[0]
            perm = [1, 0] if len(self.shapes[src]) == 2 else [0]
            return self.transpose(node, src, perm)

        if target is aten.permute.default:
            return self.transpose(node, args[0], list(args[1]))

        if target is aten.mm.default:
            return self.matmul(args[0], args[1], self.zeros(out_shape), out_shape)

        if target is aten.addmm.default:
            bias, lhs, rhs = args
            if kwargs.get("beta", 1) != 1 or kwargs.get("alpha", 1) != 1:
                raise NotImplementedError("addmm with beta/alpha != 1")
            # Initialise the accumulator with the broadcast bias, then let the
            # matmul accumulate into it: out = bias + lhs @ rhs.
            init = self.elementwise([bias], out_shape, lambda a: a[0])
            return self.matmul(lhs, rhs, init, out_shape)

        if target is aten.relu.default:
            def body(a):
                zero, r = self.fresh("cst"), self.fresh()
                self.emit(f"  {zero} = arith.constant 0.0 : f32")
                self.emit(f"  {r} = arith.maximumf {a[0]}, {zero} : f32")
                return r
            return self.elementwise([args[0]], out_shape, body)

        if target in (aten.add.Tensor, aten.sub.Tensor):
            if kwargs.get("alpha", 1) != 1:
                raise NotImplementedError(f"{target} with alpha != 1")
            op = "arith.addf" if target is aten.add.Tensor else "arith.subf"
            return self.elementwise(list(args[:2]), out_shape, self.binary(op))

        if target is aten.mul.Tensor:
            return self.elementwise(list(args[:2]), out_shape, self.binary("arith.mulf"))

        raise NotImplementedError(f"Kiln does not support {target} yet")

    # -- driver -------------------------------------------------------------

    def run(self) -> KilnProgram:
        inputs, outputs = [], []

        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self.shapes[node] = _node_shape(node)
                self.values[node] = f"%arg{len(inputs)}"
                inputs.append(node)
            elif node.op == "call_function":
                for a in node.args:
                    if not isinstance(a, (torch.fx.Node, Number, list, tuple)):
                        raise NotImplementedError(f"unsupported argument {a!r} in {node.name}")
                self.shapes[node] = _node_shape(node)
                self.values[node] = self.lower_node(node)
            elif node.op == "output":
                outputs = list(node.args[0])
            else:
                raise NotImplementedError(f"unsupported fx node kind: {node.op}")

        if any(not isinstance(o, torch.fx.Node) for o in outputs):
            raise NotImplementedError("graph returns non-tensor outputs")

        # Write each result into its caller-provided output buffer.
        out_args = []
        for i, out in enumerate(outputs):
            dest = f"%out{i}"
            out_args.append(f"{dest}: {memref_type(self.shapes[out])}")
            self.emit(
                f"bufferization.materialize_in_destination {self.values[out]} in restrict "
                f"writable {dest} : ({tensor_type(self.shapes[out])}, "
                f"{memref_type(self.shapes[out])}) -> ()"
            )
        self.emit("return")

        in_args = [f"%arg{i}: {tensor_type(self.shapes[n])}" for i, n in enumerate(inputs)]
        header = (
            f"func.func @{self.func_name}({', '.join(in_args + out_args)}) "
            f"attributes {{llvm.emit_c_interface}} {{"
        )
        text = "\n".join(["module {", "  " + header, *self.lines, "  }", "}"])

        return KilnProgram(
            mlir=text,
            input_shapes=[self.shapes[n] for n in inputs],
            output_shapes=[self.shapes[o] for o in outputs],
            func_name=self.func_name,
        )


def fx_to_mlir(gm: torch.fx.GraphModule) -> KilnProgram:
    return MLIREmitter(gm).run()