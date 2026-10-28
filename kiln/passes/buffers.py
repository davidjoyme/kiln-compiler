# File: kiln/passes/buffers.py
"""Buffer assignment: decide where every value lives.

Buffer assignment decides where every value lives:

  * graph inputs, constants and graph outputs live in caller-provided buffers;
  * every value produced by one kernel and read by another (a "kernel output")
    gets its own contiguous buffer;
  * values that only live inside a kernel get no buffer at all: that is the
    point of fusion;
  * view ops (transpose, slice, reshape, expand) get no buffer either. They
    become *strided views* of their source buffer: the same memory read with
    different offsets and strides. A transpose costs nothing.

Some views cannot be expressed as strides (reshaping a transposed tensor, for
example), and the matmul kernel needs the N dimension of its right-hand
operand contiguous to vectorise. In those cases, and only those, buffer
assignment inserts an explicit ``copy`` kernel to *realise* the value in a
contiguous buffer. This is the one place where layout is decided, which keeps
fusion and codegen free of layout logic.

Each kernel output then needs an offset in the *arena*: one preallocated
block of memory, allocated once at load time and reused on every call, so
the compiled code performs no heap allocation. In this stage every buffer
simply gets its own slot; the memory-planning stage replaces that with
liveness-based reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from kiln.ir import ops
from kiln.ir.core import Graph, Kernel, Value
from kiln.passes.manager import Pass

ALIGNMENT = 64  # bytes: one cache line, and enough for any vector load


@dataclass(frozen=True)
class Layout:
    """Where a value's elements live: base buffer + offset + strides (in elements)."""

    base: tuple            # ("input", i) | ("output", i) | ("const", name) | ("arena", Value)
    offset: int
    shape: tuple
    strides: tuple

    @property
    def contiguous(self):
        return self.strides == contiguous_strides(self.shape)


def contiguous_strides(shape):
    strides, acc = [], 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return tuple(reversed(strides))


def reshape_strides(shape, strides, new_shape):
    """Strides for viewing (shape, strides) as new_shape, or None if impossible.

    Port of PyTorch's computeStride: a reshape is a free view iff each group
    of merged dimensions is contiguous with respect to itself.
    """
    if prod(shape) == 0:
        return contiguous_strides(new_shape)
    new_strides = [0] * len(new_shape)
    view_d = len(new_shape) - 1
    chunk_base_stride = strides[-1] if strides else 1
    tensor_numel, view_numel = 1, 1
    for tensor_d in range(len(shape) - 1, -1, -1):
        tensor_numel *= shape[tensor_d]
        if tensor_d == 0 or (shape[tensor_d - 1] != 1 and
                             strides[tensor_d - 1] != tensor_numel * chunk_base_stride):
            while view_d >= 0 and (view_numel < tensor_numel or new_shape[view_d] == 1):
                new_strides[view_d] = view_numel * chunk_base_stride
                view_numel *= new_shape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if tensor_d > 0:
                chunk_base_stride = strides[tensor_d - 1]
                tensor_numel, view_numel = 1, 1
    if view_d != -1:
        return None
    return tuple(new_strides)


def view_layout(op, attrs, src: Layout, out_shape):
    """Layout of a view op applied to ``src``, or None if it needs a copy."""
    if op == "transpose":
        perm = attrs["perm"]
        return Layout(src.base, src.offset, tuple(out_shape), tuple(src.strides[p] for p in perm))
    if op == "slice":
        d, step = attrs["dim"], attrs.get("step", 1)
        strides = list(src.strides)
        offset = src.offset + attrs["start"] * strides[d]
        strides[d] *= step
        return Layout(src.base, offset, tuple(out_shape), tuple(strides))
    if op == "expand":
        lead = len(out_shape) - len(src.shape)
        strides = [0] * lead
        for i, (s, st) in enumerate(zip(src.shape, src.strides)):
            strides.append(0 if s == 1 and out_shape[lead + i] != 1 else st)
        return Layout(src.base, src.offset, tuple(out_shape), tuple(strides))
    if op == "reshape":
        strides = reshape_strides(src.shape, src.strides, tuple(out_shape))
        return None if strides is None else Layout(src.base, src.offset, tuple(out_shape), strides)
    raise ValueError(op)


def kernel_outputs(graph: Graph, k: Kernel, users=None):
    """Values computed in kernel ``k`` that are needed outside it."""
    users = users if users is not None else graph.users()
    members = set(k.nodes)
    results = set(graph.results)
    outs = []
    for n in k.nodes:
        if n.out in results or any(u not in members for u in users.get(n.out, [])):
            outs.append(n.out)
    return outs


def kernel_inputs(graph: Graph, k: Kernel):
    """Values read by kernel ``k`` that come from outside it (scalars excluded)."""
    members = set(k.nodes)
    ins = []
    for n in k.nodes:
        for v in n.inputs:
            if (v.node not in members or v.node is None) and v not in ins \
                    and not graph.is_scalar_const(v):
                ins.append(v)
    return ins


def compute_layouts(graph: Graph) -> dict:
    """Layout of every value that has storage (args, consts, kernel outputs, views).

    Pure function of the (buffer-assigned) graph, so codegen can recompute it
    from parsed IR.
    """
    layouts = {}
    for i, a in enumerate(graph.args):
        layouts[a] = Layout(("input", i), 0, a.shape, contiguous_strides(a.shape))
    output_index = {}
    for i, r in enumerate(graph.results):
        output_index.setdefault(r, i)
    for n in graph.nodes:
        v = n.out
        if n.op == "const":
            if v.type.rank > 0:
                layouts[v] = Layout(("const", n.attrs["name"]), 0, v.shape, contiguous_strides(v.shape))
        elif ops.kind(n.op) == "view":
            src = layouts.get(n.inputs[0])
            if src is not None:
                lay = view_layout(n.op, n.attrs, src, v.shape)
                if lay is not None:
                    layouts[v] = lay
        elif "buffer" in n.attrs:
            base = ("output", output_index[v]) if n.attrs["buffer"] == "output" else ("arena", v)
            layouts[v] = Layout(base, 0, v.shape, contiguous_strides(v.shape))
    return layouts


class BufferAssignment(Pass):
    name = "buffer-assign"

    def _insert_copy(self, graph: Graph, v: Value, before_node):
        index = graph.nodes.index(before_node) if before_node is not None else len(graph.nodes)
        copy = graph.add_node("copy", [v], {}, index=index)
        k = Kernel(f"copy{len(graph.kernels)}", "copy", [copy.node])
        copy.node.attrs["kernel"] = k.name
        graph.kernels.append(k)
        return copy

    def run(self, graph: Graph):
        copies = 0
        # 1. Graph results must be written by a kernel, each into its own output.
        seen = set()
        for i, r in enumerate(graph.results):
            producer_is_kernel = r.node is not None and "kernel" in r.node.attrs
            if not producer_is_kernel or r in seen:
                graph.results[i] = self._insert_copy(graph, r, None)
                copies += 1
            seen.add(graph.results[i])

        # 2. Mark materialised kernel outputs.
        users = graph.users()
        results = set(graph.results)
        for k in graph.kernels:
            for v in kernel_outputs(graph, k, users):
                v.node.attrs["buffer"] = "output" if v in results else "arena"

        # 3. Resolve views to strided layouts; realise the ones that can't be.
        changed = True
        while changed:
            changed = False
            layouts = compute_layouts(graph)
            for n in list(graph.nodes):
                if ops.kind(n.op) == "view" and n.out not in layouts and n.inputs[0] in layouts:
                    c = self._insert_copy(graph, n.inputs[0], n)
                    c.node.attrs["buffer"] = "arena"
                    n.inputs[0] = c
                    copies += 1
                    changed = True
                    break
                if n.op == "matmul":
                    b = n.inputs[1]
                    lay = layouts.get(b)
                    if lay is not None and b.shape[-1] > 1 and lay.strides[-1] != 1:
                        c = self._insert_copy(graph, b, n)
                        c.node.attrs["buffer"] = "arena"
                        n.inputs[1] = c
                        copies += 1
                        changed = True
                        break

        # 4. Annotate views with their resolved strides (for dumps and tests).
        layouts = compute_layouts(graph)
        for n in graph.nodes:
            if ops.kind(n.op) == "view" and n.out in layouts:
                lay = layouts[n.out]
                n.attrs["strides"] = list(lay.strides)
                n.attrs["offset"] = lay.offset

        _order_kernels(graph)
        return {"copies": copies} if copies else None


def _order_kernels(graph: Graph):
    """Topologically order kernels (copies are appended out of order)."""
    graph.toposort()
    pos = {n: i for i, n in enumerate(graph.nodes)}
    owner = {n: k for k in graph.kernels for n in k.nodes}

    def deps(k):
        out = set()
        for n in k.nodes:
            for v in n.inputs:
                src = v
                while src.node is not None and ops.kind(src.node.op) == "view":
                    src = src.node.inputs[0]
                if src.node in owner and owner[src.node] is not k:
                    out.add(owner[src.node])
        return out

    order, done = [], set()

    def visit(k):
        if k in done:
            return
        done.add(k)
        for d in sorted(deps(k), key=lambda d: pos[d.nodes[0]]):
            visit(d)
        order.append(k)

    for k in sorted(graph.kernels, key=lambda k: pos[k.nodes[0]]):
        visit(k)
    for i, k in enumerate(order):
        new_name = f"k{i}"
        for n in k.nodes:
            n.attrs["kernel"] = new_name
        k.name = new_name
        k.nodes.sort(key=pos.get)
    graph.kernels = order


def _align(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


class MemoryPlan(Pass):
    """Assign arena offsets: one aligned slot per buffer, no reuse (yet)."""

    name = "memory-plan"

    def run(self, graph: Graph):
        layouts = compute_layouts(graph)
        offset = 0
        count = 0
        for v, lay in layouts.items():
            if lay.base[0] == "arena" and lay.base[1] is v:
                v.node.attrs["offset"] = offset
                offset += _align(v.type.nbytes)
                count += 1
        graph.attrs["arena_bytes"] = offset
        return {"buffers": count, "arena_bytes": offset}