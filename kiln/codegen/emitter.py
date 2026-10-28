# File: kiln/codegen/emitter.py
"""Lower a Kiln graph (partitioned into kernels, buffers assigned) to MLIR.

Output structure:

    func.func private @k0(%in0: memref<...>, ..., %out0: memref<...>)   // one per kernel
    ...
    func.func @forward(%inputs..., %outputs..., %constants..., %arena: memref<Nxi8>)
        attributes {llvm.emit_c_interface}
      // carve strided views out of the arena / argument buffers
      // and call each kernel in order

Inside a kernel, values are tensors (linalg on tensors), so MLIR's tiling and
fusion transforms apply. Kernel arguments are memrefs, and kernels write their
results directly into them (destination-passing style), so after
bufferization the only memory traffic left is the traffic the memory plan
decided on.

Every tensor inside a kernel uses its *squeezed* shape: unit dimensions are
dropped. Broadcasting then shows up purely in the indexing maps (a dimension
of the iteration space the operand doesn't index), and every map is a
projected permutation, which is what linalg's tiling and fusion handle best.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kiln.codegen.scalar import ScalarBuilder, f32
from kiln.ir import ops
from kiln.ir.core import Graph, Kernel
from kiln.passes.buffers import (Layout, compute_layouts, contiguous_strides, kernel_inputs,
                                 kernel_outputs)


def squeeze(shape):
    return tuple(s for s in shape if s != 1)


def tensor_type(shape):
    dims = squeeze(shape)
    return f"tensor<{''.join(f'{d}x' for d in dims)}f32>"


def layout_memref_type(lay: Layout):
    sizes = [s for s in lay.shape if s != 1]
    strides = [st for s, st in zip(lay.shape, lay.strides) if s != 1]
    body = "".join(f"{d}x" for d in sizes) + "f32"
    if tuple(strides) == contiguous_strides(tuple(sizes)) and lay.offset == 0:
        return f"memref<{body}>", sizes, strides
    return f"memref<{body}, strided<[{', '.join(map(str, strides))}], offset: {lay.offset}>>", sizes, strides


def plain_memref_type(shape):
    return f"memref<{''.join(f'{d}x' for d in shape)}f32>"


def affine_map(nloops, results):
    dims = ", ".join(f"d{i}" for i in range(nloops))
    return f"affine_map<({dims}) -> ({', '.join(results)})>"


@dataclass
class OpInfo:
    """A linalg/tensor op emitted for a kernel, as seen by the scheduler."""

    id: str
    kind: str                  # empty | fill | generic | reduce | matmul
    roles: list                # role of each loop: b/m/n/k for matmul kernels, r/i for others
    extents: list
    operands: list = field(default_factory=list)   # ids of ops producing its inputs
    dests: list = field(default_factory=list)      # ids of ops producing its destinations (outs)
    output: bool = False                           # writes a kernel output (a function argument)


@dataclass
class KernelCode:
    kernel: Kernel
    text: str
    ops: list
    root: str | None


@dataclass
class ModuleCode:
    text: str
    kernels: list
    arg_spec: list            # ("input", i) | ("output", i) | ("const", name) | ("arena", nbytes)
    layouts: dict


class _KernelEmitter:
    def __init__(self, graph: Graph, kernel: Kernel, layouts):
        self.g, self.k, self.layouts = graph, kernel, layouts
        self.lines, self.counter = [], 0
        self.ops: list[OpInfo] = []
        self.env = {}            # Value -> tensor SSA name
        self.producer = {}       # tensor SSA name -> op id
        self.op_counter = 0
        self.out_values = set()  # Values written to the kernel's output arguments

    def fresh(self, prefix="v"):
        self.counter += 1
        return f"%{prefix}{self.counter}"

    def emit(self, line, indent=4):
        self.lines.append(" " * indent + line)

    def new_id(self):
        self.op_counter += 1
        return f"{self.k.name}.{self.op_counter}"

    # -- loop structure ---------------------------------------------------------------

    def is_matmul_kernel(self):
        return self.k.kind == "matmul"

    def output_roles(self, shape):
        """Loop roles for an iteration space equal to ``shape`` (squeezed)."""
        rank = len(shape)
        roles = []
        for i, s in enumerate(shape):
            if s == 1:
                continue
            if self.is_matmul_kernel():
                roles.append("m" if i == rank - 2 else "n" if i == rank - 1 else "b")
            else:
                roles.append("i" if i == rank - 1 else "r")
        return roles

    # -- helpers ----------------------------------------------------------------------

    def init_for(self, v, out_args):
        t = tensor_type(v.shape)
        if v in out_args:
            name = self.fresh("init")
            self.emit(f"{name} = bufferization.to_tensor {out_args[v][0]} restrict writable : "
                      f"{out_args[v][1]} to {t}")
            return name, None
        name = self.fresh("empty")
        op_id = self.new_id()
        self.emit(f'{name} = tensor.empty() {{kiln.id = "{op_id}"}} : {t}')
        self.ops.append(OpInfo(op_id, "empty", [], list(squeeze(v.shape))))
        self.producer[name] = op_id
        return name, op_id

    def fill(self, init, value, v, roles):
        name = self.fresh("fill")
        op_id = self.new_id()
        c = self.fresh("cst")
        self.emit(f"{c} = arith.constant {f32(value)} : f32")
        t = tensor_type(v.shape)
        self.emit(f'{name} = linalg.fill {{kiln.id = "{op_id}"}} ins({c} : f32) outs({init} : {t}) -> {t}')
        dests = [self.producer[init]] if init in self.producer else []
        self.ops.append(OpInfo(op_id, "fill", roles, list(squeeze(v.shape)), [], dests))
        self.producer[name] = op_id
        return name

    def generic(self, kind, roles, extents, inputs, in_maps, outs, out_maps, iterators, body_fn):
        """Emit a linalg.generic. ``outs`` are (init SSA, Value) pairs."""
        op_id = self.new_id()
        nloops = len(roles)
        ins_clause = ""
        if inputs:
            ins_clause = (f"ins({', '.join(self.env[v] for v in inputs)} : "
                          f"{', '.join(tensor_type(v.shape) for v in inputs)}) ")
        out_types = [tensor_type(v.shape) for _, v in outs]
        maps = ", ".join(affine_map(nloops, m) for m in in_maps + out_maps)
        iters = ", ".join(f'"{it}"' for it in iterators)
        result = self.fresh("r")
        multi = len(outs) > 1
        lhs = f"{result}:{len(outs)}" if multi else result
        self.emit(f'{lhs} = linalg.generic {{kiln.id = "{op_id}", indexing_maps = [{maps}], '
                  f"iterator_types = [{iters}]}} {ins_clause}"
                  f"outs({', '.join(i for i, _ in outs)} : {', '.join(out_types)}) {{")
        block_ins = [self.fresh("in") for _ in inputs]
        block_outs = [self.fresh("acc") for _ in outs]
        self.emit("^bb0(" + ", ".join(f"{a}: f32" for a in block_ins + block_outs) + "):", 2)
        sb = ScalarBuilder(self.fresh, lambda line: self.emit(line, 6))
        results = body_fn(sb, block_ins, block_outs)
        self.emit(f"  linalg.yield {', '.join(results)} : {', '.join(['f32'] * len(results))}")
        types = ", ".join(out_types)
        self.emit(f"}} -> {'(' + types + ')' if multi else types}")
        operands = [self.producer[self.env[v]] for v in inputs if self.env[v] in self.producer]
        dests = [self.producer[i] for i, _ in outs if i in self.producer]
        self.ops.append(OpInfo(op_id, kind, roles, extents, operands, dests,
                               output=any(v in self.out_values for _, v in outs)))
        names = [f"{result}#{i}" for i in range(len(outs))] if multi else [result]
        for name, (_, v) in zip(names, outs):
            self.env[v] = name
            self.producer[name] = op_id
        return op_id

    def broadcast_map(self, operand_shape, out_shape):
        loop_of, nxt = {}, 0
        for i, s in enumerate(out_shape):
            if s != 1:
                loop_of[i], nxt = nxt, nxt + 1
        offset = len(out_shape) - len(operand_shape)
        return [f"d{loop_of[offset + j]}" for j, s in enumerate(operand_shape) if s != 1]

    # -- units ------------------------------------------------------------------------

    def units(self):
        """Group kernel nodes into units: elementwise clusters, reductions, matmuls."""
        members = set(self.k.nodes)
        depth, key_of = {}, {}
        for n in self.k.nodes:
            d = 0
            for v in n.inputs:
                if v.node in members:
                    bump = 1 if ops.kind(v.node.op) in ("reduction", "matmul") else 0
                    d = max(d, depth[v.node] + bump)
            depth[n] = d
            key_of[n] = ("ew", d, n.out.shape) if ops.kind(n.op) == "elementwise" else ("op", id(n))
        units, unit_of = {}, {}
        for n in self.k.nodes:
            units.setdefault(key_of[n], []).append(n)
            unit_of[n] = key_of[n]
        deps = {u: set() for u in units}
        for u, nodes in units.items():
            for n in nodes:
                for v in n.inputs:
                    if v.node in members and unit_of[v.node] != u:
                        deps[u].add(unit_of[v.node])
        first = {u: self.k.nodes.index(nodes[0]) for u, nodes in units.items()}
        order, done = [], set()
        while len(order) < len(units):
            ready = [u for u in units if u not in done and deps[u] <= done]
            if not ready:
                raise AssertionError(f"cyclic units in kernel {self.k.name}")
            u = min(ready, key=first.get)
            order.append(units[u])
            done.add(u)
        return order

    def emit_cluster(self, nodes, needed, out_args):
        shape = nodes[0].out.shape
        members = set(nodes)
        inputs = []
        for n in nodes:
            for v in n.inputs:
                if v.node not in members and v not in inputs and not self.g.is_scalar_const(v):
                    inputs.append(v)
        outs_v = [n.out for n in nodes if n.out in needed]
        outs = [(self.init_for(v, out_args)[0], v) for v in outs_v]
        roles = self.output_roles(shape)
        extents = list(squeeze(shape))

        def body(sb, block_ins, block_outs):
            scalar = {v: a for v, a in zip(inputs, block_ins)}
            for n in nodes:
                args = []
                for v in n.inputs:
                    if self.g.is_scalar_const(v):
                        args.append(sb.const(float(self.g.const_data(v))))
                    else:
                        args.append(scalar[v])
                scalar[n.out] = sb.apply(n.op, args, n.attrs)
            return [scalar[v] for v in outs_v]

        self.generic("generic", roles, extents, inputs,
                     [self.broadcast_map(v.shape, shape) for v in inputs],
                     outs, [self.broadcast_map(shape, shape) for _ in outs],
                     ["parallel"] * len(roles), body)

    def emit_reduction(self, n, out_args):
        x = n.inputs[0]
        dims = set(n.attrs["dims"])
        loops = [i for i, s in enumerate(x.shape) if s != 1]
        roles = ["i" if i in dims else "r" for i in loops]
        iterators = ["reduction" if i in dims else "parallel" for i in loops]
        out_map = [f"d{j}" for j, i in enumerate(loops) if i not in dims]
        init, _ = self.init_for(n.out, out_args)
        init = self.fill(init, ops.REDUCTION_INIT[n.op], n.out, [r for r in roles if r != "i"])
        combine = ops.REDUCTION_COMBINER[n.op]
        op_id = self.generic("reduce", roles, [x.shape[i] for i in loops], [x],
                             [[f"d{j}" for j in range(len(loops))]], [(init, n.out)], [out_map],
                             iterators, lambda sb, i, o: [sb.apply(combine, [i[0], o[0]], {})])
        return op_id

    def emit_matmul(self, n, out_args):
        a, b = n.inputs
        batch = a.shape[:-2]
        m, kk, nn = a.shape[-2], a.shape[-1], b.shape[-1]
        roles, extents = [], []
        loop = {}
        for i, s in enumerate(batch):
            if s != 1:
                loop[("b", i)] = len(roles)
                roles.append("b")
                extents.append(s)
        for role, size in (("m", m), ("n", nn), ("k", kk)):
            if size != 1:
                loop[role] = len(roles)
                roles.append(role)
                extents.append(size)

        def operand_map(dims):
            res = []
            for i, s in enumerate(batch):
                if s != 1:
                    res.append(f"d{loop[('b', i)]}")
            for role, size in dims:
                if size != 1:
                    res.append(f"d{loop[role]}")
            return res

        init, _ = self.init_for(n.out, out_args)
        init = self.fill(init, 0.0, n.out, [r for r in roles if r != "k"])
        iterators = ["reduction" if r == "k" else "parallel" for r in roles]
        return self.generic(
            "matmul", roles, extents, [a, b],
            [operand_map([("m", m), ("k", kk)]), operand_map([("k", kk), ("n", nn)])],
            [(init, n.out)], [operand_map([("m", m), ("n", nn)])], iterators,
            lambda sb, i, o: [sb.add(o[0], sb.mul(i[0], i[1]))])

    # -- driver -------------------------------------------------------------------------

    def run_copy(self, inputs, outputs):
        """Copy kernels are emitted on buffers, not tensors.

        On tensors a copy is the identity function and canonicalization would
        rightly erase it, leaving a generic memref.copy for bufferization to
        insert (which, for strided sources, needs a runtime library call).
        On buffers the copy is a real side effect: a linalg.generic that
        tiles, vectorises and lowers to loops like any other kernel.
        """
        (src,), (dst,) = inputs, outputs
        src_t, _, _ = layout_memref_type(self.layouts[src])
        dst_t, _, _ = layout_memref_type(self.layouts[dst])
        shape = dst.shape
        roles = self.output_roles(shape)
        identity = affine_map(len(roles), [f"d{i}" for i in range(len(roles))])
        op_id = self.new_id()
        iters = ", ".join(['"parallel"'] * len(roles))
        self.emit(f'linalg.generic {{kiln.id = "{op_id}", indexing_maps = [{identity}, {identity}], '
                  f"iterator_types = [{iters}]}} ins(%in0 : {src_t}) outs(%out0 : {dst_t}) {{")
        self.emit("^bb0(%a: f32, %b: f32):", 2)
        self.emit("  linalg.yield %a : f32")
        self.emit("}")
        self.emit("return")
        self.ops.append(OpInfo(op_id, "generic", roles, list(squeeze(shape))))
        header = (f"  func.func private @{self.k.name}(%in0: {src_t}, %out0: {dst_t}) "
                  f'attributes {{kiln.kernel = "copy"}} {{')
        text = "\n".join([header, *self.lines, "  }"])
        return KernelCode(self.k, text, self.ops, op_id), [src_t, dst_t], [src, dst]

    def run(self):
        inputs = kernel_inputs(self.g, self.k)
        outputs = kernel_outputs(self.g, self.k)
        if self.k.kind == "copy":
            return self.run_copy(inputs, outputs)
        args, out_args = [], {}
        for i, v in enumerate(inputs):
            t, _, _ = layout_memref_type(self.layouts[v])
            args.append((f"%in{i}", t))
            name = self.fresh("t")
            self.emit(f"{name} = bufferization.to_tensor %in{i} restrict : {t} to {tensor_type(v.shape)}")
            self.env[v] = name
        for i, v in enumerate(outputs):
            t, _, _ = layout_memref_type(self.layouts[v])
            args.append((f"%out{i}", t))
            out_args[v] = (f"%out{i}", t)
            self.out_values.add(v)

        members = set(self.k.nodes)
        users = self.g.users()
        units = self.units()
        unit_needed = set(outputs)
        for unit in units:
            unit_set = set(unit)
            for n in unit:
                if any(u in members and u not in unit_set for u in users.get(n.out, [])):
                    unit_needed.add(n.out)

        root = None
        for unit in units:
            first = unit[0]
            if ops.kind(first.op) == "elementwise":
                self.emit_cluster(unit, unit_needed, out_args)
            elif first.op == "matmul":
                self.emit_matmul(first, out_args)
            elif ops.kind(first.op) == "reduction":
                self.emit_reduction(first, out_args)
            else:
                raise NotImplementedError(first.op)
            root = self.ops[-1].id

        for v, (name, t) in out_args.items():
            self.emit(f"bufferization.materialize_in_destination {self.env[v]} in restrict writable "
                      f"{name} : ({tensor_type(v.shape)}, {t}) -> ()")
        self.emit("return")
        sig = ", ".join(f"{n}: {t}" for n, t in args)
        header = f"  func.func private @{self.k.name}({sig}) attributes {{kiln.kernel = \"{self.k.kind}\"}} {{"
        text = "\n".join([header, *self.lines, "  }"])
        return KernelCode(self.k, text, self.ops, root), [t for _, t in args], inputs + outputs


def emit_module(graph: Graph, profile: bool = False) -> ModuleCode:
    """Emit the MLIR module. ``profile`` adds one entry point per kernel
    (``@forward__k3``) so the runtime can time kernels individually."""
    layouts = compute_layouts(graph)
    counter = [0]

    def fresh(prefix):
        counter[0] += 1
        return f"%{prefix}{counter[0]}"

    # Entry-point arguments: inputs, outputs, constants, arena.
    arg_spec, params = [], []
    for i, a in enumerate(graph.args):
        arg_spec.append(("input", i))
        params.append((f"%input{i}", plain_memref_type(a.shape)))
    for i, r in enumerate(graph.results):
        arg_spec.append(("output", i))
        params.append((f"%output{i}", plain_memref_type(r.shape)))
    const_args, const_shape = {}, {}
    for n in graph.nodes:
        if n.op == "const" and n.out.type.rank > 0 and n.out in layouts:
            name = n.attrs["name"]
            if name not in const_args:
                const_args[name] = f"%const{len(const_args)}"
                const_shape[name] = n.out.shape
                arg_spec.append(("const", name))
                params.append((const_args[name], plain_memref_type(n.out.shape)))
    arena = graph.attrs.get("arena_bytes", 0)
    if arena:
        arg_spec.append(("arena", arena))
        params.append(("%arena", f"memref<{arena}xi8>"))

    class _Body:
        """Builds an entry-point body: buffer views + kernel calls."""

        def __init__(self):
            self.lines, self.bases, self.views = [], {}, {}

        def base(self, b):
            if b in self.bases:
                return self.bases[b]
            kind = b[0]
            if kind == "input":
                res = (f"%input{b[1]}", plain_memref_type(graph.args[b[1]].shape))
            elif kind == "output":
                res = (f"%output{b[1]}", plain_memref_type(graph.results[b[1]].shape))
            elif kind == "const":
                res = (const_args[b[1]], plain_memref_type(const_shape[b[1]]))
            else:  # arena: a typed view of the buffer's byte range
                v = b[1]
                off, name = fresh("off"), fresh("buf")
                t = f"memref<{v.type.numel}xf32>"
                self.lines.append(f"    {off} = arith.constant {v.node.attrs['offset']} : index")
                self.lines.append(f"    {name} = memref.view %arena[{off}][] : memref<{arena}xi8> to {t}")
                res = (name, t)
            self.bases[b] = res
            return res

        def memref_for(self, v):
            if v in self.views:
                return self.views[v]
            lay = layouts[v]
            b_name, b_type = self.base(lay.base)
            t, sizes, strides = layout_memref_type(lay)
            if t == b_type:
                self.views[v] = (b_name, t)
                return self.views[v]
            name = fresh("view")
            self.lines.append(
                f"    {name} = memref.reinterpret_cast {b_name} to offset: [{lay.offset}], "
                f"sizes: [{', '.join(map(str, sizes))}], strides: [{', '.join(map(str, strides))}] "
                f": {b_type} to {t}")
            self.views[v] = (name, t)
            return self.views[v]

        def call(self, code, arg_types, arg_values):
            operands = [self.memref_for(v) for v in arg_values]
            self.lines.append(f"    func.call @{code.kernel.name}({', '.join(n for n, _ in operands)}) : "
                              f"({', '.join(arg_types)}) -> ()")

        def func(self, name):
            header = (f"  func.func @{name}({', '.join(f'{n}: {t}' for n, t in params)}) "
                      f"attributes {{llvm.emit_c_interface}} {{")
            return "\n".join([header, *self.lines, "    return", "  }"])

    kernels, calls = [], []
    main = _Body()
    for k in graph.kernels:
        code, arg_types, arg_values = _KernelEmitter(graph, k, layouts).run()
        kernels.append(code)
        calls.append((code, arg_types, arg_values))
        main.call(code, arg_types, arg_values)

    funcs = [main.func(graph.name)]
    if profile:
        for code, arg_types, arg_values in calls:
            body = _Body()
            body.call(code, arg_types, arg_values)
            funcs.append(body.func(f"{graph.name}__{code.kernel.name}"))

    text = "\n".join(["module {", *(k.text for k in kernels), *funcs, "}"])
    return ModuleCode(text, kernels, arg_spec, layouts)