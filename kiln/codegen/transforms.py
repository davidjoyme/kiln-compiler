# File: kiln/codegen/transforms.py
"""Loop fusion: turn each multi-op kernel into one loop nest over rows.

The fusion pass (kiln/passes/fusion.py) decides *which* ops share a kernel.
That alone only removes kernel boundaries: inside the kernel, every op would
still compute its whole result before the next op starts, so every
intermediate is a full-size tensor (allocated, written to memory and read
back). Loop fusion fixes that by interleaving the ops' loops:

    for each row r:                       # one iteration of the root's loop
        mean_r  = sum(x[r]) / n           # producers, fused into the loop,
        xc_r    = x[r] - mean_r           # compute only the slice the root
        var_r   = sum(xc_r * xc_r) / n    # needs for this row
        out[r]  = xc_r * rsqrt(var_r + eps) * w + b

Each intermediate is now one row (a few KiB), which bufferization places in
a stack buffer reused across iterations, and the row stays in L1 cache
between the ops that use it.

Kiln decides *what* to fuse and emits it as an MLIR transform-dialect script;
MLIR's transform interpreter does the rewriting:

  1. ``tile_using_for`` the kernel's root (its last op) with tile size 1 on
     every dimension except the innermost, giving a loop over rows;
  2. ``fuse_into_containing_op`` each producer of the root into that loop,
     last producer first, so each is recomputed per row from its own inputs;
  3. ``fold_tensor_empty`` so the fused ops' temporaries shrink to one row.

Kernels with a single op (or a single matmul with no epilogue) have no
intermediates and are left alone.
"""

from __future__ import annotations

from kiln.codegen.emitter import KernelCode

ANY = "!transform.any_op"


class _Script:
    """Builds a transform-dialect script, one SSA handle at a time."""

    def __init__(self):
        self.lines = []
        self.n = 0

    def fresh(self):
        self.n += 1
        return f"%h{self.n}"

    def emit(self, line):
        self.lines.append("    " + line)

    def match_id(self, op_id):
        h = self.fresh()
        self.emit(f'{h} = transform.structured.match attributes {{kiln.id = "{op_id}"}} '
                  f"in %root : ({ANY}) -> {ANY}")
        return h

    def tile(self, h, sizes):
        loops = sum(1 for s in sizes if s)
        tiled = self.fresh()
        loop_names = [self.fresh() for _ in range(loops)]
        types = ", ".join([ANY] * (loops + 1))
        self.emit(f"{', '.join([tiled, *loop_names])} = transform.structured.tile_using_for {h} "
                  f"tile_sizes [{', '.join(map(str, sizes))}] : ({ANY}) -> ({types})")
        return tiled, loop_names

    def fuse(self, producer, containing):
        fused, new_containing = self.fresh(), self.fresh()
        self.emit(f"{fused}, {new_containing} = transform.structured.fuse_into_containing_op "
                  f"{producer} into {containing} : ({ANY}, {ANY}) -> ({ANY}, {ANY})")
        return fused, new_containing

    def func(self, name):
        h = self.fresh()
        self.emit(f'{h} = transform.structured.match ops{{["func.func"]}} attributes '
                  f'{{sym_name = "{name}"}} in %root : ({ANY}) -> {ANY}')
        return h

    def text(self):
        body = "\n".join(self.lines)
        return ("module attributes {transform.with_named_sequence} {\n"
                f"  transform.named_sequence @__transform_main(%root: {ANY} {{transform.readonly}}) {{\n"
                f"{body}\n    transform.yield\n  }}\n}}")


def _closure(by_id, start):
    """``start`` plus everything it (transitively) reads or writes into."""
    seen, stack = set(), list(start)
    while stack:
        i = stack.pop()
        if i not in seen:
            seen.add(i)
            stack.extend(by_id[i].operands + by_id[i].dests)
    return seen


def fusion_plan(code: KernelCode):
    """The ops to fuse into the root's loop, in program order, or None.

    Fused: the producers of the root's inputs and everything they depend on,
    except ops whose *full* result is needed outside the loop. Those are
    computed before the loop, and the fused ops read their finished result:

      * the root's destination (its initial value, e.g. a reduction's fill),
        which after tiling feeds the loop's iteration arguments;
      * other kernel outputs, written in full to a function argument (in
        softmax, ``reduce_max`` when another kernel also reads the row max),
        and everything they depend on.

    Fusing such an op would need its result both per row inside the loop and
    in full outside it. MLIR's fuse_into_containing_op can duplicate a
    producer for that, but it is fragile (it crashed on exactly this case
    during development), and recomputing the op per row is wasted work.

    Returns None if some op is not upstream of the root at all.
    """
    by_id = {op.id: op for op in code.ops}
    root = by_id[code.root]
    outside = _closure(by_id, root.dests + [op.id for op in code.ops
                                            if op.output and op.id != root.id])
    fused = _closure(by_id, [i for i in root.operands if i not in outside])
    fused -= outside
    if fused | outside | {root.id} != set(by_id):
        return None
    return [op for op in code.ops if op.id in fused]


def row_tile_sizes(op):
    """Tile size 1 on every loop of ``op`` except the innermost; 0 on reductions.

    For a matmul-kernel epilogue the roles are b/m/n (batch, rows, columns):
    each iteration computes one output row, so the fused matmul computes one
    row of its result. For row kernels (LayerNorm, softmax) the roles are r
    (row) and i (innermost, possibly reduced).
    """
    sizes = [1] * len(op.roles)
    if sizes:
        sizes[-1] = 0
    for i, role in enumerate(op.roles):
        if role == "k":
            sizes[i] = 0
    return sizes


def build_script(code: KernelCode) -> str | None:
    """Loop-fusion script for one kernel, or None if there's nothing to fuse."""
    if code.kernel.kind == "copy" or code.root is None:
        return None
    root = next(op for op in code.ops if op.id == code.root)
    plan = fusion_plan(code)
    if not plan or not any(op.kind in ("generic", "reduce", "matmul") for op in plan):
        return None   # nothing but fills / temporaries upstream: no intermediates
    sizes = row_tile_sizes(root)
    if not any(sizes):
        return None   # a single row already (e.g. a 1-D kernel)

    s = _Script()
    _, loops = s.tile(s.match_id(root.id), sizes)
    containing = loops[-1]
    for op in reversed(plan):
        _, containing = s.fuse(s.match_id(op.id), containing)
    func = s.func(code.kernel.name)
    s.emit(f"transform.apply_patterns to {func} {{")
    s.emit("  transform.apply_patterns.tensor.fold_tensor_empty")
    s.emit("  transform.apply_patterns.canonicalization")
    s.emit(f"}} : {ANY}")
    return s.text()