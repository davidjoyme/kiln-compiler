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

Tiling and vectorisation
------------------------

When the schedule pass (kiln/passes/schedule.py) has annotated a kernel,
the script also tiles for the cache and registers and vectorises:

  tiled      (matmul kernels) tile the root into mc x nc cache tiles and fuse
             the matmul and its producers in, so the epilogue reads a tile
             that's still in cache; split k by kc; tile the matmul into
             mr x nr x 1 register tiles; vectorise, which turns each register
             tile's k step into mr broadcasts and mr * nr/VL vector FMAs
             (outer-product form); hoist the accumulator out of the k loop
             so it lives in registers.
  rows       (LayerNorm, softmax) loop fusion over rows as above, then split
             each row-wide op into vector-width chunks. Reductions use a
             vector of partial accumulators and one horizontal reduction at
             the end (tile_reduction_using_for).
  vectorize  (elementwise and copy kernels) tile to vector-width chunks of
             the innermost dimension and vectorise.

Every op ends up as a loop nest over vectors of at most ``vector`` lanes.
Vectorising a whole tensor at once (one 98304-element vector) is legal but
makes LLVM's instruction selection take seconds and spill everywhere.
"""

from __future__ import annotations

from kiln.codegen.emitter import KernelCode
from kiln.passes.schedule import largest_divisor

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

    def tile_reduction(self, h, sizes):
        names = [self.fresh() for _ in range(4)]
        self.emit(f"{', '.join(names)} = transform.structured.tile_reduction_using_for {h} "
                  f"by tile_sizes = [{', '.join(map(str, sizes))}] : ({ANY}) -> ({', '.join([ANY] * 4)})")
        return names

    def patterns(self, target, names):
        self.emit(f"transform.apply_patterns to {target} {{")
        for name in names:
            self.emit(f"  transform.apply_patterns.{name}")
        self.emit(f"}} : {ANY}")

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


def loop_fusion_script(code: KernelCode) -> str | None:
    """Loop fusion only (no tiling/vectorisation), or None if there's nothing to fuse."""
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


# ---------------------------------------------------------------------------
# Scheduled kernels: tiling + vectorisation
# ---------------------------------------------------------------------------

def _fuse_into_rows(s: _Script, code: KernelCode, root, sizes, plan):
    """Tile ``root`` with ``sizes`` and fuse ``plan`` into the innermost loop."""
    tiled, loops = s.tile(s.match_id(root.id), sizes)
    handles = {root.id: tiled}
    containing = loops[-1]
    for op in reversed(plan):
        handles[op.id], containing = s.fuse(s.match_id(op.id), containing)
    s.patterns(s.func(code.kernel.name), ["tensor.fold_tensor_empty"])
    return handles


def _vector_chunks(s: _Script, code: KernelCode, handles, vw, only=None, tiled=None):
    """Split ops into loops over vectors of at most ``vw`` lanes.

    Outer dimensions are walked one at a time; the innermost dimension is cut
    into the largest chunk <= vw that divides it. Ops already fused into a
    tile loop (those in ``handles``) see the *tiled* extents: ``tiled`` maps a
    loop role to its extent inside the tile (1 for each row of a row kernel,
    mc / nc for a matmul's cache tile). Using the original extents there
    would cut a size-1 dimension into 64-wide chunks, leaving dynamically
    shaped tiles the vectoriser rejects.
    """
    tiled = tiled or {}
    for op in code.ops:
        if op.kind not in ("generic", "reduce", "fill") or not op.roles:
            continue
        if only is not None and op.id not in only:
            continue
        extents = [tiled.get(r, e) if op.id in handles else e for r, e in zip(op.roles, op.extents)]
        outer = [1 if e > 1 else 0 for e in extents[:-1]]
        inner = extents[-1]
        chunk = largest_divisor(inner, vw)
        chunk = chunk if chunk < inner else 0
        h = handles.get(op.id) or s.match_id(op.id)
        if op.kind == "reduce" and op.roles[-1] == "i":
            # Rows with a plain loop, then the reduction with partial vector sums.
            if any(outer):
                h, _ = s.tile(h, outer + [0])
            if chunk:
                s.tile_reduction(h, [0] * len(outer) + [chunk])
            continue
        if any(outer) or chunk:
            s.tile(h, outer + [chunk])


def _rows(s: _Script, code: KernelCode, vw):
    root = next(op for op in code.ops if op.id == code.root)
    plan = fusion_plan(code)
    sizes = row_tile_sizes(root)
    if plan is None or not any(sizes) or root.kind not in ("generic", "reduce"):
        _vector_chunks(s, code, {}, vw)
        return
    handles = _fuse_into_rows(s, code, root, sizes, plan)
    in_row = {r: 1 for r, size in zip(root.roles, sizes) if size}
    _vector_chunks(s, code, handles, vw, tiled=in_row)


def _matmul(s: _Script, code: KernelCode, attrs, vw):
    mm = next(op for op in code.ops if op.kind == "matmul")
    root = next(op for op in code.ops if op.id == code.root)

    # 1. Cache tiles of the root (b -> 1, m -> mc, n -> nc), matmul fused in.
    cache = {"b": 1, "m": attrs["mc"], "n": attrs["nc"], "k": 0}
    sizes = [cache[r] if cache[r] < e or r == "b" else 0 for r, e in zip(root.roles, root.extents)]
    plan = fusion_plan(code)
    handles = {}
    if any(sizes) and plan is not None:
        handles = _fuse_into_rows(s, code, root, sizes, plan)
    in_tile = {r: size for r, size in zip(root.roles, sizes) if size}
    mm_h = handles.get(mm.id) or s.match_id(mm.id)

    # 2. Split k into kc slices, then mr x nr x 1 register tiles.
    if "k" in mm.roles and attrs["kc"] < mm.extents[mm.roles.index("k")]:
        mm_h, _ = s.tile(mm_h, [attrs["kc"] if r == "k" else 0 for r in mm.roles])
    micro = {"b": 1, "m": attrs["mr"], "n": attrs["nr"], "k": 1}
    s.tile(mm_h, [micro[r] for r in mm.roles])

    # 3. Epilogue and the accumulator's zero-fill: vector chunks of a row.
    others = {op.id for op in code.ops if op.id != mm.id and op.kind in ("generic", "fill")}
    _vector_chunks(s, code, handles, vw, only=others, tiled=in_tile)


VECTOR_LOWERING = [
    'vector.lower_contraction lowering_strategy = "outerproduct"',
    "vector.lower_outerproduct",
    'vector.multi_reduction_flattening lowering_strategy = "innerreduction"',
    'vector.multi_reduction_unrolling lowering_strategy = "innerreduction"',
    "vector.transfer_permutation_patterns",
    "tensor.fold_tensor_subset_ops_into_vector_transfers",
    "linalg.tiling_canonicalization",
    "scf.for_loop_canonicalization",
    "canonicalization",
]


def scheduled_script(code: KernelCode) -> str | None:
    attrs = code.kernel.attrs
    strategy = attrs.get("strategy")
    if strategy == "loops":
        return loop_fusion_script(code)
    vw = attrs.get("vector", 16)
    s = _Script()
    if strategy == "tiled":
        _matmul(s, code, attrs, vw)
    elif strategy == "rows":
        _rows(s, code, vw)
    elif strategy == "vectorize":
        _vector_chunks(s, code, {}, vw)
    else:
        raise ValueError(f"unknown schedule strategy {strategy!r}")

    # Vectorise the linalg ops (not tensor.insert_slice: vectorising those turns
    # in-place tile updates into redundant copies after bufferization).
    # linalg.reduce is the final combine tile_reduction_using_for creates for a
    # split reduction: left scalar, it's a chain of 64 dependent max/add ops
    # per row, which made vectorised softmax 2x slower than scalar loops.
    # Vectorised, it becomes a horizontal reduction (a log2(64)-step shuffle tree).
    func = s.func(code.kernel.name)
    ops = s.fresh()
    s.emit(f'{ops} = transform.structured.match ops{{["linalg.generic", "linalg.fill", "linalg.reduce"]}} '
           f"in {func} : ({ANY}) -> {ANY}")
    s.emit(f"transform.structured.vectorize {ops} : {ANY}")
    # Recognise contractions again after vectorising (generic -> multi_reduction
    # -> contract), then lower them to outer products: broadcast + FMA.
    s.patterns(func, ["vector.reduction_to_contract", "vector.transfer_permutation_patterns",
                      "canonicalization"])
    s.patterns(func, VECTOR_LOWERING)
    # Keep the accumulator in registers: hoist its transfer_read/transfer_write
    # out of the k loop and carry it as a loop iteration argument.
    loops = s.fresh()
    s.emit(f'{loops} = transform.structured.match ops{{["scf.for"]}} in {func} : ({ANY}) -> {ANY}')
    s.emit(f"transform.loop.hoist_loop_invariant_subsets {loops} : {ANY}")
    s.patterns(func, ["canonicalization"])
    return s.text()


def build_script(code: KernelCode) -> str | None:
    """The transform script for one kernel, or None to compile it as it is."""
    if code.root is None:
        return None
    if "strategy" in code.kernel.attrs:
        return scheduled_script(code)
    if code.kernel.kind == "copy":
        return None
    return loop_fusion_script(code)