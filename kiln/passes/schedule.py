# File: kiln/passes/schedule.py
"""Loop scheduling: choose tile sizes and vector widths for every kernel.

This pass doesn't transform the IR. It annotates each kernel with a schedule
(``kernel @k3 matmul {strategy = "tiled", mc = 128, ...}``) that codegen turns
into an MLIR transform-dialect script. Keeping the *decision* (an analytical
cost model, here) separate from the *mechanism* (MLIR's tiling, fusion and
vectorisation transforms) is the split Halide, TVM and IREE all use.

Matmul kernels: the BLIS / GotoBLAS decomposition
-------------------------------------------------

C[m, n] += A[m, k] * B[k, n] is computed as nested tiles, each sized for one
level of the memory hierarchy:

    for each mc x nc tile of C           # C tile and B panel in L2
      for each kc slice of k
        for each mr x nr register tile   # the micro-kernel
          acc = C[mr, nr]                # mr * nr/VL vector registers
          for each k in kc:              # A and B slivers stream from L1
            acc += broadcast(A[i, k]) * B[k, j:j+nr]     (vector FMAs)
          C[mr, nr] = acc

* Register tile mr x nr. Each k step loads nr/VL vectors of B, broadcasts mr
  scalars of A and issues mr * nr/VL FMAs into accumulators that stay in
  registers for the whole k loop. FMAs per load grow with mr and nr, until
  the accumulators, the B vectors and one broadcast register no longer fit
  in the register file and LLVM starts spilling. nr is one or two vectors.
* kc: an mr x kc sliver of A and a kc x nr sliver of B are reused across the
  k loop, so both should fit in half the L1: (mr + nr) * kc * 4 <= L1 / 2.
* nc: the kc x nc panel of B is reused by every register tile in an mc x nc
  tile of C, so it should fit in half the L2. mc is capped at 128 rows.

Measured on the GPT-2 MLP matmul (128 x 3072 x 768, 1 thread, AVX-512
machine; GFLOP/s), the model's picks (marked *) are the best or close to it:

    AVX-512 (16 lanes, 32 regs)        AVX2 schedules (8 lanes, 16 regs)
    mr x nr   accumulators  GFLOP/s    mr x nr   accumulators  GFLOP/s
    4 x 16         4          64.8     4 x 8          4          44.4
    8 x 16         8         113.3     8 x 8          8          59.7
    4 x 32         8         125.5     2 x 16         4          48.0
    8 x 32 *      16         144.5     4 x 16 *       8          79.2
    4 x 64        16         141.3     2 x 32         8          84.9

Tiles of 3 vectors (nr = 48 on AVX-512, 24 on AVX2) fail to vectorise in
this MLIR version and run at ~3 GFLOP/s, so nr is always 1 or 2 vectors.

Every tile size divides its loop extent, so every tile is full and the
vectoriser sees static shapes: no masking or remainder loops. For odd sizes
that means smaller tiles (a prime n gets nr = 1), which is slower but
always correct.

Other kernels
-------------

Elementwise kernels are vectorised along their innermost dimension in
chunks of 4 vectors: enough independent work to hide FMA latency.

Row kernels (LayerNorm, softmax) use smaller chunks. Each reduction over a
row ends with a horizontal combine of one chunk's partial results, a fixed
cost per row, and rows are short (128 elements for attention at 128 tokens),
so that overhead dominates unless chunks are small. Measured on AVX-512 with
1 thread (chunk width in floats; ms):

    workload             16      32      64      128
    softmax 12x128x128   0.145   0.172   0.214   0.360
    softmax 12x512x512   2.076   1.948   2.225   2.448
    LayerNorm 128x768    0.032   0.029   0.036   0.047
    GELU 128x3072        0.327   0.249   0.235   0.240   (elementwise)

So row kernels use one vector per chunk for rows up to 8 vectors long and
two vectors beyond that.
"""

from __future__ import annotations

from kiln.ir.core import Graph
from kiln.passes.fusion import is_row_reduction
from kiln.passes.manager import Pass
from kiln.target import Target, host

MAX_MR = 8       # rows per register tile: more barely helps and lengthens the A broadcasts
MAX_MC = 128     # rows per cache tile
MAX_NC = 256     # columns per cache tile


def largest_divisor(n, limit, multiple_of=1):
    """Largest d with d | n, d <= limit and multiple_of | d (or plain largest divisor)."""
    best = None
    for d in range(1, min(n, max(limit, 1)) + 1):
        if n % d == 0 and d % multiple_of == 0:
            best = d
    if best is None:
        return largest_divisor(n, limit) if multiple_of > 1 else 1
    return best


def max_mr(nr, t: Target):
    """Most rows a register tile can have: accumulators (mr * nr/VL vectors),
    the B vectors (nr/VL) and one broadcast register must all fit."""
    nr_vectors = -(-nr // t.vector_lanes)
    return max(1, (t.vector_registers - nr_vectors - 1) // nr_vectors)


def cache_tiles(m, n, k, mr, nr, t: Target):
    """kc, mc, nc for a given register tile (see module docstring)."""
    kc = largest_divisor(k, max(1, (t.l1_bytes // 2) // ((mr + nr) * 4)))
    mc = largest_divisor(m, MAX_MC, multiple_of=mr)
    nc = largest_divisor(n, min(MAX_NC, max(nr, (t.l2_bytes // 2) // (kc * 4))), multiple_of=nr)
    return {"mc": mc, "nc": nc, "kc": kc}


def matmul_tiles(m, n, k, t: Target):
    """BLIS-style tile sizes for C[m, n] += A[m, k] * B[k, n] (see module docstring)."""
    vl = t.vector_lanes
    if n % (2 * vl) == 0:
        nr = 2 * vl
    elif n % vl == 0:
        nr = vl
    else:
        nr = largest_divisor(n, 2 * vl)
    mr = largest_divisor(m, min(MAX_MR, max_mr(nr, t)))
    return {**cache_tiles(m, n, k, mr, nr, t), "mr": mr, "nr": nr}


class Schedule(Pass):
    """Annotate kernels with schedules.

    ``matmul_tiles_for(batch, m, n, k, target)`` chooses matmul tiles; it
    defaults to the cost model above. The autotuner (kiln/passes/autotune.py)
    plugs in here, replacing the model's guess with a measured choice.
    """

    name = "schedule"

    def __init__(self, target: Target | None = None, matmul_tiles_for=None):
        self.target = target or host()
        self.matmul_tiles_for = matmul_tiles_for or (
            lambda batch, m, n, k, target: matmul_tiles(m, n, k, target))

    def run(self, graph: Graph):
        t = self.target
        vector = 4 * t.vector_lanes
        counts = {}
        for k in graph.kernels:
            if k.kind == "matmul":
                mm = next(n for n in k.nodes if n.op == "matmul")
                a, b = mm.inputs[0].shape, mm.inputs[1].shape
                batch = 1
                for d in a[:-2]:
                    batch *= d
                tiles = self.matmul_tiles_for(batch, a[-2], b[-1], a[-1], t)
                k.attrs = {"strategy": "tiled", **tiles, "vector": vector}
            elif k.kind == "reduction":
                reductions = [n for n in k.nodes if n.op.startswith("reduce_")]
                if all(is_row_reduction(n) for n in reductions):
                    row = max(n.inputs[0].shape[-1] for n in reductions)
                    lanes = t.vector_lanes
                    k.attrs = {"strategy": "rows", "vector": lanes if row <= 8 * lanes else 2 * lanes}
                else:
                    k.attrs = {"strategy": "loops"}   # e.g. a column sum: plain loops
            else:   # elementwise and copy
                k.attrs = {"strategy": "vectorize", "vector": vector}
            counts[k.attrs["strategy"]] = counts.get(k.attrs["strategy"], 0) + 1
        return counts