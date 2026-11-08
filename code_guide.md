<!-- File: docs/README.md -->

# Kiln code guide

A file-by-file tour of Kiln, written for studying the code and explaining it
in interviews. The top-level `README.md` says *what* Kiln does and how fast it
is. This guide says *where* each piece lives and *why* it's built that way.

How to use it:

1. Read **The pitch** and **The pipeline** until you can say them without notes.
2. Walk through **One layer through the whole compiler**. It shows every
   stage on one tiny example.
3. Read the files in the order of **File by file**. Each entry has what the
   file does, the ideas worth knowing, and a question to be ready for.
4. Finish with **Interview questions** and **Numbers to know**.

---

## The pitch

> Kiln is an inference compiler for PyTorch. It takes a model, imports it into
> its own graph IR, and optimizes it with the passes production compilers use:
> weight freezing, constant folding, operator fusion, buffer assignment and
> static memory planning. Then it generates code for two targets. On CPU it
> lowers to MLIR's linalg dialect and uses transform-dialect scripts to tile
> and vectorize, with tile sizes from a BLIS-style cost model or an autotuner.
> On NVIDIA GPUs it generates Triton kernels, replays them as a CUDA graph and
> autotunes the matmul tiles. On an RTX 5090, a GPT-2 transformer block runs
> 1.2–1.9x faster than eager PyTorch and matches torch.compile.

The one-line version: **the same graph optimizer feeds two code generators,
the way TorchInductor and XLA are built.**

---

## The pipeline

```
PyTorch model
   │  torch.export + ATen decompositions               kiln/api.py
   ▼
FX graph  ──────────────────────────────────────────►  kiln/frontend/fx_import.py
   ▼
Kiln IR (SSA graph of f32 tensors)                     kiln/ir/
   │
   │  Canonicalize, CSE                                kiln/passes/simplify.py
   │  FreezeParameters, ConstantFold                   kiln/passes/freeze.py
   │  DCE                                              kiln/passes/simplify.py
   │  Fusion  (or OneOpPerKernel)                      kiln/passes/fusion.py, partition.py
   │  BufferAssignment                                 kiln/passes/buffers.py
   │  MemoryPlan                                       kiln/passes/memory_plan.py
   │  Schedule (CPU only; tiles from the cost model    kiln/passes/schedule.py
   │            or the autotuner)                      kiln/passes/autotune.py
   ▼
   ├── CPU ───────────────────────────────────┐   ┌── GPU ─────────────────────────────┐
   │ emit MLIR: linalg on tensors             │   │ emit Triton source, one kernel per │
   │   kiln/codegen/emitter.py, scalar.py     │   │ Kiln kernel   kiln/gpu/codegen.py  │
   │ transform-dialect scripts: fuse, tile,   │   │ autotune matmul tiles              │
   │ vectorize   kiln/codegen/transforms.py   │   │   kiln/gpu/autotune.py             │
   │ bufferize, lower to LLVM                 │   │ arena on device, CUDA graph replay │
   │   kiln/codegen/lowering.py               │   │   kiln/gpu/runtime.py              │
   │ JIT + ctypes calls                       │   └────────────────────────────────────┘
   │   kiln/runtime/executor.py               │
   └──────────────────────────────────────────┘
```

The design choice to emphasize: **every graph-level pass is shared.** The GPU
backend only replaces the last step (code generation and runtime). Fusion
decisions, buffer layouts and the memory plan are identical on both targets.

---

## One layer through the whole compiler

This is real `kiln-opt` output for `nn.Linear(16, 32)` on a `[4, 16]` input
(from `tests/lit/freeze.kir`).

**1. After import.** ATen lowers `linear` to `addmm(bias, x, weight.t())`, and
the frontend splits that into `transpose`, `matmul` and `add`. The weight and
bias are graph arguments tagged `param`:

```
graph @linear(%arg0: f32[32, 16] {param = "fc.weight"}, %arg1: f32[32] {param = "fc.bias"},
              %arg2: f32[4, 16]) -> (f32[4, 32]) {
  %0 = transpose %arg0 {perm = [1, 0]} : f32[16, 32]
  %1 = matmul %arg2, %0 : f32[4, 32]
  %2 = add %1, %arg1 : f32[4, 32]
  return %2
}
```

**2. After freeze, const-fold, DCE, fusion, buffer-assign and memory-plan:**

```
graph @linear(%arg0: f32[4, 16]) -> (f32[4, 32]) attributes {arena_bytes = 0} {
  kernel @k0 matmul
  %0 = const {name = "fc.bias"} : f32[32]
  %1 = const {name = "fc.weight.transpose"} : f32[16, 32]
  %2 = matmul %arg0, %1 {kernel = "k0"} : f32[4, 32]
  %3 = add %2, %0 {kernel = "k0", buffer = "output"} : f32[4, 32]
  return %3
}
```

What happened, one pass at a time:

- **Freeze** turned the weight and bias from runtime inputs into constants.
- **Constant folding** ran the transpose once, at compile time, and stored the
  weight in the `[in, out]` layout the matmul wants (`fc.weight.transpose`).
  That's weight prepacking.
- **Fusion** put the bias `add` into the matmul's kernel as an *epilogue*, so
  the bias is added while the output tile is still in registers.
- **Buffer assignment** found that `%3` is the graph output, so it is written
  straight into the caller's buffer.
- **Memory planning** needed no arena at all (`arena_bytes = 0`): nothing
  crosses a kernel boundary.

On CPU, `k0` becomes two `linalg.generic` ops (the matmul contraction and the bias add),
tiled and vectorized by a transform script. On GPU it becomes one Triton
kernel that calls `tl.dot` and adds the bias before its single store.

You can reproduce this yourself:

```bash
kiln-opt tests/lit/freeze.kir --passes=freeze,const-fold,dce,fusion,buffer-assign,memory-plan
kiln-opt tests/lit/freeze.kir --pipeline=default --emit=mlir      # the MLIR Kiln generates
```

---

## File by file

Read in this order. Line counts are approximate and tell you where the weight
of the project is.

### Entry points

#### `kiln/__init__.py`
The public surface: `kiln.compile`, `kiln.compile_graph`, `kiln.Options`,
`kiln.CompiledModule` and `kiln.backend`. Nothing else is meant to be
imported by users.

#### `kiln/api.py` (~280 lines)
The compile pipeline, start to finish.

- **`Options`** has one flag per optimization (`freeze`, `fusion`,
  `memory_plan`, `tiling`, `autotune`) so every optimization can be switched
  off for ablations, and `Options.none()` gives the all-off baseline. GPU
  flags: `device="cuda"`, `allow_tf32`, `cuda_graphs`.
- **`build_passes`** builds the pass list from the options. This is the
  single place that says which pass runs in which order.
- **`compile_graph`** runs the passes, then either emits MLIR, lowers it and
  JITs it (CPU), or calls `_compile_graph_gpu`, which runs the same passes
  without `Schedule` and generates Triton.
- **`compile`** is the user-facing function: `torch.export` the model, run
  ATen decompositions, import, compile, and wrap the result in a
  `CompiledModule` that holds the parameters and takes tensors in and out.
- **`Report`** records per-pass statistics and timings, the kernel list,
  arena size against the no-reuse size and the lower bound, and autotune
  results. `print(compiled.report)` shows all of it.

*Be ready for:* "Walk me through what happens when I call `kiln.compile`."

#### `kiln/backend.py`
A `torch.compile` backend (`torch.compile(model, backend=kiln.backend)`).
TorchDynamo doesn't say which inputs are weights, so this path can't freeze.
Unsupported graphs fall back to eager PyTorch unless `KILN_STRICT=1`.
`kiln.compile` is the main path; this file shows Kiln plugs into PyTorch's
compiler stack too.

#### `kiln/target.py`
Describes the CPU being compiled for: f32 lanes per vector register, number of
vector registers, and L1 and L2 sizes, read from `/proc/cpuinfo` and sysfs.
These four numbers are all the CPU cost model needs. `KILN_TARGET=avx2`
(or `avx512`, `neon`, `generic`) overrides it for experiments.

---

### The IR: `kiln/ir/`

#### `kiln/ir/core.py` (~240 lines)
Kiln's graph IR, the equivalent of XLA's HLO.

- **`TensorType`** is a static shape of f32. **`Value`** is an SSA value:
  a graph argument or a node's result. **`Node`** is one op with inputs and
  attributes. **`Kernel`** is a fusion group. **`Graph`** owns all of them,
  plus the constants' data.
- Design choices worth saying out loud:
  - Every node produces exactly **one** value, so multi-result ATen ops are
    split by the frontend.
  - Values have **no layout**. Strides and aliasing are decided late, by
    buffer assignment, so views can be resolved without copies.
  - Passes **annotate** the IR with attributes (`kernel`, `buffer`, `offset`)
    instead of converting it to new data structures. So there is one printer
    and one parser, and any stage can be dumped and FileCheck-tested.
- Utilities: `users()` (def-use chains), `replace_all_uses`, `remove_dead`
  (DCE), `toposort`, and `verify()`, which checks SSA dominance and
  re-infers every type.

#### `kiln/ir/ops.py` (~190 lines)
The op set. Each op has a *kind* (elementwise, reduction, matmul, view,
const), a type-inference function, and a NumPy reference implementation.

- The reference implementations serve two purposes. The constant folder
  evaluates them at compile time, and **`interpret(graph)`** runs a whole graph
  in NumPy, so tests can check that any pass preserves meaning without
  compiling anything.
- `broadcast_shapes` implements NumPy broadcasting for elementwise ops.

*Be ready for:* "Why a small core op set?" Composite ops (LayerNorm, softmax,
addmm) are decomposed into a few primitives. Every pass then only has to
understand those primitives, and fusion puts the pieces back together.

#### `kiln/ir/printer.py` and `kiln/ir/parser.py`
The textual format (`.kir` files) and its parser. It round-trips: print, parse
and print again gives the same text. That's what lets `kiln-opt` run
individual passes on hand-written IR. Constants written without data get
deterministic pseudo-random data seeded by name, so test IR can be executed
without weight files.

---

### The frontend: `kiln/frontend/`

#### `kiln/frontend/fx_import.py` (~290 lines)
Converts an ATen-level `torch.fx` graph into Kiln IR.

- `import_exported` handles `torch.export` programs. The graph signature says
  which placeholders are parameters, so they get tagged `param` for the
  freeze pass.
- `import_fx` handles graphs from `torch.compile`, where every placeholder
  is a runtime input.
- It decomposes composite ops: `addmm` becomes `matmul` + `add`; `layer_norm`
  becomes mean, subtract, variance, rsqrt, scale and shift; `softmax` becomes
  max, subtract, exp, sum and divide.
- Anything unsupported raises `Unsupported`, which the `torch.compile`
  backend catches to fall back to eager.

---

### Graph passes: `kiln/passes/`

#### `kiln/passes/manager.py`
`Pass` (subclasses set `name` and implement `run(graph)`, optionally returning
statistics) and `PassManager`, which runs a list of passes, times each one and
can print the IR after every pass (`print_ir=True`). This is the same idea as
MLIR's pass manager and `-print-ir-after-all`.

#### `kiln/passes/simplify.py`
The classic cleanups:

- **`Canonicalize`**: local rewrites iterated to a fixed point. Examples:
  remove identity transposes and no-op reshapes, `x * 1 → x`, `x + 0 → x`.
- **`CSE`**: common subexpression elimination, keyed on op, operands and
  attributes.
- **`DCE`**: removes nodes nothing uses.

They run right after import, and again after freezing, because constant
folding exposes more to clean up.

#### `kiln/passes/freeze.py`
- **`FreezeParameters`** turns `param`-tagged graph arguments into constants.
  For inference, weights never change between calls.
- **`ConstantFold`** evaluates any node whose inputs are all constants, using
  the NumPy reference implementations. It **refuses folds that would grow the
  data** (for example, materializing a broadcast), because folding should
  never increase model size.

*Be ready for:* "What does freezing buy you if it does nothing by itself?"
It enables folding. The motivating case is `nn.Linear`'s `weight.t()`, which
becomes a weight already stored in the layout the matmul kernel wants.

#### `kiln/passes/partition.py`
`OneOpPerKernel`: the trivial partition, one op per kernel. It is the
`fusion=False` baseline that every fusion speedup is measured against.

#### `kiln/passes/fusion.py` (~230 lines). **Know this one well.**
Greedily groups nodes into kernels, in topological order. There are three
kinds of kernel:

- **matmul + epilogue**: a matmul followed by elementwise ops on its output
  (bias, GELU, residual add, attention scale and mask), applied to the output
  tile while it's in registers.
- **row (reduction) kernel**: reductions over the innermost dimension plus
  the elementwise ops around them on the same rows. This is LayerNorm or
  softmax as **one** kernel.
- **elementwise**: chains of elementwise ops become one loop.

The **legality rules** (the most likely interview topic in this file). A node
may join a kernel only if:

1. it's compatible with the kernel's kind (`_can_join`);
2. it's reached by a direct edge, because **views end kernels** (views are
   resolved later as strided reads);
3. joining **doesn't create a cycle between kernels**. If the node also
   depends on kernel H, and H depends on the target kernel K, then K would be
   both producer and consumer of H. `_reaches` checks for this path. XLA and
   TVM use the same check;
4. the node doesn't **read its own kernel's result through a view**. A
   transpose of a value in kernel K needs element `[j, i]` while computing
   `[i, j]`, a different iteration's result. The random-graph fuzzer found this
   bug.

Also, a matmul takes an epilogue only if it has a **single user**; otherwise
its full output must be materialized anyway.

*Be ready for:* "Why does fusion make things faster?" Memory-bound ops do a
few FLOPs per element, so their cost is memory traffic. Unfused, every op
writes its result to memory and the next op reads it back. Fused, each element
is read once and written once.

#### `kiln/passes/buffers.py` (~270 lines)
Buffer assignment: decides **where every value lives**.

- Graph inputs, outputs and constants live in caller-provided buffers.
- Values produced in one kernel and read by another (**kernel outputs**) get
  a contiguous buffer.
- Values internal to a kernel get **no buffer**. That's the point of fusion.
- View ops get no buffer either. They become a **`Layout`** (base buffer,
  offset, strides), so a transpose is just different strides and costs
  nothing. `reshape_strides` works out whether a reshape of a strided view
  can still be expressed as strides.
- When a view can't be expressed as strides (such as reshaping a transposed
  tensor), or a matmul needs its right operand contiguous along N, it inserts
  an explicit **`copy`** kernel. This is the only place in Kiln that decides
  layout.
- Buffers are 64-byte aligned (one cache line).

#### `kiln/passes/memory_plan.py` (~110 lines). **Know this one well.**
Packs every intermediate buffer into **one arena**, allocated once at load
time, so a call allocates nothing.

1. **Live ranges**: for each buffer, `[first, last]` = the kernel that
   writes it to the last kernel that reads it, *including reads through
   views*. Ranges are closed: a kernel reading A and writing B needs both.
2. **Greedy-by-size** (TensorFlow Lite's arena planner): sort buffers
   largest first, and place each at the lowest offset that doesn't collide
   with an already-placed buffer whose live range overlaps.
3. **Lower bound**: the most bytes live during any single kernel
   (`peak_live_bytes`). No plan can beat it, so the report shows how close
   the plan got.

`MemoryPlan(reuse=False)` gives every buffer its own slot, the ablation
baseline.

*Be ready for:* "Why not solve it optimally?" It's 2-D bin packing (time ×
address), which is NP-hard. Greedy-by-size is near the lower bound in
practice, and Kiln reports the gap.

#### `kiln/passes/schedule.py` (~165 lines)
The CPU **cost model**. It doesn't change the IR. It annotates each kernel
with a schedule (`strategy`, tile sizes, vector width) that codegen turns into
a transform script. *Decision separate from mechanism* is the split used by
Halide, TVM and IREE.

- **Matmul (BLIS/GotoBLAS decomposition):**
  - **mr × nr register tile**: accumulators stay in vector registers for the
    whole k loop. `max_mr` takes the largest mr that fits the register file
    (accumulators + B vectors + one broadcast). nr is 1 or 2 vectors.
  - **kc**: A and B slivers fit in half of L1.
  - **nc**: the kc × nc panel of B fits in half of L2. **mc** is capped at 128.
  - Every tile divides its loop extent, so there are no remainder loops and
    no masking.
- **Row kernels** use one-vector chunks for short rows (≤ 8 vectors) and two
  vectors beyond. **Elementwise** kernels use chunks of 4 vectors.

The module docstring contains the measured tables that justify those choices.
Quote them if asked how the numbers were chosen.

#### `kiln/passes/autotune.py` (~260 lines)
The CPU **autotuner**. It measures tile sizes on the real CPU instead of
predicting them.

- Two-phase search: (1) register tiles mr × nr, (2) cache tiles kc, nc and
  mc around the best register tile.
- Each candidate is compiled as a standalone matmul, checked for correctness,
  and timed.
- **Noise handling**: the top few candidates and the cost model's choice are
  re-timed in an **interleaved** final round (A, B, C, A, B, C …), and the
  cost model's tiles are kept unless a candidate wins by **≥ 3%**
  (`MIN_GAIN`). So tuning never makes things worse.
- Results are cached in `~/.cache/kiln/autotune.json`, keyed by target and
  shape. Small matmuls (< `MIN_WORK`) skip tuning.

*Be ready for:* "Why tune if you have a cost model?" The model knows vector
width, register count and cache sizes. It doesn't know FMA port count,
latencies, prefetchers or LLVM's register allocation. It matched MKL on
AVX-512 but was 15–25% behind on AVX2.

---

### CPU code generation: `kiln/codegen/` (the MLIR part)

#### `kiln/codegen/emitter.py` (~500 lines)
Writes the **MLIR module** for the optimized graph.

- One `func.func` per kernel, taking **memrefs** and writing results directly
  into them (destination-passing style). Inside, values are **tensors**
  (linalg on tensors), which is the form MLIR's tiling and fusion transforms
  operate on.
- A `@forward` function carves strided views out of the arena and argument
  buffers and calls each kernel in order.
- Every op is emitted as a `linalg.generic` (plus `linalg.fill` to zero
  accumulators). A matmul is a generic with loops `(m, n, k)`, `k` marked
  `reduction`; a row sum is a generic whose last loop is `reduction`; an
  elementwise cluster is one generic over the output shape. Using generics
  everywhere means one code path, and MLIR's tiling and vectorization still
  recognize the matmul as a contraction.
- Unit dimensions are squeezed away, so broadcasting appears only in
  indexing maps and every map is a projected permutation, the form linalg
  tiling handles best.

*Be ready to sketch:* a `linalg.generic` with its indexing maps and iterator
types for a matmul (`(m, k)`, `(k, n)` → `(m, n)`; `k` is a reduction), a bias
add, or a row sum.

#### `kiln/codegen/scalar.py` (~150 lines)
Builds the scalar bodies of elementwise ops (`arith` and `math` ops).
`exp`, `tanh` and `erf` are expanded into **polynomial approximations**
(Cody–Waite range reduction plus a Cephes polynomial for exp, and
Abramowitz–Stegun for erf). `math.exp` would otherwise become a libm call per
element, which LLVM can't vectorize. This made fused softmax about 4x faster.

#### `kiln/codegen/transforms.py` (~360 lines)
Builds a **transform-dialect script** for each kernel. Kiln decides; MLIR's
transform interpreter rewrites.

- **Loop fusion** for multi-op kernels: `tile_using_for` on the kernel's root
  with tile size 1 on outer dimensions (a loop over rows), then
  `fuse_into_containing_op` for each producer. Each intermediate shrinks to
  one row on the stack instead of a full tensor on the heap.
  `fusion_plan` keeps ops whose results leave the kernel *outside* the loop.
  Fusing them in caused a segfault during development.
- **With a schedule**:
  - `tiled` (matmul): mc × nc cache tiles, a kc split, mr × nr register tiles,
    `vectorize`, `lower_contraction` to outer-product FMAs, and hoisting the
    accumulator out of the k loop.
  - `rows` (LayerNorm, softmax): loop fusion plus vector-width chunks;
    reductions use `tile_reduction_using_for` (partial sums in a vector, one
    horizontal reduce).
  - `vectorize` (elementwise and copy): vector-width chunks.

#### `kiln/codegen/lowering.py` (~140 lines)
Applies each kernel's transform script, then runs MLIR's pass pipelines to
reach the LLVM dialect: one-shot bufferization, buffer hoisting and
promotion of small buffers to the stack, then lowering of linalg, vector,
scf, affine and memref to LLVM.

- **Safety net**: if MLIR rejects a kernel's script, that kernel is restored
  from a snapshot and compiled as plain loops with a warning. A scheduling
  bug costs speed, never correctness. Tests turn that warning into an error.
- `copies_to_loops` rewrites leftover `memref.copy` into `linalg.copy` so
  no runtime library call is needed.

---

### CPU runtime: `kiln/runtime/`

#### `kiln/runtime/executor.py` (~140 lines)
JIT-compiles the LLVM-dialect module with **MLIR's `ExecutionEngine`** and
calls it through **ctypes**. Everything static (memref descriptors for
constants and the 64-byte-aligned arena, and the packed argument array) is
built once. A call only patches input and output data pointers, so overhead
is a few microseconds. `profile()` times each kernel separately.

---

### GPU backend: `kiln/gpu/` (Triton)

#### `kiln/gpu/codegen.py` (~610 lines). **Know this one well.**
Writes **Triton source code**, one Triton kernel per Kiln kernel. Shapes are
static, so every size, stride and arena offset is a constant in the source.
Views are strided loads, never copies.

Four kernel kinds:

- **elementwise**: a flat 1-D loop over the output, 1024 elements per program,
  with every intermediate in registers.
- **row** (LayerNorm, softmax): one program per row. Rows of up to 4096 floats
  are loaded once and stay in registers. Longer rows are processed in chunks,
  one pass per level of reduction, **recomputing** elementwise values instead
  of storing them, so memory use stays constant for any row length.
- **matmul**: `tl.dot` on BM × BN tiles with a loop over K in BK steps, and
  the fused epilogue applied to the accumulator before the single store.
  `input_precision` is `"ieee"` (exact fp32) or `"tf32"` (tensor cores).
- **reduction**: any other reduction (over a leading dimension).

Other pieces:

- `PRELUDE`: device helpers for tanh, sigmoid, erf and GELU.
- `_KernelWriter.address()`: builds pointer expressions from base, offset and
  strides. It adds a `* 0` term so pointers keep the full tile shape even for
  size-1 dimensions. That fixed a store-shape bug at M=1.
- **`default_matmul_config` is SM-aware**: it picks the largest tile that
  still launches at least one program per SM (`sm_count()` is 170 on a
  5090). Fixed 128×128 tiles gave only 18 programs on 170 SMs, which ran 5x
  slower.

*Be ready for:* "How do you choose GPU tile sizes?" Too large and SMs sit
idle on small matrices; too small and you waste tensor-core throughput and
reload operands. The default uses SM count, and the autotuner measures.

#### `kiln/gpu/runtime.py` (~210 lines)
- Writes the generated source to `~/.cache/kiln/triton/kiln_<hash>.py` and
  imports it (Triton reads kernel source from a file).
- **`GPUExecutable`**:
  - allocates the **arena on the device once**, at the memory plan's offsets;
  - copies weights to the device once;
  - checks free GPU memory first, raising a clear `MemoryError` with sizes
    instead of a CUDA error;
  - **captures the whole launch sequence as a CUDA graph** and replays it
    with one call. That removes the Python and Triton launch overhead per
    kernel, which can exceed the kernel time itself. This is what
    `torch.compile(mode="reduce-overhead")` does.
- **`kernel_seconds`**: benchmarks GPU time by capturing many calls in one
  CUDA graph and timing the replay with CUDA events, excluding CPU launch
  overhead.
- Without a GPU, `TRITON_INTERPRET=1` runs the same kernels on the CPU,
  which is how the GPU tests pass on machines without a GPU.

#### `kiln/gpu/autotune.py` (~190 lines)
The GPU autotuner, with the same design as the CPU one:

- Knobs: BM, BN, BK, `num_warps`, `num_stages` (software pipelining depth).
- Candidates are filtered by problem size and **shared memory**:
  `(BM + BN) × BK × 4 bytes × stages ≤ 99 KiB`.
- Correctness check against a float64 reference, with error **relative to
  the output's scale**. A per-element tolerance rejected every correct TF32
  kernel, which caused the `--tf32` crash during development.
- Interleaved final round, a 3% `MIN_GAIN` over the default, and a fallback
  to the default config if every candidate fails.
- Cache: `~/.cache/kiln/gpu_autotune.json`, keyed by GPU name, precision and
  shape.

---

### Tools: `kiln/tools/`

#### `kiln/tools/opt.py`: `kiln-opt`
Kiln's version of `mlir-opt`: reads `.kir` IR, runs the passes you name
(`--passes=freeze,const-fold,fusion`) or a pipeline (`--pipeline=default`), and
prints the IR. With `--emit` it prints transform scripts or MLIR instead, and
`--print-ir-after-all` shows every step. The lit tests are written against it.

---

### Tests: `tests/`

171 tests in total, plus 7 FileCheck files. Run everything with `pytest -q`.

| File | What it checks |
|---|---|
| `conftest.py` | Sets `TRITON_INTERPRET=1` when there's no GPU (before Triton is imported); seeds RNGs |
| `test_ir.py` | Print/parse round trip, the verifier, the NumPy interpreter |
| `test_freeze.py` | Weight transpose folded; folding never grows constants; semantics preserved |
| `test_compile.py` | End to end: every op and model matches eager PyTorch; views without copies; the `torch.compile` backend and its fallback |
| `test_fusion.py` | LayerNorm and softmax are one kernel each; matmul epilogues; the legality rules (no cycles, views end kernels, two users means no epilogue) |
| `test_memory_plan.py` | Plan safety (live buffers never overlap); **random-graph fuzzer** comparing against the interpreter; identical results with and without reuse |
| `test_schedule.py` | Tiles are legal and divide extents; real GPT-2 sizes compile with **warnings as errors**, so a rejected schedule fails instead of silently running slowly |
| `test_autotune.py` | Every candidate legal; tuning never worse than the model; caching; a corrupt cache is ignored |
| `test_gpu.py` | Every GPU kernel kind matches eager; long rows; views are strided loads; TF32 option; random graphs; memory error; autotune; CUDA-graph replay (needs a real GPU) |
| `test_lit.py` | Runs the FileCheck suite from pytest |
| `models/transformer.py` | The GPT-2-style pre-LayerNorm transformer block used by tests and benchmarks |
| `lit/*.kir` | FileCheck tests, one per pass: `freeze`, `const_fold`, `canonicalize`, `fusion`, `buffer_assign`, `memory_plan`, `schedule` |

*Be ready for:* "How did you test a compiler?" The answer has four layers:

- unit tests per pass against the NumPy interpreter;
- FileCheck tests on the IR, as LLVM and MLIR do;
- end-to-end comparison with eager PyTorch;
- **fuzzers** (random graphs, random matmul shapes) that found real bugs: the
  fusion self-view cycle, and a missing `memrefCopy` for a copy of a strided
  input.

---

### Benchmarks: `benchmarks/`

| File | Measures |
|---|---|
| `bench_mlp.py` | A 3-layer MLP, eager vs Kiln, with each optimization switched off in turn |
| `bench_fusion.py` | LayerNorm, softmax and bias + GELU: unfused vs fused vs eager (CPU) |
| `bench_memory.py` | Arena size with and without reuse, against the lower bound (compile only, fast) |
| `bench_matmul.py` | CPU matmul GFLOP/s: scalar vs tiled vs autotuned vs eager, at GPT-2 shapes |
| `bench_block.py` | GPT-2 block ablation on CPU: add optimizations one at a time, then remove one at a time |
| `bench_gpu.py` | GPU: matmul TFLOP/s vs cuBLAS; fused ops vs eager and Inductor; GPT-2 block vs eager and `torch.compile` at 128–2048 tokens. `--tf32` and `--smoke` flags |

`examples/mlp.py` is the smallest end-to-end demo (`--ir` prints the IR after
every pass, `--mlir` prints the generated MLIR). It's good for a live demo in
an interview.

---

## Numbers to know

Measured on an RTX 5090 shared with other jobs, so treat them as approximate.

| Result | fp32 | TF32 |
|---|---|---|
| GPT-2 block vs eager PyTorch (128–2048 tokens) | 1.34–1.86x faster | 1.17–1.57x faster |
| GPT-2 block vs `torch.compile` reduce-overhead | faster at all 4 lengths | equal or faster at all 4 lengths |
| Best matmul, 128×2304×768 (attention QKV) | 36.4 TFLOP/s | 49.3 TFLOP/s (cuBLAS 42.1) |
| Fused bias + GELU vs eager | 2.3x | 2.1x |
| Fused LayerNorm vs eager | 1.7x | 1.7x |
| SM-aware default tiles | 6.6 → 34.9 TFLOP/s (5x) | |
| Single-token decode vs cuBLAS | 0.5x (known weakness) | 0.84x |

GPU memory use is the same as eager PyTorch. Don't claim a GPU memory saving.

---

## Interview questions

**What's the difference between fusion and loop fusion here?**
`passes/fusion.py` decides which ops share a kernel. That removes the kernel
boundary, but inside the kernel each op could still produce a full-size
temporary. `codegen/transforms.py` interleaves their loops, so each
intermediate is one row on the stack. You need both.

**Why have your own IR instead of going straight to MLIR?**
Graph-level decisions (fusion, layout, memory planning) are much easier on a
small dataflow graph with about two dozen op types than on linalg. It's also what lets one
optimizer feed both MLIR and Triton. XLA (HLO) and Inductor work the same way.

**Why does the decode case lose to cuBLAS?**
With M=1, a matmul is really a matrix-vector product, which is memory-bound.
`tl.dot` needs at least a 16-row tile, so 15 of 16 rows are masked-off work.
cuBLAS switches to a GEMV kernel. The fix is a GEMV path in
`gpu/codegen.py` that streams the weight once.

**Why CUDA graphs?**
A GPT-2-sized kernel takes a few microseconds, which is less than launching
it from Python. Shapes are static and every buffer sits at a fixed arena
address, so the whole launch sequence can be recorded once and replayed with
one call.

**What would you do next?**
- a GEMV path for decode;
- fused attention (FlashAttention-style, so the T×T score matrix never hits
  memory; this is also the GPU memory win);
- fp16/bf16;
- dynamic shapes;
- multithreading on CPU.

**What was the hardest bug?**
Good candidates:
- the fusion self-view cycle, found by the fuzzer;
- the TF32 autotuner rejecting every correct kernel because of a per-element
  tolerance;
- 128×128 tiles leaving most of 170 SMs idle.

For each, tell it as: symptom, diagnosis, fix, and the test that now guards it.

---

## Glossary

| Term | Meaning |
|---|---|
| SSA | Static single assignment: every value is defined exactly once |
| Freezing | Treating weights as compile-time constants (inference only) |
| Epilogue | Elementwise ops fused onto a matmul's output tile |
| Row kernel | A kernel that reduces over the last dimension (LayerNorm, softmax) |
| Arena | One preallocated block holding every intermediate buffer |
| Live range | First and last kernel that need a buffer's contents |
| Destination-passing style | Kernels write into buffers passed in by the caller |
| BLIS decomposition | Nested cache and register tiling of a matmul |
| Transform dialect | MLIR scripts that say *how* to tile, fuse and vectorize a payload |
| Bufferization | MLIR turning tensors (values) into memrefs (memory) |
| SM | Streaming multiprocessor, the GPU's unit of parallel execution (170 on a 5090) |
| TF32 | Tensor-core format: fp32 range, 10-bit mantissa |
| CUDA graph | A recorded sequence of GPU launches replayed with one call |