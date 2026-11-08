<!-- File: README.md -->

# Kiln

An inference compiler for PyTorch with two backends: **CPU through MLIR** and
**NVIDIA GPU through Triton**.

Kiln exports a PyTorch model and optimizes it on its own graph IR with weight
freezing, operator fusion and static memory planning. Each fused kernel then
goes to a backend. On CPU, MLIR's `linalg` dialect is tiled and vectorized by
transform-dialect scripts and lowered to LLVM. On GPU, Kiln generates Triton
kernels and replays them as a CUDA graph. It's the same split as
TorchInductor: one graph optimizer feeding a CPU and a GPU code generator.

```
PyTorch model ──torch.export──► Kiln IR ──► graph passes ──┬──► MLIR linalg ──► LLVM ──► CPU
                                                           └──► Triton kernels ──────► GPU
```

## Results

### GPU

These numbers are from an NVIDIA RTX 5090, with TF32 matmuls in both Kiln and
PyTorch (`python benchmarks/bench_gpu.py --tf32`).

**GPT-2 small transformer block** (d_model 768, 12 heads), end-to-end latency
per call:

| tokens | Kiln | eager PyTorch | torch.compile* | Kiln vs eager |
|---:|---:|---:|---:|---:|
| 128 | **0.064 ms** | 0.109 ms | 0.068 ms | 1.69x |
| 512 | 0.159 ms | 0.183 ms | **0.158 ms** | 1.15x |
| 1024 | **0.316 ms** | 0.428 ms | 0.356 ms | 1.36x |
| 2048 | **0.926 ms** | 1.555 ms | 0.985 ms | 1.68x |

\* `torch.compile(mode="reduce-overhead")`: Inductor's Triton kernels, replayed as a CUDA graph.

**Matmul** (`nn.Linear`, autotuned), in TFLOP/s:

| shape (M × N × K) | Kiln | cuBLAS | ratio |
|---|---:|---:|---:|
| 2048 × 3072 × 768 (MLP up) | **87.6** | 63.6 | 1.38x |
| 128 × 2304 × 768 (attention QKV) | **50.0** | 42.2 | 1.19x |
| 128 × 768 × 768 (attention out) | **25.3** | 14.1 | 1.80x |
| 128 × 3072 × 768 (MLP up) | 44.0 | **54.9** | 0.80x |
| 128 × 768 × 3072 (MLP down) | 33.7 | **35.7** | 0.94x |
| 1 × 3072 × 768 (decode) | 1.0 | **1.2** | 0.81x |

**Fused memory-bound ops**, compared with eager PyTorch:

| op | Kiln vs eager |
|---|---:|
| bias + GELU [2048, 3072] | 2.18x |
| LayerNorm [2048, 768] | 1.60x |
| softmax [12, 512, 512] | 1.07x |

Kiln and Inductor are the same speed on these three ops.


### CPU

These numbers are from an Intel Core Ultra 9 285K (AVX2), single-threaded on
both sides.

**Matmul**, autotuned, compared with eager PyTorch (oneDNN/MKL), in GFLOP/s:

| shape (M × N × K) | Kiln scalar | Kiln tuned | eager PyTorch | ratio |
|---|---:|---:|---:|---:|
| 128 × 2304 × 768 (attention QKV) | – | **153.3** | 147.8 | 1.04x |
| 128 × 768 × 768 (attention out) | 3.6 | **153.1** | 149.1 | 1.03x |
| 128 × 768 × 3072 (MLP down) | – | 144.2 | **146.4** | 0.98x |
| 128 × 3072 × 768 (MLP up) | – | 139.2 | **148.6** | 0.94x |
| 1 × 3072 × 768 (decode) | 1.8 | 26.5 | **28.6** | 0.92x |

Kiln's generated code runs at 0.92–1.04x the vendor library and beats it on
two shapes. It's about 40x faster than Kiln's own scalar loops.

**GPT-2 block ablation** (128 tokens), adding one optimization per row:

| configuration | kernels | arena | latency | speedup |
|---|---:|---:|---:|---:|
| no optimizations | 48 | 45.0 MiB | 813.2 ms | 1x |
| + freezing | 44 | 18.0 MiB | 768.2 ms | 1.1x |
| + fusion | 11 | 6.4 MiB | 810.6 ms | 1.0x |
| + memory planning | 11 | 2.6 MiB | 834.7 ms | 1.0x |
| + tiling and vectorization | 11 | 2.6 MiB | 15.5 ms | 52x |
| + autotuning | 11 | 2.6 MiB | **14.5 ms** | **56x** |
| eager PyTorch (oneDNN) | | | 13.7 ms | 59x |

The full CPU pipeline runs within 5% of eager PyTorch. Until tiling, the
scalar matmuls dominate the run time and hide the other optimizations.
Removing one optimization at a time from the full pipeline shows what each is
worth. The biggest is freezing: without it, the block is 1.61x slower,
because every call has to transpose the `nn.Linear` weights. On CPU, fusion
and memory planning mainly cut memory (45 MiB down to 2.6 MiB of
intermediates) rather than latency.

### Memory planning

Memory planning happens at compile time, so its results don't depend on the
hardware (`benchmarks/bench_memory.py`). For the GPT-2 block at 128 tokens:

- **Fusion plus planning:** after freezing, intermediate memory drops from 18.0 MiB to 2.6 MiB (6.9x).
- **Lower bound:** the plan reaches the theoretical minimum, the most bytes live during any single kernel.

## How it works

| Stage | What it does | Code |
|---|---|---|
| Import | `torch.export`, then decompose ATen ops into a small core op set | `kiln/frontend` |
| IR | SSA graph with a textual form, parser and `kiln-opt` tool | `kiln/ir`, `kiln/tools` |
| Simplify | canonicalize, CSE, DCE | `kiln/passes/simplify.py` |
| Freeze | weights become constants; constant folding prepacks them | `kiln/passes/freeze.py` |
| Fuse | matmul + epilogue, row kernels (LayerNorm, softmax), elementwise chains, with legality checks | `kiln/passes/fusion.py` |
| Buffers | views become strided accesses, with copies only where unavoidable | `kiln/passes/buffers.py` |
| Memory plan | live ranges plus greedy-by-size packing into one arena | `kiln/passes/memory_plan.py` |
| Schedule (CPU) | BLIS-style cost model for tiles and vector widths, or measured autotuning | `kiln/passes/schedule.py`, `autotune.py` |
| CPU codegen | `linalg` plus transform-dialect scripts (tile, fuse, vectorize), lowered to LLVM and JIT-compiled | `kiln/codegen`, `kiln/runtime` |
| GPU codegen | Triton kernels, CUDA-graph replay, GPU autotuning | `kiln/gpu` |

For a file-by-file walkthrough, see **[docs/README.md](docs/README.md)**.

## Installation

```bash
conda create -n kiln -c conda-forge python=3.11 cmake ninja c-compiler cxx-compiler lld
conda activate kiln
./scripts/build_llvm.sh        # builds LLVM/MLIR with Python bindings (pinned in llvm-version.txt)
source scripts/env.sh          # puts the MLIR bindings on PYTHONPATH
pip install -e ".[dev]"
```

The GPU backend needs a CUDA build of PyTorch, which includes Triton.

## Usage

```python
import torch, kiln

compiled = kiln.compile(model, (example_input,))                  # CPU
compiled = kiln.compile(model, (example_input,), device="cuda",   # GPU
                        autotune=True, allow_tf32=True)
y = compiled(x)
print(compiled.report)    # kernels, arena size, per-pass statistics
```

To use Kiln through PyTorch's compiler stack instead:
`torch.compile(model, backend=kiln.backend)`.

To inspect the compiler:

```bash
python examples/mlp.py --ir                              # Kiln IR after every pass
python examples/mlp.py --mlir                            # generated MLIR
kiln-opt tests/lit/fusion.kir --passes=fusion,buffer-assign,memory-plan
```

## Testing

```bash
pytest -q                 # 171 tests: unit, end-to-end, FileCheck, fuzzers
```

The test suite includes:

- **End-to-end checks:** every compiled result is compared with eager PyTorch.
- **FileCheck tests:** each pass is checked on textual IR with `kiln-opt`, LLVM style.
- **Fuzzers:** random graphs compared with a NumPy reference interpreter.
- **GPU tests without a GPU:** they fall back to Triton's CPU interpreter when no GPU is present.

## Benchmarks

| Script | Measures |
|---|---|
| `bench_gpu.py` | GPU: matmul vs cuBLAS, fused ops, GPT-2 block vs eager and torch.compile |
| `bench_block.py` | CPU: GPT-2 block ablation, switching each optimization on and off |
| `bench_matmul.py` | CPU: matmul GFLOP/s, scalar vs tiled vs autotuned vs eager |
| `bench_fusion.py` | CPU: fused vs unfused LayerNorm, softmax, GELU |
| `bench_memory.py` | Arena size with and without reuse, against the lower bound |

## Limitations

- **Precision and shapes:** fp32 and TF32 only, static shapes.
- **No fused attention:** attention is three kernels, not one FlashAttention-style kernel.
- **Slow single-token decode on GPU:** it needs a dedicated matrix-vector kernel.
- **Single-threaded CPU backend.**