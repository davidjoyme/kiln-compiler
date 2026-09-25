# Kiln

An MLIR-based compiler for PyTorch models. Kiln plugs into `torch.compile` as a
custom backend, translates the captured ATen graph into MLIR, lowers it to LLVM,
and JIT-compiles it to native code.

```
PyTorch model
   │  torch.compile + aot_autograd
   ▼
ATen FX graph ──► MLIR (tensor / linalg / arith) ──► bufferize ──► loops ──► LLVM ──► native code
   kiln/frontend           kiln/passes                                   kiln/runtime
```

## Status

Walking skeleton: end-to-end compilation works for float32, static-shape
inference graphs built from `linear`, `matmul`, `relu`, and broadcasting
`add` / `sub` / `mul`. Unsupported graphs fall back to eager PyTorch.

No optimisations yet, so Kiln is currently slower than eager. That's the
baseline that fusion, tiling, and vectorisation will improve on.

## Setup

```bash
conda create -n kiln -c conda-forge python=3.11 cmake ninja c-compiler cxx-compiler lld
conda activate kiln
./scripts/build_llvm.sh          # builds LLVM/MLIR at the commit in llvm-version.txt
source scripts/env.sh            # puts the MLIR Python bindings on PYTHONPATH
pip install -e ".[dev]"
```

## Usage

```python
import torch, kiln

compiled = torch.compile(model, backend=kiln.backend, dynamic=False)
with torch.no_grad():
    y = compiled(x)
```

```bash
python examples/mlp.py                   # compile an MLP and check it against eager
KILN_DUMP_IR=1 python examples/mlp.py    # print the IR after every pipeline stage
pytest                                   # correctness tests
python benchmarks/bench_mlp.py           # eager vs Kiln timing
```

## Roadmap

- [x] End-to-end pipeline: FX → MLIR → LLVM → JIT
- [ ] Elementwise fusion pass
- [ ] Tiling and vectorisation of matmuls
- [ ] Transformer block ops (softmax, layernorm, GELU, batched matmul)
- [ ] GPT-2 small end to end
- [ ] GPU code generation