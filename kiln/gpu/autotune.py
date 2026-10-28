# File: kiln/gpu/autotune.py
"""GPU matmul autotuning: pick Triton block sizes, warps and pipeline stages by measurement.

The same design as the CPU tuner (kiln/passes/autotune.py), applied to the
knobs that matter for a Triton matmul:

  BM x BN     the output tile one program computes (and so the grid size:
              too large leaves SMs idle on small matrices, too small wastes
              tensor-core throughput and reloads operands)
  BK          the K step per iteration of the main loop
  num_warps   threads per program (4 or 8 warps of 32)
  num_stages  software pipelining depth: how many K steps of A and B tiles
              are in flight in shared memory at once

Candidates are filtered to fit the GPU's shared memory (fp32 tiles take
(BM + BN) * BK * 4 bytes per stage) and to the problem size, then each is
compiled as a standalone matmul and timed as GPU time only: many calls are
captured in one CUDA graph and its replay timed with CUDA events, because
these kernels take a few microseconds, less than Python needs to launch
them (kiln.gpu.runtime.kernel_seconds). As on the CPU,
the fastest few and the default config are re-timed interleaved, and the
default is kept unless a candidate beats it by MIN_GAIN, so noise from other
work on the GPU can't change the decision. Results are cached by GPU name,
shape and precision in $KILN_CACHE_DIR/gpu_autotune.json.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import torch

from kiln.gpu.codegen import default_matmul_config, next_pow2
from kiln.gpu.runtime import device, interpreting, kernel_seconds

MIN_GAIN = 1.03
FINALISTS = 3
SHARED_MEMORY_BYTES = 99 * 1024      # per block on consumer GPUs (RTX 40/50); data-centre GPUs have more

CANDIDATES = [  # BM, BN, BK, num_warps, num_stages
    (128, 128, 32, 8, 2), (128, 128, 32, 8, 3), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3),
    (64, 64, 32, 4, 4), (128, 128, 16, 8, 4), (64, 64, 64, 4, 2), (32, 64, 32, 4, 4),
    (64, 32, 32, 4, 4), (16, 128, 64, 4, 3), (16, 64, 64, 4, 4), (32, 128, 32, 4, 3),
]


def cache_path() -> Path:
    base = os.environ.get("KILN_CACHE_DIR") or os.path.join(Path.home(), ".cache", "kiln")
    return Path(base) / "gpu_autotune.json"


def device_name():
    return "interpreter" if interpreting() else torch.cuda.get_device_name()


def candidates(m, n, k):
    out = []
    for bm, bn, bk, warps, stages in CANDIDATES:
        if bm > max(16, next_pow2(m)) or bn > max(16, next_pow2(n)) or bk > max(16, next_pow2(k)):
            continue    # a tile much bigger than the problem only adds masked-off work
        if (bm + bn) * bk * 4 * stages > SHARED_MEMORY_BYTES:
            continue
        out.append({"BM": bm, "BN": bn, "BK": bk, "num_warps": warps, "num_stages": stages})
    return out


def _matmul_graph(batch, m, n, k):
    from kiln.ir import Graph, TensorType

    g = Graph("matmul")
    lead = (batch,) if batch > 1 else ()
    a = g.add_arg(TensorType((*lead, m, k)), "a")
    b = g.add_arg(TensorType((*lead, k, n)), "b")
    g.results = [g.add_node("matmul", [a, b])]
    return g


class GPUAutotuner:
    """Chooses Triton matmul configs by measurement, with an on-disk cache.

    Pass ``config_for`` to kiln.gpu.codegen.generate (kiln.compile does this
    with ``device="cuda", autotune=True``).
    """

    def __init__(self, allow_tf32=False, use_cache=True):
        self.allow_tf32 = allow_tf32
        self.use_cache = use_cache
        self.log = []
        self._cache = self._load() if use_cache else {}

    def _load(self):
        try:
            return json.loads(cache_path().read_text())
        except (OSError, ValueError):
            return {}

    def _save(self):
        if not self.use_cache:
            return
        path = cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, indent=1, sort_keys=True))
            tmp.replace(path)
        except OSError as err:
            warnings.warn(f"Kiln: couldn't write the GPU autotune cache ({err})")

    def config_for(self, graph, kernel):
        mm = next(n for n in kernel.nodes if n.op == "matmul")
        a, b = mm.inputs
        batch = 1
        for d in a.shape[:-2]:
            batch *= d
        m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
        key = f"{device_name()}|tf32={self.allow_tf32}|{batch}x{m}x{n}x{k}"
        if key not in self._cache:
            entry = self.tune(batch, m, n, k, default_matmul_config(graph, kernel))
            if entry["candidates"]:             # don't cache a failed search
                self._cache[key] = entry
                self._save()
            source = "tuned"
        else:
            entry, source = self._cache[key], "cache"
        self.log.append({"shape": (batch, m, n, k), "source": source, **entry})
        return dict(entry["config"])

    def _compile(self, batch, m, n, k, config):
        from kiln.api import Options, compile_graph

        options = Options(device="cuda", cuda_graphs=False, allow_tf32=self.allow_tf32,
                          gpu_matmul_config=lambda *_: dict(config))
        exe, _ = compile_graph(_matmul_graph(batch, m, n, k), options)
        return exe

    def tune(self, batch, m, n, k, default):
        start = time.perf_counter()
        dev = device()
        torch.manual_seed(0)
        lead = (batch,) if batch > 1 else ()
        a = torch.randn(*lead, m, k, device=dev)
        b = torch.randn(*lead, k, n, device=dev) / k ** 0.5     # outputs of order 1, like a real layer
        expected = (a.double() @ b.double()).float()            # exact reference, independent of TF32
        # error relative to the output's scale: TF32 rounds inputs to 10 mantissa bits, so its error
        # grows with K and an elementwise atol would reject every correct TF32 kernel near zero
        tol = 2e-2 if self.allow_tf32 else 1e-4
        scale = expected.abs().max().item() or 1.0
        runs, seconds, rejected = {}, {}, []
        configs = [default] + [c for c in candidates(m, n, k) if c != default]
        for cfg in configs:
            key = tuple(sorted(cfg.items()))
            try:
                exe = self._compile(batch, m, n, k, cfg)
                (out,) = exe(a, b)
                err = (out - expected).abs().max().item() / scale
                if not err <= tol:
                    rejected.append(f"{cfg}: relative error {err:.1e}")
                    continue
            except Exception as e:  # e.g. out of shared memory / registers on this GPU
                rejected.append(f"{cfg}: {type(e).__name__}: {str(e).splitlines()[0][:80] if str(e) else ''}")
                continue
            runs[key] = lambda exe=exe: exe(a, b)
            seconds[key] = kernel_seconds(runs[key])
        default_key = tuple(sorted(default.items()))
        if not seconds:                         # nothing usable: keep the default rather than fail
            warnings.warn(f"Kiln: no matmul config passed for {batch}x{m}x{n}x{k}; keeping the default. "
                          + "; ".join(rejected[:3]))
            return {"config": dict(default), "default_config": default, "tflops": None,
                    "default_tflops": None, "candidates": 0,
                    "seconds": round(time.perf_counter() - start, 1)}
        ranked = sorted(seconds, key=seconds.get)[:FINALISTS]
        if default_key in seconds and default_key not in ranked:
            ranked.append(default_key)
        final = {}
        for _ in range(3):                      # interleaved re-timing
            for key in ranked:
                final[key] = min(final.get(key, float("inf")), kernel_seconds(runs[key], rounds=2))
        winner = min(final, key=final.get)
        if default_key in final and final[default_key] <= final[winner] * MIN_GAIN:
            winner = default_key
        flops = 2 * batch * m * n * k
        return {
            "config": dict(winner),
            "default_config": default,
            "tflops": round(flops / final[winner] / 1e12, 2),
            "default_tflops": round(flops / final[default_key] / 1e12, 2) if default_key in final else None,
            "candidates": len(seconds),
            "seconds": round(time.perf_counter() - start, 1),
        }