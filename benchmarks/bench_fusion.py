# File: benchmarks/bench_fusion.py
"""Fusion on memory-bound ops: Kiln unfused vs fused vs eager PyTorch.

LayerNorm, softmax and bias + GELU do a few FLOPs per element, so their cost
is memory traffic: how many times each element goes to and from memory.
Unfused, every op in the chain writes its full result and the next op reads
it back. Fused, each element is read once and written once.

Shapes are GPT-2 small (d_model 768, 12 heads, 128 tokens). Kiln kernels
are still scalar loops at this stage; vectorisation comes in a later stage.

    python benchmarks/bench_fusion.py
"""

import time

import torch
import torch.nn.functional as F

import kiln


def bench(fn, *xs, iters=50):
    with torch.no_grad():
        for _ in range(3):
            fn(*xs)
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(iters):
                fn(*xs)
            best = min(best, (time.perf_counter() - start) / iters)
    return best * 1e3


def main():
    torch.set_num_threads(1)  # Kiln is single-threaded: compare like with like
    torch.manual_seed(0)

    bias = torch.randn(3072)
    workloads = [
        ("LayerNorm [128, 768]", torch.nn.LayerNorm(768).eval(), (torch.randn(128, 768),)),
        ("softmax [12, 128, 128]", lambda x: F.softmax(x, dim=-1), (torch.randn(12, 128, 128),)),
        ("bias + GELU [128, 3072]", lambda x: F.gelu(x + bias, approximate="tanh"),
         (torch.randn(128, 3072),)),
    ]

    print(f"{'workload':<26}{'kernels':>9}{'intermediate MB':>17}{'unfused ms':>12}"
          f"{'fused ms':>10}{'speedup':>9}{'eager ms':>10}")
    for name, fn, xs in workloads:
        results = {}
        for fusion in (False, True):
            compiled = kiln.compile(fn, xs, fusion=fusion)
            with torch.no_grad():
                torch.testing.assert_close(compiled(*xs), fn(*xs), rtol=1e-4, atol=1e-5)
            results[fusion] = (compiled, bench(compiled, *xs))
        (unfused, t_unfused), (fused, t_fused) = results[False], results[True]
        kernels = f"{len(unfused.report.kernels)}->{len(fused.report.kernels)}"
        # Bytes materialised between kernels (before memory reuse): what fusion avoids writing.
        mb = f"{unfused.report.naive_bytes / 2**20:.2f}->{fused.report.naive_bytes / 2**20:.2f}"
        print(f"{name:<26}{kernels:>9}{mb:>17}{t_unfused:>12.3f}{t_fused:>10.3f}"
              f"{t_unfused / t_fused:>8.2f}x{bench(fn, *xs):>10.3f}")


if __name__ == "__main__":
    main()