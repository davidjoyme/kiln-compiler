# File: benchmarks/bench_matmul.py
"""Matmul throughput (GFLOP/s) at GPT-2 small shapes: Kiln scalar / tiled / autotuned vs eager.

Each workload is an nn.Linear (matmul + bias epilogue) with 128 tokens, the
four projections of a GPT-2 small block plus a single-token decode step.
Everything runs on one thread: Kiln doesn't parallelise yet, so this
compares code quality, not core count. "tiled" uses the cost model's tiles,
"tuned" the autotuner's (the first run tunes each shape, ~10 s each; later
runs read the cache).

    python benchmarks/bench_matmul.py            # scalar column only for small shapes
    python benchmarks/bench_matmul.py --all      # scalar for every shape (slow)
"""

import sys
import time

import torch

import kiln
from kiln.target import host

SHAPES = [
    ("attention qkv", 128, 2304, 768),
    ("attention out", 128, 768, 768),
    ("MLP up", 128, 3072, 768),
    ("MLP down", 128, 768, 3072),
    ("decode, 1 token", 1, 3072, 768),
]


def seconds_per_call(fn, x, min_time=0.3):
    """Best-of-5 average, with enough calls per round to run ~min_time / 5."""
    with torch.no_grad():
        fn(x)
        start, calls = time.perf_counter(), 0
        while time.perf_counter() - start < min_time / 5:
            fn(x)
            calls += 1
        best = float("inf")
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(calls):
                fn(x)
            best = min(best, (time.perf_counter() - start) / calls)
    return best


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    run_all = "--all" in sys.argv
    print(f"target: {host()}  (1 thread)")
    print(f"{'workload':<18}{'M x N x K':>16}{'kiln scalar':>13}{'kiln tiled':>12}{'kiln tuned':>12}"
          f"{'eager':>9}{'tuned/eager':>13}   (GFLOP/s)")
    for name, m, n, k in SHAPES:
        linear = torch.nn.Linear(k, n).eval()
        x = torch.randn(m, k)
        flops = 2 * m * n * k
        results = {}
        for config, flags in (("scalar", {"tiling": False}), ("tiled", {}), ("tuned", {"autotune": True})):
            if config == "scalar" and m * n * k > 128 * 768 * 768 and not run_all:
                continue
            compiled = kiln.compile(linear, (x,), **flags)
            with torch.no_grad():
                torch.testing.assert_close(compiled(x), linear(x), rtol=1e-3, atol=1e-4)
            results[config] = flops / seconds_per_call(compiled, x) / 1e9
        eager = flops / seconds_per_call(linear, x) / 1e9
        scalar = results.get("scalar")
        print(f"{name:<18}{f'{m}x{n}x{k}':>16}"
              f"{(f'{scalar:.1f}' if scalar else '-'):>13}{results['tiled']:>12.1f}{results['tuned']:>12.1f}"
              f"{eager:>9.1f}{results['tuned'] / eager:>12.2f}x", flush=True)


if __name__ == "__main__":
    main()