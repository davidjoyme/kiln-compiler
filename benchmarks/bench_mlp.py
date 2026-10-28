# File: benchmarks/bench_mlp.py
"""Time a 3-layer MLP: eager PyTorch vs Kiln, with each major option off in turn.

    python benchmarks/bench_mlp.py
    python benchmarks/bench_mlp.py --batch 1 --hidden 1024
"""

import argparse
import time

import torch

import kiln


def bench(fn, x, min_time=0.5):
    """Best of 5 rounds, each long enough (~min_time / 5) to average out timer noise."""
    with torch.no_grad():
        for _ in range(10):
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
    return best * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    args = p.parse_args()

    torch.set_num_threads(1)  # Kiln is single-threaded: compare like with like
    torch.manual_seed(0)
    h = args.hidden
    model = torch.nn.Sequential(
        torch.nn.Linear(h, h), torch.nn.ReLU(),
        torch.nn.Linear(h, h), torch.nn.ReLU(),
        torch.nn.Linear(h, h),
    ).eval()
    x = torch.randn(args.batch, h)
    flops = 3 * 2 * args.batch * h * h

    print(f"MLP batch={args.batch} hidden={h} (1 thread)")
    rows = [("eager", model)]
    for name, flags in [("kiln", {}), ("kiln, no freezing", {"freeze": False}),
                        ("kiln, no tiling", {"tiling": False})]:
        compiled = kiln.compile(model, (x,), **flags)
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), model(x), rtol=1e-3, atol=1e-4)
        rows.append((name, compiled))
    for name, fn in rows:
        ms = bench(fn, x)
        print(f"  {name:<20} {ms:9.3f} ms  {flops / ms / 1e6:8.1f} GFLOP/s")


if __name__ == "__main__":
    main()