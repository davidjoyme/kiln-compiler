# File: benchmarks/bench_mlp.py
"""Time an MLP in eager PyTorch vs Kiln, with and without freezing.

Stage 1 compiles every kernel to plain scalar loops, so Kiln is much slower
than eager here. That's expected: this is the baseline that fusion, memory
planning and (above all) tiling + vectorisation will improve on.
"""

import argparse
import time

import torch

import kiln


def bench(fn, x, warmup=3, iters=20):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        start = time.perf_counter()
        for _ in range(iters):
            fn(x)
    return (time.perf_counter() - start) / iters * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--iters", type=int, default=20)
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
    for freeze in (True, False):
        compiled = kiln.compile(model, (x,), freeze=freeze)
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-4)
        rows.append((f"kiln freeze={freeze}", compiled))
    for name, fn in rows:
        ms = bench(fn, x, iters=args.iters)
        print(f"  {name:<20} {ms:9.3f} ms  {flops / ms / 1e6:7.2f} GFLOP/s")


if __name__ == "__main__":
    main()