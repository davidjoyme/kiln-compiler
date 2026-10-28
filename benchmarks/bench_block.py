# File: benchmarks/bench_block.py
"""Ablation on a GPT-2 small transformer block.

Two tables. The first switches the optimisations on one at a time (each row
adds one to the row above). The second starts from everything on and
switches *one* off: while matmuls are scalar loops they dominate the time and
hide everything else, so "what does this optimisation cost me if I remove
it" is the clearer measure of the others. The block is GPT-2 small (d_model
768, 12 heads, MLP 3072) over 128 tokens, single-threaded.

The first run autotunes the block's matmul shapes (~40 s, cached in
~/.cache/kiln/autotune.json); later runs reuse the results.

    python benchmarks/bench_block.py
    python benchmarks/bench_block.py --tokens 64
"""

import argparse
import os
import sys
import time

import torch

import kiln
from kiln.target import host

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from models.transformer import Block  # noqa: E402

CONFIGS = [
    ("no optimisations", dict(freeze=False, fusion=False, memory_plan=False, tiling=False)),
    ("+ freezing", dict(freeze=True, fusion=False, memory_plan=False, tiling=False)),
    ("+ fusion", dict(freeze=True, fusion=True, memory_plan=False, tiling=False)),
    ("+ memory planning", dict(freeze=True, fusion=True, memory_plan=True, tiling=False)),
    ("+ tiling + vectorisation", dict(freeze=True, fusion=True, memory_plan=True, tiling=True)),
    ("+ autotuning", dict(freeze=True, fusion=True, memory_plan=True, tiling=True, autotune=True)),
]


def ms_per_call(fn, x, min_time=1.0):
    with torch.no_grad():
        fn(x)
        start, calls = time.perf_counter(), 0
        while time.perf_counter() - start < min_time / 3:
            fn(x)
            calls += 1
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(calls):
                fn(x)
            best = min(best, (time.perf_counter() - start) / calls)
    return best * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=128)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model, x = Block(768, 12, args.tokens).eval(), torch.randn(args.tokens, 768)
    with torch.no_grad():
        expected = model(x)

    print(f"GPT-2 small block, {args.tokens} tokens, target: {host()} (1 thread)")
    print(f"{'configuration':<27}{'kernels':>8}{'arena MiB':>11}{'compile s':>11}{'latency ms':>12}{'speedup':>9}")
    baseline = None
    for name, flags in CONFIGS:
        compiled = kiln.compile(model, (x,), **flags)
        torch.testing.assert_close(compiled(x), expected, rtol=1e-3, atol=1e-4)
        ms = ms_per_call(compiled, x)
        baseline = baseline or ms
        r = compiled.report
        print(f"{name:<27}{len(r.kernels):>8}{r.arena_bytes / 2**20:>11.2f}{r.compile_seconds:>11.2f}"
              f"{ms:>12.2f}{baseline / ms:>8.0f}x")
    eager = ms_per_call(model, x)
    print(f"{'eager PyTorch':<27}{'':>8}{'':>11}{'':>11}{eager:>12.2f}{baseline / eager:>8.0f}x")
    full = ms

    print(f"\n{'all optimisations, minus':<27}{'kernels':>8}{'arena MiB':>11}{'compile s':>11}"
          f"{'latency ms':>12}{'slowdown':>10}")
    print(f"{'(nothing; autotuned)':<27}{'':>8}{'':>11}{'':>11}{full:>12.2f}{1:>9.2f}x")
    for option in ("freeze", "fusion", "memory_plan"):
        compiled = kiln.compile(model, (x,), autotune=True, **{option: False})
        torch.testing.assert_close(compiled(x), expected, rtol=1e-3, atol=1e-4)
        ms = ms_per_call(compiled, x)
        r = compiled.report
        print(f"{'- ' + option.replace('_', ' '):<27}{len(r.kernels):>8}{r.arena_bytes / 2**20:>11.2f}"
              f"{r.compile_seconds:>11.2f}{ms:>12.2f}{ms / full:>9.2f}x")

    if full <= eager:
        print(f"\nKiln (all optimisations) is {eager / full:.2f}x faster than eager PyTorch")
    else:
        print(f"\nKiln (all optimisations) is {full / eager:.2f}x slower than eager PyTorch")


if __name__ == "__main__":
    main()