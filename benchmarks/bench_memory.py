# File: benchmarks/bench_memory.py
"""Arena size with and without memory planning, on real models.

The arena holds every intermediate that crosses a kernel boundary. Without
reuse each buffer gets its own slot; with reuse, buffers whose live ranges
don't overlap share memory. The lower bound is the most bytes live during any
one kernel: no plan can do better. Weights are listed for scale; they live
outside the arena.

Memory planning is decided entirely at compile time, so this only runs the
compiler's passes (no code generation): it's fast even for GPT-2 sizes.

    python benchmarks/bench_memory.py
"""

import torch

from kiln.api import Options, build_passes
from kiln.frontend.fx_import import import_exported
from kiln.passes.manager import PassManager

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from models.transformer import Block  # noqa: E402


def mb(nbytes):
    return f"{nbytes / 2**20:.2f}"


def plan(model, x, fusion, reuse):
    g = import_exported(torch.export.export(model, (x,)).run_decompositions())
    pm = PassManager(build_passes(Options(fusion=fusion, memory_plan=reuse)))
    pm.run(g)
    weights = sum(c.nbytes for c in g.constants.values())
    return pm.stats["memory-plan"], len(g.kernels), weights


def main():
    torch.manual_seed(0)
    h = 1024
    workloads = [
        ("MLP 3 x Linear(1024) b=64", torch.nn.Sequential(
            torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h)).eval(), torch.randn(64, h)),
        ("GPT-2 small block T=128", Block(768, 12, 128).eval(), torch.randn(128, 768)),
        ("GPT-2 small block T=512", Block(768, 12, 512).eval(), torch.randn(512, 768)),
    ]
    print(f"{'workload':<28}{'fusion':>7}{'kernels':>9}{'no reuse MiB':>14}{'planned MiB':>13}"
          f"{'saving':>8}{'lower bound MiB':>17}{'weights MiB':>13}")
    for name, model, x in workloads:
        for fusion in (False, True):
            naive, kernels, weights = plan(model, x, fusion, reuse=False)
            planned, _, _ = plan(model, x, fusion, reuse=True)
            print(f"{name:<28}{'on' if fusion else 'off':>7}{kernels:>9}"
                  f"{mb(naive['arena_bytes']):>14}{mb(planned['arena_bytes']):>13}"
                  f"{naive['arena_bytes'] / planned['arena_bytes']:>7.2f}x"
                  f"{mb(planned['lower_bound_bytes']):>17}{mb(weights):>13}")


if __name__ == "__main__":
    main()