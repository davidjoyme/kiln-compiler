# File: tests/test_memory_plan.py
"""Static memory planning: safety (live buffers never share memory), quality, correctness."""

import random

import numpy as np
import pytest
import torch

import kiln
from kiln.api import Options, build_passes
from kiln.frontend.fx_import import import_exported
from kiln.ir import Graph, TensorType, interpret, parse
from kiln.passes.buffers import BufferAssignment
from kiln.passes.manager import PassManager
from kiln.passes.memory_plan import MemoryPlan, align, live_ranges, overlaps
from kiln.passes.partition import OneOpPerKernel
from models.transformer import Block


def assert_plan_is_safe(graph):
    """No two buffers that are live at the same time may share any byte."""
    ranges = live_ranges(graph)
    spans = {v: (v.node.attrs["offset"], v.node.attrs["offset"] + align(v.type.nbytes)) for v in ranges}
    buffers = list(ranges)
    for i, a in enumerate(buffers):
        assert spans[a][0] % 64 == 0, "buffers must be 64-byte aligned"
        assert spans[a][1] <= graph.attrs["arena_bytes"]
        for b in buffers[i + 1:]:
            if overlaps(ranges[a], ranges[b]):
                (s0, e0), (s1, e1) = spans[a], spans[b]
                assert e0 <= s1 or e1 <= s0, f"live buffers share memory: {a} and {b}"


def planned(src, reuse=True):
    g = parse(src)
    pm = PassManager([OneOpPerKernel(), BufferAssignment(), MemoryPlan(reuse=reuse)])
    pm.run(g)
    return g, pm.stats["memory-plan"]


@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "fused"])
def test_transformer_block_plan(fusion):
    model, x = Block(d=64, nh=4, T=16).eval(), torch.randn(16, 64)
    g = import_exported(torch.export.export(model, (x,)).run_decompositions())
    pm = PassManager(build_passes(Options(fusion=fusion)))
    pm.run(g)
    stats = pm.stats["memory-plan"]
    assert_plan_is_safe(g)
    assert stats["arena_bytes"] == stats["lower_bound_bytes"]       # optimal here
    assert stats["naive_bytes"] >= 2 * stats["arena_bytes"]


def test_chain_needs_two_buffers():
    # a -> b -> c -> d -> e: each buffer dies right after the next one is
    # written, so two slots (ping-pong) are enough for any chain length.
    g, stats = planned("""
    graph @chain(%arg0: f32[16, 16]) -> (f32[16, 16]) {
      %0 = exp %arg0 : f32[16, 16]
      %1 = relu %0 : f32[16, 16]
      %2 = neg %1 : f32[16, 16]
      %3 = exp %2 : f32[16, 16]
      %4 = relu %3 : f32[16, 16]
      return %4
    }""")
    assert stats["buffers"] == 4 and stats["naive_bytes"] == 4 * 1024
    assert stats["arena_bytes"] == 2 * 1024
    assert_plan_is_safe(g)


def test_a_kernel_never_writes_over_its_own_input():
    # %1 reads %0 and writes its own buffer in the same kernel: their live
    # ranges touch at that kernel, so they must not share memory.
    g, _ = planned("""
    graph @f(%arg0: f32[8, 8]) -> (f32[8, 8]) {
      %0 = exp %arg0 : f32[8, 8]
      %1 = relu %0 : f32[8, 8]
      %2 = neg %1 : f32[8, 8]
      return %2
    }""")
    v0, v1 = (n.out for n in g.nodes[:2])
    assert v0.node.attrs["offset"] != v1.node.attrs["offset"]


def test_reads_through_views_extend_lifetimes():
    # %0 is read again by the last kernel through a transpose; buffers
    # written in between must not reuse its memory.
    g, _ = planned("""
    graph @f(%arg0: f32[8, 8]) -> (f32[8, 8]) {
      %0 = exp %arg0 : f32[8, 8]
      %1 = relu %0 : f32[8, 8]
      %2 = neg %1 : f32[8, 8]
      %3 = transpose %0 {perm = [1, 0]} : f32[8, 8]
      %4 = add %2, %3 : f32[8, 8]
      return %4
    }""")
    ranges = live_ranges(g)
    v0 = g.nodes[0].out
    assert ranges[v0] == [0, 3]
    assert_plan_is_safe(g)


def test_no_reuse_gives_every_buffer_its_own_slot():
    _, stats = planned("""
    graph @chain(%arg0: f32[16, 16]) -> (f32[16, 16]) {
      %0 = exp %arg0 : f32[16, 16]
      %1 = relu %0 : f32[16, 16]
      %2 = neg %1 : f32[16, 16]
      return %2
    }""", reuse=False)
    assert stats["arena_bytes"] == stats["naive_bytes"] == 2 * 1024


def random_graph(rng, n_ops):
    """A random DAG of elementwise ops, transposes and row reductions on 2-D tensors.

    Values stay bounded (no exp; reductions centre a row instead of scaling
    it), so a mismatch means a compiler bug, not float overflow.
    """
    g = Graph("random")
    size = rng.choice([4, 8, 12])
    values = [g.add_arg(TensorType((size, size)), "x")]
    for _ in range(n_ops):
        a = rng.choice(values[-4:])
        kind = rng.random()
        if kind < 0.15:
            values.append(g.add_node("transpose", [a], {"perm": [1, 0]}))
        elif kind < 0.3:
            mean = g.add_node("mul", [g.add_node("reduce_sum", [a], {"dims": [1]}), g.scalar(1 / size)])
            values.append(g.add_node("sub", [a, mean]))
        elif kind < 0.6:
            b = rng.choice(values)
            values.append(g.add_node(rng.choice(["add", "sub", "mul", "maximum"]), [a, b]))
        else:
            values.append(g.add_node(rng.choice(["tanh", "sigmoid", "relu", "neg"]), [a]))
    g.results = [values[-1], rng.choice(values[1:-1])]
    return g


# Seeds 4, 37, 45 and 59 found a fusion bug (an op reading its own kernel's
# result through a transpose); they stay in the list as regression tests.
SEEDS = [0, 1, 2, 3, 4, 5, 37, 45, 59]


@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("seed", SEEDS)
def test_random_graphs_match_the_interpreter(seed, fusion):
    rng = random.Random(seed)
    g = random_graph(rng, n_ops=rng.randint(6, 22))
    x = np.random.default_rng(seed).standard_normal(g.args[0].shape).astype(np.float32) * 0.5
    expected = interpret(g, x)
    exe, report = kiln.compile_graph(g, Options(fusion=fusion))
    assert_plan_is_safe(g)
    assert report.arena_bytes <= report.naive_bytes
    for out, ref in zip(exe(torch.from_numpy(x)), expected):
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)


def test_compiled_results_identical_with_and_without_reuse():
    model, x = Block(d=64, nh=4, T=16).eval(), torch.randn(16, 64)
    with_reuse = kiln.compile(model, (x,), memory_plan=True)
    without = kiln.compile(model, (x,), memory_plan=False)
    torch.testing.assert_close(with_reuse(x), without(x), rtol=0, atol=0)
    assert with_reuse.report.arena_bytes < without.report.arena_bytes