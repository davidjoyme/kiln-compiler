# File: tests/test_schedule.py
"""Tiling + vectorisation: the cost model's invariants, and correctness at real sizes.

Every compile here runs with Kiln's warnings turned into errors, so a
schedule that MLIR rejects (and silently falls back to plain loops in normal
use) fails the test instead of passing slowly.
"""

import warnings

import pytest
import torch
import torch.nn.functional as F

import kiln
from kiln.passes.schedule import matmul_tiles
from kiln.target import Target, host
from models.transformer import Block

AVX512 = Target("AVX-512", vector_lanes=16, vector_registers=32, l1_bytes=48 * 1024, l2_bytes=2 << 20)
AVX2 = Target("AVX2", vector_lanes=8, vector_registers=16, l1_bytes=48 * 1024, l2_bytes=3 << 20)
NEON = Target("NEON", vector_lanes=4, vector_registers=32, l1_bytes=64 * 1024, l2_bytes=1 << 20)
TARGETS = pytest.mark.parametrize("target", [AVX512, AVX2, NEON], ids=lambda t: t.name)

SHAPES = [(128, 2304, 768), (128, 768, 3072), (256, 256, 256), (1, 3072, 768),
          (7, 13, 5), (100, 320, 96), (1, 1, 1), (64, 17, 31)]


@TARGETS
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_matmul_tiles_are_legal(m, n, k, target):
    t = matmul_tiles(m, n, k, target)
    # every tile divides its loop, so every tile is full (no remainder loops)
    assert m % t["mc"] == 0 and t["mc"] % t["mr"] == 0
    assert n % t["nc"] == 0 and t["nc"] % t["nr"] == 0
    assert k % t["kc"] == 0
    # the register tile fits: accumulators + B vectors + one broadcast register
    nr_vectors = -(-t["nr"] // target.vector_lanes)
    assert t["mr"] * nr_vectors + nr_vectors + 1 <= target.vector_registers
    # A and B slivers of one kc slice fit in half the L1
    assert (t["mr"] + t["nr"]) * t["kc"] * 4 <= target.l1_bytes // 2 or t["kc"] == 1


@TARGETS
def test_gpt2_matmul_uses_full_vector_register_tiles(target):
    t = matmul_tiles(128, 3072, 768, target)
    assert t["nr"] == 2 * target.vector_lanes        # two full vectors per row
    assert t["mr"] >= 4                              # at least 8 accumulators


def compile_strict(fn, xs, **flags):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return kiln.compile(fn, xs, **flags)


def check(fn, *xs, rtol=2e-4, atol=2e-5, **flags):
    c = compile_strict(fn, xs, **flags)
    with torch.no_grad():
        torch.testing.assert_close(c(*xs), fn(*xs), rtol=rtol, atol=atol)
    return c


@TARGETS
def test_gpt2_sized_kernels(target):
    # Real sizes matter: rows of 768 and 128 are split into several vector
    # chunks, which the small shapes elsewhere in the suite never need.
    ln = torch.nn.LayerNorm(768).eval()
    check(ln, torch.randn(128, 768), target=target)
    check(lambda t: F.softmax(t, -1), torch.randn(12, 128, 128), target=target)
    mlp_up = torch.nn.Sequential(torch.nn.Linear(768, 3072), torch.nn.GELU(approximate="tanh")).eval()
    c = check(mlp_up, torch.randn(128, 768), target=target, rtol=1e-3, atol=1e-4)
    assert c.report.lowering["vector_ops"] > 0


@TARGETS
def test_vector_chunk_that_does_not_divide_the_cache_tile(target):
    # n = 320: nc = 160 (a multiple of nr = 32), but a 64-wide epilogue chunk
    # doesn't divide 160. Chunks must be sized from the tile, not the tensor.
    check(torch.nn.Sequential(torch.nn.Linear(96, 320), torch.nn.ReLU()).eval(),
          torch.randn(100, 96), target=target)


@pytest.mark.parametrize("shape", [(1, 64), (3, 7), (130, 50), (2, 5, 33)])
def test_row_kernels_odd_shapes(shape):
    x = torch.randn(*shape)
    check(lambda t: F.layer_norm(t, (shape[-1],)) * F.softmax(t, -1), x)


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (7, 13, 5), (17, 31, 64), (3, 100, 9)])
def test_matmul_odd_shapes(m, n, k):
    b = torch.randn(k, n)
    check(lambda a: a @ b, torch.randn(m, k))


@TARGETS
def test_transformer_block(target):
    c = check(Block(d=96, nh=3, T=40).eval(), torch.randn(40, 96), target=target)
    assert c.report.lowering["fallback"] == []
    assert c.report.lowering["heap_allocs"] == 0


def test_tiling_off_means_no_vector_code():
    lin = torch.nn.Linear(32, 64).eval()
    x = torch.randn(8, 32)
    assert check(lin, x, tiling=False).report.lowering["vector_ops"] == 0
    assert check(lin, x, tiling=True).report.lowering["vector_ops"] > 0


def test_host_target_is_detected():
    t = host()
    assert t.vector_lanes in (4, 8, 16) and t.vector_registers >= 16
    assert t.l1_bytes >= 16 * 1024 and t.l2_bytes >= t.l1_bytes