# File: tests/test_fusion.py
"""Operator fusion: which ops share a kernel, the legality rules, and correctness."""

import pytest
import torch
import torch.nn.functional as F

import kiln
from kiln.ir import parse
from kiln.passes.fusion import Fusion
from kiln.passes.manager import PassManager
from models.transformer import Block


def kernels_of(src):
    g = parse(src)
    PassManager([Fusion()]).run(g)
    return [(k.kind, [n.op for n in k.nodes]) for k in g.kernels]


def check_both_ways(fn, *xs, rtol=1e-4, atol=1e-5):
    """Fused and unfused must both match eager; returns (unfused, fused)."""
    with torch.no_grad():
        expected = fn(*xs)
    out = []
    for fusion in (False, True):
        compiled = kiln.compile(fn, xs, fusion=fusion)
        torch.testing.assert_close(compiled(*xs), expected, rtol=rtol, atol=atol)
        out.append(compiled)
    return out


def test_transformer_block_kernels():
    unfused, fused = check_both_ways(Block(d=64, nh=4, T=16).eval(), torch.randn(16, 64))
    kinds = [k.split()[1] for k in fused.report.kernels]
    assert len(unfused.report.kernels) == 44
    assert kinds == ["reduction", "matmul", "copy", "matmul", "reduction", "matmul", "copy",
                     "matmul", "reduction", "matmul", "matmul"]
    assert "k9 matmul      matmul add gelu" in fused.report.kernels   # MLP: bias + GELU epilogue
    assert fused.report.lowering["heap_allocs"] == 0                   # row temporaries on the stack
    assert fused.report.lowering["fallback"] == []
    assert fused.report.arena_bytes < unfused.report.arena_bytes / 2


def test_layernorm_is_one_kernel():
    unfused, fused = check_both_ways(torch.nn.LayerNorm(96).eval(), torch.randn(8, 96))
    assert len(fused.report.kernels) == 1 and len(unfused.report.kernels) == 11
    assert fused.report.arena_bytes == 0   # no intermediate leaves the kernel


def test_softmax_is_one_kernel():
    _, fused = check_both_ways(lambda x: F.softmax(x, -1), torch.randn(3, 5, 40))
    assert len(fused.report.kernels) == 1


def test_matmul_epilogue():
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(approximate="tanh")).eval()
    _, fused = check_both_ways(model, torch.randn(8, 32))
    assert fused.report.kernels == ["k0 matmul      matmul add gelu"]


def test_matmul_with_two_users_is_not_epilogue_fused():
    # The matmul result is needed in full by two consumers, so fusing either
    # one into the matmul's tiles would compute the matmul twice. relu and
    # exp are siblings (neither reads the other), and Kiln only fuses
    # producer -> consumer edges, so each gets its own kernel. Merging
    # siblings ("horizontal fusion") is left as future work.
    assert kernels_of("""
    graph @f(%arg0: f32[4, 8], %arg1: f32[8, 16]) -> (f32[4, 16], f32[4, 16]) {
      %0 = matmul %arg0, %arg1 : f32[4, 16]
      %1 = relu %0 : f32[4, 16]
      %2 = exp %0 : f32[4, 16]
      return %1, %2
    }""") == [("matmul", ["matmul"]), ("elementwise", ["relu"]), ("elementwise", ["exp"])]


def test_fusion_never_creates_a_cycle_between_kernels():
    # %1 reduces over rows (dim 0), so it can't join %0's kernel. %2 reads
    # both %0 and %1: putting it in %0's kernel would make that kernel both
    # a producer and a consumer of %1's kernel.
    src = """
    graph @f(%arg0: f32[6, 10]) -> (f32[6, 10]) {
      %0 = exp %arg0 : f32[6, 10]
      %1 = reduce_sum %0 {dims = [0]} : f32[1, 10]
      %2 = mul %0, %1 : f32[6, 10]
      return %2
    }"""
    assert kernels_of(src) == [("elementwise", ["exp"]), ("reduction", ["reduce_sum"]),
                               ("elementwise", ["mul"])]
    check_both_ways(lambda x: x.exp() * x.exp().sum(0, keepdim=True), torch.randn(6, 10))


def test_views_end_kernels():
    # A transpose between two elementwise ops is a view, not a computation:
    # the second op reads the first op's buffer through it.
    assert kernels_of("""
    graph @f(%arg0: f32[4, 6]) -> (f32[6, 4]) {
      %0 = exp %arg0 : f32[4, 6]
      %1 = transpose %0 {perm = [1, 0]} : f32[6, 4]
      %2 = relu %1 : f32[6, 4]
      return %2
    }""") == [("elementwise", ["exp"]), ("elementwise", ["relu"])]


@pytest.mark.parametrize("shape", [(1, 64), (7, 33), (2, 3, 48)])
def test_row_kernels_with_odd_shapes(shape):
    x = torch.randn(*shape)
    check_both_ways(lambda t: F.layer_norm(t, (shape[-1],)) + F.softmax(t, -1), x)