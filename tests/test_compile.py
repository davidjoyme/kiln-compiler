# File: tests/test_compile.py
"""End-to-end: every result Kiln computes must match eager PyTorch."""

import pytest
import torch
import torch.nn.functional as F

import kiln
from models.transformer import Block


def check(fn, *inputs, rtol=1e-4, atol=1e-5, **flags):
    compiled = kiln.compile(fn, inputs, **flags)
    with torch.no_grad():
        expected = fn(*inputs)
    actual = compiled(*inputs)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return compiled


FREEZE = pytest.mark.parametrize("freeze", [True, False], ids=["frozen", "unfrozen"])


@FREEZE
@pytest.mark.parametrize("shape", [(2, 4, 8, 3), (16, 32, 64, 10), (1, 7, 5, 3)])
def test_mlp(shape, freeze):
    b, i, h, o = shape
    model = torch.nn.Sequential(torch.nn.Linear(i, h), torch.nn.ReLU(), torch.nn.Linear(h, o))
    check(model, torch.randn(b, i), freeze=freeze)


@FREEZE
def test_transformer_block(freeze):
    c = check(Block(d=64, nh=4, T=16).eval(), torch.randn(16, 64), freeze=freeze)
    assert c.report.lowering["heap_allocs"] == 0


@pytest.mark.parametrize("op", [torch.exp, torch.tanh, torch.sigmoid, torch.relu, torch.neg,
                                torch.rsqrt, torch.sqrt, torch.erf,
                                lambda x: F.gelu(x), lambda x: F.gelu(x, approximate="tanh")],
                         ids=["exp", "tanh", "sigmoid", "relu", "neg", "rsqrt", "sqrt", "erf",
                              "gelu", "gelu_tanh"])
def test_unary(op):
    x = torch.randn(8, 33) * 3
    if op in (torch.rsqrt, torch.sqrt):
        x = x.abs() + 0.1
    check(op, x)


def test_exp_extremes():
    x = torch.tensor([[-200.0, -87.0, -10.0, 0.0, 1e-6, 10.0, 80.0, float("-inf")]])
    check(torch.exp, x, rtol=1e-5, atol=1e-30)


def test_broadcasting_binary_ops():
    def fn(x, row, col):
        return torch.maximum((x + row) * col - x / 2.0, torch.minimum(x, row))

    check(fn, torch.randn(4, 5), torch.randn(5), torch.randn(4, 1))


def test_reductions_and_softmax():
    def fn(x):
        return x.sum(-1, keepdim=True) + x.amax(-1, keepdim=True) + x.mean(0), F.softmax(x, -1)

    check(fn, torch.randn(6, 10))


def test_layer_norm():
    check(torch.nn.LayerNorm(48).eval(), torch.randn(5, 48))


def test_views_are_resolved_without_copies():
    w = torch.randn(3, 4, 5)

    def fn(x):
        y = x.view(4, 3, 5).transpose(0, 1)       # strided view
        return y[:, 1:3, :] * w[:, 1:3, :]        # slice of a view

    c = check(fn, torch.randn(12, 5))
    assert not any(k.kind == "copy" for k in c.graph.kernels)


def test_batched_matmul_with_transposed_operand():
    def fn(a, b):
        return torch.bmm(a, b.transpose(1, 2))   # B is not N-contiguous: needs a copy

    c = check(fn, torch.randn(3, 4, 5), torch.randn(3, 6, 5))
    assert any(k.kind == "copy" for k in c.graph.kernels)


def test_multiple_outputs_and_returning_an_input_view():
    def fn(x):
        return x * 2, x.t()

    check(fn, torch.randn(3, 4))


def test_unfrozen_module_sees_weight_updates():
    model = torch.nn.Linear(4, 4)
    x = torch.randn(2, 4)
    compiled = kiln.compile(model, (x,), freeze=False)
    with torch.no_grad():
        model.weight.add_(1.0)
        torch.testing.assert_close(compiled(x), model(x))


def test_torch_compile_backend():
    kiln.stats.update(compiled=0, fallback=0)
    torch._dynamo.reset()
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU()).eval()
    compiled = torch.compile(model, backend=kiln.backend, dynamic=False)
    x = torch.randn(4, 8)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-5)
    assert kiln.stats["compiled"] == 1


def test_backend_falls_back_on_unsupported_ops(monkeypatch):
    monkeypatch.setenv("KILN_STRICT", "0")
    kiln.stats.update(compiled=0, fallback=0)
    torch._dynamo.reset()
    fn = lambda x: torch.cumsum(x, 0)
    compiled = torch.compile(fn, backend=kiln.backend, dynamic=False)
    x = torch.randn(5)
    with pytest.warns(UserWarning, match="falling back"):
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), fn(x))
    assert kiln.stats["fallback"] == 1


def test_kernel_that_folds_to_a_copy_of_a_strided_input():
    # maximum(x, x) folds to x, so the kernel becomes "copy the input", and
    # the input is a transposed (strided) view. Bufferization emits a
    # memref.copy, which must lower to loops, not a runtime-library call.
    fn = lambda x: torch.maximum(x.t(), x.t())
    c = check(fn, torch.randn(4, 6))
    assert c.report.lowering["copies_to_loops"] >= 1