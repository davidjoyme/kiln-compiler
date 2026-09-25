"""Correctness tests: every Kiln result must match eager PyTorch."""

import pytest
import torch

import kiln


@pytest.fixture(autouse=True)
def fresh_compiler_state(monkeypatch):
    # Fail loudly on unsupported ops instead of silently falling back.
    monkeypatch.setenv("KILN_STRICT", "1")
    torch._dynamo.reset()
    kiln.stats.update(compiled=0, fallback=0)
    torch.manual_seed(0)
    yield


def compile_and_compare(fn, *inputs):
    compiled = torch.compile(fn, backend=kiln.backend, dynamic=False)
    with torch.no_grad():
        expected = fn(*inputs)
        actual = compiled(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert kiln.stats["compiled"] >= 1, "graph did not go through Kiln"


@pytest.mark.parametrize("batch,d_in,d_hidden,d_out", [(2, 4, 8, 3), (16, 32, 64, 10), (1, 7, 5, 3)])
def test_mlp(batch, d_in, d_hidden, d_out):
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, d_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(d_hidden, d_out),
    ).eval()
    compile_and_compare(model, torch.randn(batch, d_in))


def test_repeated_calls_reuse_compiled_code():
    model = torch.nn.Linear(4, 4).eval()
    compiled = torch.compile(model, backend=kiln.backend, dynamic=False)
    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(3, 4)
            torch.testing.assert_close(compiled(x), model(x))
    assert kiln.stats["compiled"] == 1


def test_matmul_without_bias():
    compile_and_compare(lambda a, b: a @ b, torch.randn(5, 6), torch.randn(6, 7))


def test_broadcasting_elementwise():
    def fn(x, row, col):
        return (x + row) * col - x

    compile_and_compare(fn, torch.randn(4, 5), torch.randn(5), torch.randn(4, 1))


def test_scalar_operands():
    compile_and_compare(lambda x: torch.relu(x * 2.0 + 1.0), torch.randn(3, 3))


def test_multiple_outputs():
    compile_and_compare(lambda x, y: (x + y, x * y), torch.randn(2, 3), torch.randn(2, 3))


def test_unsupported_op_falls_back(monkeypatch):
    monkeypatch.setenv("KILN_STRICT", "0")
    fn = lambda x: torch.sigmoid(x)
    compiled = torch.compile(fn, backend=kiln.backend, dynamic=False)
    x = torch.randn(3)
    with pytest.warns(UserWarning, match="falling back"):
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), fn(x))
    assert kiln.stats["fallback"] == 1