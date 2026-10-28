# File: tests/test_gpu.py
"""The GPU backend (generated Triton kernels): every result must match eager PyTorch.

Runs on a CUDA GPU if there is one, otherwise on Triton's interpreter (see
tests/conftest.py), so it passes on machines without a GPU too.
"""

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import kiln
from kiln.api import Options
from kiln.gpu import device, interpreting
from kiln.gpu.autotune import SHARED_MEMORY_BYTES, GPUAutotuner, candidates
from kiln.gpu.runtime import check_memory
from kiln.ir import interpret
from models.transformer import Block
from test_memory_plan import random_graph

ON_GPU = not interpreting()


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KILN_CACHE_DIR", str(tmp_path))
    yield


def as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def check(fn, *xs, rtol=1e-4, atol=1e-5, **flags):
    compiled = kiln.compile(fn, xs, device="cuda", **flags)
    with torch.no_grad():
        expected = as_tuple(fn(*xs))
    actual = as_tuple(compiled(*xs))
    for a, e in zip(actual, expected):
        assert a.device.type == device().type
        torch.testing.assert_close(a.cpu(), e, rtol=rtol, atol=atol)
    return compiled


W = torch.randn(37, 70)
Y = torch.randn(3, 1)
MASK = torch.full((2, 5000), float("-inf")).triu(1)

CASES = {
    "mlp": (torch.nn.Sequential(torch.nn.Linear(24, 40), torch.nn.GELU(approximate="tanh"),
                                torch.nn.Linear(40, 8)).eval(), (torch.randn(5, 24),)),
    "layernorm": (torch.nn.LayerNorm(48).eval(), (torch.randn(7, 48),)),
    "softmax": (lambda x: F.softmax(x, -1), (torch.randn(3, 5, 33),)),
    "column sum": (lambda x: x.sum(0, keepdim=True) * x, (torch.randn(6, 10),)),
    "full reduction": (lambda x: x.sum((0, 1), keepdim=True) * 2.0, (torch.randn(3, 4),)),
    "reduce dims 0, 2": (lambda x: x.sum((0, 2), keepdim=True) + x.amax(0, keepdim=True), (torch.randn(4, 5, 6),)),
    "views": (lambda x: (x.view(4, 3, 5).transpose(0, 1) * 2.0, x.t()), (torch.randn(12, 5),)),
    "bmm, transposed operand": (lambda a, b: torch.bmm(a, b.transpose(1, 2)),
                                (torch.randn(3, 9, 20), torch.randn(3, 17, 20))),
    "decode, M = 1": (lambda a: F.gelu(a @ W, approximate="tanh"), (torch.randn(1, 37),)),
    "odd K": (lambda a, b: a @ b + 1.0, (torch.randn(33, 45), torch.randn(45, 19))),
    "per-row scalar input": (lambda x: x / (x.sum(-1, keepdim=True) ** 2 + 1.0) + Y, (torch.randn(3, 50),)),
    "multi-shape outputs": (lambda x: (x.exp() * 2, (x.exp() + Y[:2]).sum(-1, keepdim=True)), (torch.randn(2, 9),)),
    "unary ops": (lambda x: torch.tanh(x) + torch.sigmoid(x) + torch.erf(x) + F.gelu(x) + torch.relu(-x),
                  (torch.randn(4, 30) * 3,)),
    "transformer block": (Block(d=32, nh=2, T=8).eval(), (torch.randn(8, 32),)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_eager(name):
    fn, xs = CASES[name]
    check(fn, *xs)


@pytest.mark.parametrize("fn", [torch.nn.LayerNorm(10000).eval(), lambda x: F.softmax(x, -1),
                                lambda x: F.softmax(x + MASK, -1)], ids=["layernorm", "softmax", "masked"])
def test_long_rows_use_constant_memory(fn):
    # Rows longer than MAX_ROW_BLOCK are processed in chunks, one pass per
    # level of reduction, recomputing elementwise values instead of storing them.
    x = torch.randn(2, 10000 if isinstance(fn, torch.nn.LayerNorm) else 5000)
    c = check(fn, x)
    assert "for c0 in range" in open(c.report.lowering["source"]).read()


def test_fusion_and_memory_planning_apply_on_the_gpu():
    model, x = Block(d=32, nh=2, T=8).eval(), torch.randn(8, 32)
    fused = check(model, x)
    unfused = check(model, x, fusion=False, memory_plan=False)
    assert fused.report.lowering["launches"] < unfused.report.lowering["launches"] / 3
    assert fused.report.arena_bytes < unfused.report.arena_bytes / 4


def test_views_are_strided_loads_not_copies():
    fn = lambda x: x.view(4, 3, 5).transpose(0, 1)[:, 1:3, :] * 2.0
    c = check(fn, torch.randn(12, 5))
    assert not any("copy" in k for k in c.report.kernels)


def test_unfrozen_weights_are_shared_with_a_model_on_the_device():
    # With freeze=False the weights are runtime inputs. A model that already
    # lives on the GPU shares its weight memory with the compiled module, so
    # in-place updates are seen. (A CPU model's weights are copied to the GPU
    # once, at compile time.)
    model = torch.nn.Linear(16, 8).to(device())
    x = torch.randn(4, 16, device=device())
    c = kiln.compile(model, (x,), device="cuda", freeze=False)
    with torch.no_grad():
        model.weight.mul_(2.0)
        torch.testing.assert_close(c(x), model(x), rtol=1e-4, atol=1e-5)


def test_tf32_option_changes_the_generated_matmul():
    model = torch.nn.Linear(64, 32).eval()
    x = torch.randn(16, 64)
    exact = check(model, x)
    fast = check(model, x, allow_tf32=True, rtol=1e-2, atol=1e-2)
    assert 'input_precision="ieee"' in open(exact.report.lowering["source"]).read()
    assert 'input_precision="tf32"' in open(fast.report.lowering["source"]).read()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "fused"])
def test_random_graphs(seed, fusion):
    rng = random.Random(seed)
    g = random_graph(rng, n_ops=rng.randint(6, 22))
    x = np.random.default_rng(seed).standard_normal(g.args[0].shape).astype(np.float32) * 0.5
    expected = interpret(g, x)
    exe, _ = kiln.compile_graph(g, Options(fusion=fusion, device="cuda"))
    for out, ref in zip(exe(torch.from_numpy(x)), expected):
        np.testing.assert_allclose(out.cpu().numpy(), ref, rtol=1e-4, atol=1e-5)


def test_memory_check_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (10 * 2**20, 32 * 2**30))
    with pytest.raises(MemoryError, match="only 10.0 MiB"):
        check_memory(torch.device("cuda"), 64 * 2**20, "the arena")


def test_autotune_candidates_fit_shared_memory():
    for m, n, k in [(128, 3072, 768), (1, 768, 768), (7, 13, 5)]:
        for c in candidates(m, n, k):
            assert (c["BM"] + c["BN"]) * c["BK"] * 4 * c["num_stages"] <= SHARED_MEMORY_BYTES
            assert c["BM"] <= max(16, 2 * m) and c["BN"] <= max(16, 2 * n)


def test_autotune_is_correct_and_cached(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(48, 64), torch.nn.GELU(approximate="tanh")).eval()
    x = torch.randn(20, 48)
    first = check(model, x, autotune=True)
    (entry,) = first.report.autotune
    assert entry["source"] == "tuned" and entry["candidates"] >= 1
    monkeypatch.setattr(GPUAutotuner, "tune", lambda *a, **k: pytest.fail("cached shape re-tuned"))
    second = check(model, x, autotune=True)
    assert second.report.autotune[0]["source"] == "cache"


@pytest.mark.skipif(not ON_GPU, reason="CUDA graphs need a real GPU")
def test_cuda_graph_replays_with_new_inputs():
    model = Block(d=64, nh=4, T=16).eval()
    c = kiln.compile(model, (torch.randn(16, 64),), device="cuda")
    assert c.report.lowering["cuda_graphs"]
    for _ in range(3):
        x = torch.randn(16, 64)
        with torch.no_grad():
            torch.testing.assert_close(c(x).cpu(), model(x), rtol=1e-4, atol=1e-5)