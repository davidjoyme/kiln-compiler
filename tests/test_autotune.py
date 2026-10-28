# File: tests/test_autotune.py
"""Matmul autotuning: legal tiles, never worse than the cost model, cached, correct."""

import json

import pytest
import torch

import kiln
import kiln.passes.autotune as autotune
from kiln.passes.autotune import Autotuner, cache_path, cache_variants, register_tiles
from kiln.passes.schedule import matmul_tiles, max_mr
from kiln.target import Target, host

AVX2 = Target("AVX2", vector_lanes=8, vector_registers=16, l1_bytes=48 * 1024, l2_bytes=3 << 20)
SHAPE = (1, 64, 256, 128)     # 2.1M multiply-adds: just above MIN_WORK, quick to tune


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KILN_CACHE_DIR", str(tmp_path))
    yield


def legal(tiles, m, n, k, target):
    return (m % tiles["mc"] == 0 and tiles["mc"] % tiles["mr"] == 0
            and n % tiles["nc"] == 0 and tiles["nc"] % tiles["nr"] == 0
            and k % tiles["kc"] == 0 and tiles["mr"] <= max_mr(tiles["nr"], target))


@pytest.mark.parametrize("target", [host(), AVX2], ids=["host", "avx2"])
@pytest.mark.parametrize("m,n,k", [(128, 3072, 768), (128, 768, 768), (64, 256, 128), (12, 40, 30)])
def test_every_candidate_is_legal(m, n, k, target):
    phase1 = register_tiles(m, n, k, target)
    assert all(legal(t, m, n, k, target) for t in phase1)
    best = phase1[0] if phase1 else matmul_tiles(m, n, k, target)
    assert all(legal(t, m, n, k, target) for t in cache_variants(m, n, k, best))


def test_tuning_beats_or_matches_the_cost_model():
    tuner = Autotuner()
    tiles = tuner.tiles_for(*SHAPE)
    (entry,) = tuner.log
    assert entry["source"] == "tuned"
    assert legal(tiles, *SHAPE[1:], host())
    # The model's own tiles are a candidate, so the winner is at least as fast.
    assert entry["gflops"] >= entry["model_gflops"]


def test_results_are_cached_and_reused(monkeypatch):
    first = Autotuner()
    tiles = first.tiles_for(*SHAPE)
    assert cache_path().exists()

    def no_measuring(*args, **kwargs):
        raise AssertionError("a cached shape must not be measured again")

    monkeypatch.setattr(autotune, "measure", no_measuring)
    second = Autotuner()
    assert second.tiles_for(*SHAPE) == tiles
    assert second.log[0]["source"] == "cache"


def test_small_matmuls_use_the_cost_model(monkeypatch):
    monkeypatch.setattr(autotune, "measure", lambda *a, **k: pytest.fail("small matmul was tuned"))
    tuner = Autotuner()
    assert tuner.tiles_for(1, 16, 32, 32) == matmul_tiles(16, 32, 32, host())
    assert tuner.log == []


def test_corrupt_cache_is_ignored():
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    Autotuner().tiles_for(*SHAPE)
    json.loads(path.read_text())      # rewritten as valid JSON


def test_compile_with_autotune_is_correct():
    model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(approximate="tanh")).eval()
    x = torch.randn(64, 128)
    compiled = kiln.compile(model, (x,), autotune=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x), rtol=1e-3, atol=1e-4)
    (entry,) = compiled.report.autotune
    assert entry["shape"] == (1, 64, 256, 128)
    kernel = next(k for k in compiled.graph.kernels if k.kind == "matmul")
    assert {name: kernel.attrs[name] for name in entry["tiles"]} == entry["tiles"]