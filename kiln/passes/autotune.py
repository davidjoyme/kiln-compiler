# File: kiln/passes/autotune.py
"""Matmul autotuning: measure tile sizes on the real CPU instead of predicting them.

The cost model in kiln/passes/schedule.py reasons from a few numbers (vector
width, register count, cache sizes), but real performance also depends on
things it doesn't model: how many FMA units there are and their latency, the
load ports, the prefetchers, how LLVM schedules and allocates registers. On
the AVX-512 machine Kiln was developed on, the model's tiles matched MKL; on
an AVX2 laptop CPU they were 15-25% behind. Search fixes that, the way
AutoTVM / Ansor and Triton's autotuner do: try candidate schedules on the
actual hardware, time them, keep the fastest.

The search is small and structured, in two phases:

  1. Register tile. Every mr x nr with nr = 1, 2 or 4 vectors (dividing n)
     and mr up to what the register file holds; kc / mc / nc from the cost
     model's formulas for that register tile.
  2. Cache tiles. Around the best register tile: kc from {64, ..., 512},
     nc from {64, ..., 512} (multiples of nr), mc from {32, 64, 128}, all
     dividing their loop extents.

Each candidate is compiled as a standalone matmul, checked against a
reference result, and timed (best of five rounds). Candidates MLIR can't
vectorise are discarded.

Timing noise is the main hazard: a machine running anything else can swing
20% between rounds, and a tuner that trusts single measurements picks
different "winners" on every run. So the fastest few candidates and the cost
model's choice are re-timed in a final round, interleaved (A, B, C, A, B,
C, ...) so that slow drift hits them all equally, and the model's tiles are
kept unless the winner beats them by at least MIN_GAIN. Tuning can then only
switch away from the model for a real, repeatable improvement.

Results are cached on disk, keyed by target and shape, at
$KILN_CACHE_DIR/autotune.json (default ~/.cache/kiln/autotune.json): the
search runs once per shape per machine, and later compiles just look it up.
Matmuls smaller than MIN_WORK multiply-adds use the cost model directly; for
them a schedule makes little difference and tuning isn't worth its time.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import torch

from kiln.passes.schedule import MAX_MR, cache_tiles, largest_divisor, matmul_tiles, max_mr
from kiln.target import Target, host

MIN_WORK = 1 << 21          # multiply-adds (batch * m * n * k); smaller matmuls aren't tuned
MIN_GAIN = 1.03             # a candidate must beat the cost model by 3% to replace it
FINALISTS = 3               # candidates re-timed (with the model) in the final round


def cache_path() -> Path:
    base = os.environ.get("KILN_CACHE_DIR") or os.path.join(Path.home(), ".cache", "kiln")
    return Path(base) / "autotune.json"


def shape_key(target: Target, batch, m, n, k):
    return f"{target.name}|L1={target.l1_bytes}|L2={target.l2_bytes}|{batch}x{m}x{n}x{k}"


# -- candidates ------------------------------------------------------------------

def register_tiles(m, n, k, t: Target):
    """Phase 1: (mr, nr) choices with cost-model cache tiles for each."""
    out = []
    for vectors in (1, 2, 4):
        nr = vectors * t.vector_lanes
        if n % nr:
            continue
        limit = min(MAX_MR, max_mr(nr, t))
        mrs = sorted({largest_divisor(m, limit), largest_divisor(m, max(1, limit // 2))})
        for mr in mrs:
            out.append({**cache_tiles(m, n, k, mr, nr, t), "mr": mr, "nr": nr})
    return out


def cache_variants(m, n, k, best):
    """Phase 2: kc / nc / mc variants around the best register tile."""
    mr, nr = best["mr"], best["nr"]
    kcs = sorted({kc for kc in (64, 128, 256, 512) if k % kc == 0} | {best["kc"]})
    ncs = sorted({nc for nc in (64, 128, 256, 512) if n % nc == 0 and nc % nr == 0} | {best["nc"]})
    mcs = sorted({mc for mc in (32, 64, 128) if m % mc == 0 and mc % mr == 0} | {best["mc"]})
    return [{"mc": mc, "nc": nc, "kc": kc, "mr": mr, "nr": nr} for kc in kcs for nc in ncs for mc in mcs]


# -- measurement -----------------------------------------------------------------

def _matmul_graph(batch, m, n, k):
    from kiln.ir import Graph, TensorType

    g = Graph("matmul")
    lead = (batch,) if batch > 1 else ()
    a = g.add_arg(TensorType((*lead, m, k)), "a")
    b = g.add_arg(TensorType((*lead, k, n)), "b")
    g.results = [g.add_node("matmul", [a, b])]
    return g


def _best_seconds(fn, min_round=0.03, rounds=5):
    fn()
    start, calls = time.perf_counter(), 0
    while time.perf_counter() - start < min_round:
        fn()
        calls += 1
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        best = min(best, (time.perf_counter() - start) / calls)
    return best


def compile_candidate(batch, m, n, k, tiles, target: Target, inputs, expected):
    """A callable running one candidate, or None if it's invalid or falls back."""
    from kiln.api import Options, compile_graph

    options = Options(target=target, matmul_tiles_for=lambda *_: dict(tiles))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)     # a fallback disqualifies the candidate
        try:
            exe, _ = compile_graph(_matmul_graph(batch, m, n, k), options)
        except UserWarning:
            return None
    (out,) = exe(*inputs)
    if not torch.allclose(out, expected, rtol=1e-3, atol=1e-3):
        return None
    return lambda: exe(*inputs)


def measure(batch, m, n, k, tiles, target: Target, inputs, expected):
    """(seconds per call, runner) for one candidate, or None if it's invalid."""
    run = compile_candidate(batch, m, n, k, tiles, target, inputs, expected)
    return None if run is None else (_best_seconds(run), run)


def final_round(runners, rounds=5):
    """Re-time runners interleaved; returns the best seconds per call of each."""
    for run in runners:
        run()
    best = [float("inf")] * len(runners)
    calls = 1
    start = time.perf_counter()
    while time.perf_counter() - start < 0.02:
        runners[0]()
        calls += 1
    for _ in range(rounds):
        for i, run in enumerate(runners):
            t0 = time.perf_counter()
            for _ in range(calls):
                run()
            best[i] = min(best[i], (time.perf_counter() - t0) / calls)
    return best


# -- the tuner -------------------------------------------------------------------

class Autotuner:
    """Chooses matmul tiles by measurement, with an on-disk cache.

    Pass ``tiles_for`` to the Schedule pass. ``log`` records what happened
    for each shape (for the compile report and benchmarks).
    """

    def __init__(self, target: Target | None = None, verbose=False, use_cache=True):
        self.target = target or host()
        self.verbose = verbose
        self.use_cache = use_cache
        self.log = []
        self._cache = self._load() if use_cache else {}

    def _load(self):
        try:
            return json.loads(cache_path().read_text())
        except (OSError, ValueError):
            return {}

    def _save(self):
        if not self.use_cache:
            return
        path = cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, indent=1, sort_keys=True))
            tmp.replace(path)
        except OSError as err:
            warnings.warn(f"Kiln: couldn't write the autotune cache ({err})")

    def tiles_for(self, batch, m, n, k, target=None):
        target = target or self.target
        model = matmul_tiles(m, n, k, target)
        if batch * m * n * k < MIN_WORK:
            return model
        key = shape_key(target, batch, m, n, k)
        if key in self._cache:
            entry = self._cache[key]
            self.log.append({"shape": (batch, m, n, k), "source": "cache", **entry})
            return dict(entry["tiles"])
        entry = self.tune(batch, m, n, k, target, model)
        self._cache[key] = entry
        self._save()
        self.log.append({"shape": (batch, m, n, k), "source": "tuned", **entry})
        return dict(entry["tiles"])

    def tune(self, batch, m, n, k, target, model):
        start = time.perf_counter()
        torch.manual_seed(0)
        lead = (batch,) if batch > 1 else ()
        inputs = (torch.randn(*lead, m, k), torch.randn(*lead, k, n))
        expected = inputs[0] @ inputs[1]
        flops = 2 * batch * m * n * k
        results = {}   # tiles key -> (seconds, runner) or None

        def key(tiles):
            return tuple(sorted(tiles.items()))

        def seconds(tiles):
            k_ = key(tiles)
            if k_ not in results:
                results[k_] = measure(batch, m, n, k, tiles, target, inputs, expected)
            r = results[k_]
            return float("inf") if r is None else r[0]

        # Phase 1: register tiles. Phase 2: cache tiles around the best one.
        phase1 = [model] + register_tiles(m, n, k, target)
        best = min(phase1, key=seconds)
        for t in cache_variants(m, n, k, best):
            seconds(t)

        # Final round: the fastest few and the model, re-timed side by side.
        valid = {k_: r for k_, r in results.items() if r is not None}
        ranked = sorted(valid, key=lambda k_: valid[k_][0])[:FINALISTS]
        model_key = key(model)
        if model_key in valid and model_key not in ranked:
            ranked.append(model_key)
        times = dict(zip(ranked, final_round([valid[k_][1] for k_ in ranked])))
        winner = min(times, key=times.get)
        if model_key in times and times[model_key] <= times[winner] * MIN_GAIN:
            winner = model_key          # not a clear, repeatable win: keep the model's tiles
        model_seconds = times.get(model_key)
        entry = {
            "tiles": dict(winner),
            "model_tiles": model,
            "gflops": round(flops / times[winner] / 1e9, 1),
            "model_gflops": round(flops / model_seconds / 1e9, 1) if model_seconds else None,
            "candidates": len(results),
            "seconds": round(time.perf_counter() - start, 1),
        }
        if self.verbose:
            print(f"autotune {batch}x{m}x{n}x{k}: {entry['model_gflops']} -> {entry['gflops']} GFLOP/s "
                  f"({entry['candidates']} candidates, {entry['seconds']} s), tiles {entry['tiles']}")
        return entry