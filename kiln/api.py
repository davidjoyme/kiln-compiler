# File: kiln/api.py
"""Public API: ``kiln.compile(model, example_inputs)`` and the compile pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from kiln.ir.core import Graph


@dataclass
class Options:
    """Which optimisations to run. Every flag can be turned off for ablations."""

    freeze: bool = True          # parameters -> constants, then constant folding
    fusion: bool = True          # multi-op kernels (otherwise one op per kernel)
    memory_plan: bool = True     # buffers share arena memory when lifetimes don't overlap
    tiling: bool = True          # cache/register tiling + vectorisation (otherwise scalar loops)
    target: object = None        # kiln.target.Target to schedule for; default: the host CPU
    autotune: bool = False       # choose matmul tiles by measurement (cached on disk)
    matmul_tiles_for: object = None   # advanced: fn(batch, m, n, k, target) -> tiles
    print_ir: bool = False       # print the Kiln IR after every pass
    dump_mlir: bool = False      # print MLIR (and transform scripts) during lowering
    profile: bool = False        # emit per-kernel entry points for CompiledModule.profile()

    @classmethod
    def none(cls, **kw):
        """Every optimisation off: the baseline for ablations."""
        return cls(freeze=False, fusion=False, memory_plan=False, tiling=False, **kw)


@dataclass
class Report:
    passes: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    lowering: dict = field(default_factory=dict)
    kernels: list = field(default_factory=list)
    arena_bytes: int = 0
    naive_bytes: int = 0         # arena size without reuse (one slot per buffer)
    lower_bound_bytes: int = 0   # peak live bytes: no plan can be smaller
    compile_seconds: float = 0.0
    autotune: list = field(default_factory=list)   # one entry per tuned / cached matmul shape

    def __str__(self):
        lines = [f"Kiln: {len(self.kernels)} kernels, compiled in {self.compile_seconds:.2f}s"]
        for k in self.kernels:
            lines.append(f"  {k}")
        low = self.lowering
        lines.append(f"  arena: {self.arena_bytes} bytes (no reuse: {self.naive_bytes}, "
                     f"lower bound: {self.lower_bound_bytes}), "
                     f"heap allocations per call: {low.get('heap_allocs', 0)}, "
                     f"transformed kernels: {len(low.get('transformed', []))}, "
                     f"fallbacks: {len(low.get('fallback', []))}")
        for name, stats in self.passes.items():
            lines.append(f"  {name}: {stats}")
        for e in self.autotune:
            b, m, n, k = e["shape"]
            lines.append(f"  autotune {b}x{m}x{n}x{k} ({e['source']}): {e['model_gflops']} -> "
                         f"{e['gflops']} GFLOP/s, tiles {e['tiles']}")
        return "\n".join(lines)


def build_passes(options: Options, tuner=None):
    from kiln.passes.buffers import BufferAssignment
    from kiln.passes.memory_plan import MemoryPlan
    from kiln.passes.freeze import ConstantFold, FreezeParameters
    from kiln.passes.fusion import Fusion
    from kiln.passes.partition import OneOpPerKernel
    from kiln.passes.schedule import Schedule
    from kiln.passes.simplify import CSE, DCE, Canonicalize

    passes = [Canonicalize(), CSE()]
    if options.freeze:
        passes += [FreezeParameters(), ConstantFold(), Canonicalize(), CSE()]
    passes += [DCE(), Fusion() if options.fusion else OneOpPerKernel(),
               BufferAssignment(), MemoryPlan(reuse=options.memory_plan)]
    if options.tiling:
        tiles_for = options.matmul_tiles_for or (tuner.tiles_for if tuner is not None else None)
        passes.append(Schedule(options.target, matmul_tiles_for=tiles_for))
    return passes


def compile_graph(graph: Graph, options: Options | None = None):
    """Optimise a Kiln graph and JIT it. Returns (Executable, Report)."""
    from kiln.codegen.emitter import emit_module
    from kiln.codegen.lowering import lower
    from kiln.passes.manager import PassManager
    from kiln.runtime.executor import Executable

    options = options or Options()
    start = time.perf_counter()
    tuner = None
    if options.autotune and options.tiling and options.matmul_tiles_for is None:
        from kiln.passes.autotune import Autotuner

        tuner = Autotuner(options.target)
    pm = PassManager(build_passes(options, tuner), print_after_all=options.print_ir)
    pm.run(graph)

    def dump(title, text):
        print(f"// ----- {title} -----\n{text}")

    code = emit_module(graph, profile=options.profile)
    if options.dump_mlir:
        dump("MLIR from Kiln IR", code.text)
    module, ctx, lowering = lower(code, dump if options.dump_mlir else None,
                                  transform=options.fusion or options.tiling)
    exe = Executable(module, ctx, code.arg_spec, [a.shape for a in graph.args],
                     [r.shape for r in graph.results], graph.constants, entry=graph.name,
                     kernel_names=[k.name for k in graph.kernels] if options.profile else ())

    report = Report(passes=pm.stats, timings=pm.timings, lowering=lowering)
    report.kernels = [f"{k.name} {k.kind:<11} {' '.join(n.op for n in k.nodes)}" for k in graph.kernels]
    report.arena_bytes = graph.attrs.get("arena_bytes", 0)
    plan = pm.stats.get("memory-plan", {})
    report.naive_bytes = plan.get("naive_bytes", 0)
    report.lower_bound_bytes = plan.get("lower_bound_bytes", 0)
    report.autotune = tuner.log if tuner is not None else []
    report.compile_seconds = time.perf_counter() - start
    return exe, report


class CompiledModule:
    """Callable returned by ``kiln.compile``."""

    def __init__(self, exe, report, graph, param_tensors, single_output):
        self._exe = exe
        self.report = report
        self.graph = graph
        self._params = param_tensors     # (position, tensor) for unfrozen parameters
        self._single = single_output
        self._positions = [i for i, a in enumerate(graph.arg_attrs) if "param" not in a]

    def _args(self, inputs):
        if len(inputs) != len(self._positions):
            raise TypeError(f"expected {len(self._positions)} inputs, got {len(inputs)}")
        args = [None] * len(self.graph.args)
        for pos, t in zip(self._positions, inputs):
            args[pos] = t
        for pos, t in self._params:
            args[pos] = t
        return args

    def profile(self, *inputs, iters=20):
        """Per-kernel timings in ms. Compile with ``profile=True`` first."""
        if not self._exe.kernel_names:
            raise RuntimeError("compile with kiln.compile(..., profile=True) to profile kernels")
        times = self._exe.profile(*self._args(inputs), iters=iters)
        kinds = {k.name: k for k in self.graph.kernels}
        return [(name, kinds[name].kind, " ".join(n.op for n in kinds[name].nodes), ms)
                for name, ms in times]

    def __call__(self, *inputs):
        if len(inputs) != len(self._positions):
            raise TypeError(f"expected {len(self._positions)} inputs, got {len(inputs)}")
        args = [None] * len(self.graph.args)
        for pos, t in zip(self._positions, inputs):
            args[pos] = t
        for pos, t in self._params:
            args[pos] = t
        outs = self._exe(*args)
        return outs[0] if self._single else tuple(outs)


class _FnModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def compile(model, example_inputs, options: Options | None = None, **flags) -> CompiledModule:
    """Compile a model (or function) for inference on CPU.

    Example:
        compiled = kiln.compile(model, (torch.randn(32, 256),))
        y = compiled(x)
        print(compiled.report)

    Keyword flags (freeze, fusion, memory_plan, tiling, target, autotune, print_ir,
    dump_mlir, profile) override fields of
    ``options``.
    """
    from kiln.frontend.fx_import import import_exported

    options = options or Options()
    for key, value in flags.items():
        if not hasattr(options, key):
            raise TypeError(f"unknown option {key!r}")
        setattr(options, key, value)

    module = model if isinstance(model, torch.nn.Module) else _FnModule(model)
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            ep = torch.export.export(module, tuple(example_inputs)).run_decompositions()
            single = not isinstance(module(*example_inputs), (tuple, list))
    finally:
        module.train(was_training)

    graph = import_exported(ep)
    live = dict(module.named_parameters()) | dict(module.named_buffers())
    exe, report = compile_graph(graph, options)
    params = [(i, live[a["param"]].detach()) for i, a in enumerate(graph.arg_attrs) if "param" in a]
    return CompiledModule(exe, report, graph, params, single)