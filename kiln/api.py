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
    print_ir: bool = False       # print the Kiln IR after every pass
    dump_mlir: bool = False      # print MLIR (and transform scripts) during lowering
    profile: bool = False        # emit per-kernel entry points for CompiledModule.profile()

    @classmethod
    def none(cls, **kw):
        """Every optimisation off: the baseline for ablations."""
        return cls(freeze=False, **kw)


@dataclass
class Report:
    passes: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    lowering: dict = field(default_factory=dict)
    kernels: list = field(default_factory=list)
    arena_bytes: int = 0
    compile_seconds: float = 0.0

    def __str__(self):
        lines = [f"Kiln: {len(self.kernels)} kernels, compiled in {self.compile_seconds:.2f}s"]
        for k in self.kernels:
            lines.append(f"  {k}")
        lines.append(f"  arena: {self.arena_bytes} bytes, "
                     f"heap allocations per call: {self.lowering.get('heap_allocs', 0)}")
        for name, stats in self.passes.items():
            lines.append(f"  {name}: {stats}")
        return "\n".join(lines)


def build_passes(options: Options):
    from kiln.passes.buffers import BufferAssignment, MemoryPlan
    from kiln.passes.freeze import ConstantFold, FreezeParameters
    from kiln.passes.partition import OneOpPerKernel
    from kiln.passes.simplify import CSE, DCE, Canonicalize

    passes = [Canonicalize(), CSE()]
    if options.freeze:
        passes += [FreezeParameters(), ConstantFold(), Canonicalize(), CSE()]
    passes += [DCE(), OneOpPerKernel(), BufferAssignment(), MemoryPlan()]
    return passes


def compile_graph(graph: Graph, options: Options | None = None):
    """Optimise a Kiln graph and JIT it. Returns (Executable, Report)."""
    from kiln.codegen.emitter import emit_module
    from kiln.codegen.lowering import lower
    from kiln.passes.manager import PassManager
    from kiln.runtime.executor import Executable

    options = options or Options()
    start = time.perf_counter()
    pm = PassManager(build_passes(options), print_after_all=options.print_ir)
    pm.run(graph)

    def dump(title, text):
        print(f"// ----- {title} -----\n{text}")

    code = emit_module(graph, profile=options.profile)
    if options.dump_mlir:
        dump("MLIR from Kiln IR", code.text)
    module, ctx, lowering = lower(code, dump if options.dump_mlir else None)
    exe = Executable(module, ctx, code.arg_spec, [a.shape for a in graph.args],
                     [r.shape for r in graph.results], graph.constants, entry=graph.name,
                     kernel_names=[k.name for k in graph.kernels] if options.profile else ())

    report = Report(passes=pm.stats, timings=pm.timings, lowering=lowering)
    report.kernels = [f"{k.name} {k.kind:<11} {' '.join(n.op for n in k.nodes)}" for k in graph.kernels]
    report.arena_bytes = graph.attrs.get("arena_bytes", 0)
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

    Keyword flags (freeze, print_ir, dump_mlir, profile) override fields of
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