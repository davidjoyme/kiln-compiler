# File: kiln/gpu/runtime.py
"""GPU runtime: load the generated Triton module, own the device memory, launch.

Everything static is set up once, at load time:

  * the arena: one device allocation holding every intermediate buffer, at
    the offsets the memory planner chose (kiln/passes/memory_plan.py), so a
    call allocates nothing but its outputs;
  * the frozen weights, copied to the device once;
  * the launch list: for each kernel, its Triton function, grid and the
    device buffers it reads and writes.

Before allocating, the runtime checks free device memory and raises a clear
MemoryError (with the sizes involved) rather than failing inside CUDA.

CUDA graphs. Shapes are static and every buffer lives at a fixed address, so
the whole launch sequence can be captured once as a CUDA graph and replayed
with a single call: that removes Python and Triton launch overhead per kernel
(tens of microseconds each, which adds up when a GPU kernel itself takes
microseconds). Inputs are copied into static input buffers before replay and
outputs copied out after. This is what torch.compile's "reduce-overhead" mode
does; here it's on by default on real GPUs.

Without a GPU, Triton's interpreter (TRITON_INTERPRET=1) runs the same
kernels on CPU tensors, which is how Kiln's GPU tests run in CI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch


def interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


def device() -> torch.device:
    """Where GPU-backend tensors live: CUDA, or the CPU under the Triton interpreter."""
    if interpreting():
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Kiln's GPU backend needs a CUDA GPU (or TRITON_INTERPRET=1 to run "
                           "the generated kernels on the CPU for testing)")
    return torch.device("cuda")


def kernel_dir() -> Path:
    base = os.environ.get("KILN_CACHE_DIR") or os.path.join(Path.home(), ".cache", "kiln")
    return Path(base) / "triton"


def load_module(source: str):
    """Write the generated source to a file (Triton reads kernel source from it) and import it."""
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    path = kernel_dir() / f"kiln_{digest}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(source)
        tmp.replace(path)
    spec = importlib.util.spec_from_file_location(f"kiln_triton_{digest}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def check_memory(dev, needed_bytes, what):
    if dev.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info(dev)
    if needed_bytes > free:
        mib = lambda b: f"{b / 2**20:.1f} MiB"
        raise MemoryError(f"Kiln needs {mib(needed_bytes)} of GPU memory for {what}, but only "
                          f"{mib(free)} of {mib(total)} is free (other processes may be using the GPU). "
                          f"Try a smaller batch or sequence length, or free GPU memory.")


def kernel_seconds(fn, repeats=20, rounds=5):
    """GPU time per call of ``fn``, excluding CPU launch overhead.

    A GPT-2-sized matmul takes a few microseconds on a modern GPU, less than
    Python needs to launch it, so timing calls back to back measures the CPU,
    not the kernel. Instead, ``repeats`` calls are captured in one CUDA graph
    and the graph replay is timed with CUDA events. Without a GPU (the Triton
    interpreter) this falls back to wall-clock time.
    """
    import time

    if device().type != "cuda":
        fn()
        best = float("inf")
        for _ in range(rounds):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(rounds):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3 / repeats)
    return best


class GPUExecutable:
    """Runs one compiled graph on the GPU. Not reentrant: calls share one arena."""

    def __init__(self, program, arena_bytes, constants, input_shapes, output_shapes,
                 cuda_graphs=True):
        self.device = device()
        self.program = program
        self.module, self.source_path = load_module(program.source)
        self.input_shapes = [tuple(s) for s in input_shapes]
        self.output_shapes = [tuple(s) for s in output_shapes]

        used_consts = {key[1] for launch in program.launches for key in launch.bases if key[0] == "const"}
        const_bytes = sum(constants[name].nbytes for name in used_consts)
        out_bytes = sum(4 * int(np.prod(s)) for s in self.output_shapes)
        check_memory(self.device, arena_bytes + const_bytes + out_bytes,
                     f"the arena ({arena_bytes / 2**20:.1f} MiB), weights ({const_bytes / 2**20:.1f} MiB) "
                     f"and outputs")
        self.arena = torch.empty(max(arena_bytes, 4), dtype=torch.uint8, device=self.device).view(torch.float32)
        self.constants = {name: torch.from_numpy(np.ascontiguousarray(constants[name], np.float32)).to(self.device)
                          for name in used_consts}
        self.launches = [(getattr(self.module, l.name), l) for l in program.launches]
        self.arena_bytes = arena_bytes
        self.weight_bytes = const_bytes

        self.graph = None
        self.use_cuda_graphs = cuda_graphs and self.device.type == "cuda"

    # -- launching ------------------------------------------------------------------------

    def _run(self, inputs, outputs):
        for fn, launch in self.launches:
            args = []
            for key in launch.bases:
                if key[0] == "input":
                    args.append(inputs[key[1]])
                elif key[0] == "output":
                    args.append(outputs[key[1]])
                elif key[0] == "const":
                    args.append(self.constants[key[1]])
                else:
                    args.append(self.arena)
            options = {"num_warps": launch.num_warps}
            if launch.num_stages is not None and not interpreting():
                options["num_stages"] = launch.num_stages
            fn[launch.grid](*args, **options)

    def _prepare(self, inputs):
        out = []
        for t, shape in zip(inputs, self.input_shapes):
            if tuple(t.shape) != shape:
                raise ValueError(f"Kiln was compiled for input shape {shape}, got {tuple(t.shape)}")
            if t.device != self.device or t.dtype != torch.float32 or not t.is_contiguous():
                t = t.detach().to(self.device, torch.float32).contiguous()
            out.append(t)
        return out

    def _capture(self):
        """Record the launch sequence as a CUDA graph over static input/output buffers."""
        self.static_inputs = [torch.zeros(s, device=self.device) for s in self.input_shapes]
        self.static_outputs = [torch.empty(s, device=self.device) for s in self.output_shapes]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):          # warm up: compiles every Triton kernel
            for _ in range(2):
                self._run(self.static_inputs, self.static_outputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run(self.static_inputs, self.static_outputs)
        self.graph = graph

    def __call__(self, *inputs):
        inputs = self._prepare(inputs)
        if self.use_cuda_graphs:
            if self.graph is None:
                self._capture()
            for static, t in zip(self.static_inputs, inputs):
                static.copy_(t)
            self.graph.replay()
            return [o.clone() for o in self.static_outputs]
        outputs = [torch.empty(s, device=self.device) for s in self.output_shapes]
        self._run(inputs, outputs)
        return outputs