# File: kiln/runtime/executor.py
"""Runtime: JIT the lowered module and call it with minimal per-call overhead.

The compiled entry point takes memref descriptors (pointer + sizes + strides).
Everything that doesn't change between calls is prepared once at load time:
the descriptors for constants and the arena, and the packed argument array
the JIT'd function reads. A call then only patches the data pointers of the
input and output descriptors and jumps into native code through ctypes,
a few microseconds of overhead, instead of rebuilding descriptors and
converting tensors to NumPy arrays on every call.
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch
from mlir.execution_engine import ExecutionEngine

_DESCRIPTOR_TYPES = {}


def descriptor_type(rank):
    """ctypes layout of MLIR's ranked memref descriptor (void* for pointers)."""
    if rank not in _DESCRIPTOR_TYPES:
        fields = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p), ("offset", ctypes.c_int64)]
        if rank:
            fields += [("sizes", ctypes.c_int64 * rank), ("strides", ctypes.c_int64 * rank)]
        _DESCRIPTOR_TYPES[rank] = type(f"MemRef{rank}D", (ctypes.Structure,), {"_fields_": fields})
    return _DESCRIPTOR_TYPES[rank]


def make_descriptor(shape, address=0):
    desc = descriptor_type(len(shape))()
    desc.allocated = desc.aligned = address
    desc.offset = 0
    if shape:
        stride = 1
        for i in range(len(shape) - 1, -1, -1):
            desc.sizes[i] = shape[i]
            desc.strides[i] = stride
            stride *= shape[i]
    return desc


def aligned_buffer(nbytes, alignment=64):
    raw = np.zeros(nbytes + alignment, dtype=np.uint8)
    start = (-raw.ctypes.data) % alignment
    return raw[start:start + nbytes]


class Executable:
    """Native code for one Kiln graph. Not reentrant: calls share one arena."""

    def __init__(self, llvm_module, ctx, arg_spec, input_shapes, output_shapes, constants,
                 entry="forward", opt_level=3, kernel_names=()):
        self.ctx = ctx                      # the engine needs the context alive
        self.engine = ExecutionEngine(llvm_module, opt_level=opt_level)
        self._fn = self._lookup(entry)
        self.entry = entry
        self.kernel_names = list(kernel_names)
        self.input_shapes = [tuple(s) for s in input_shapes]
        self.output_shapes = [tuple(s) for s in output_shapes]

        self._keepalive = []
        self._inputs, self._outputs = {}, {}
        descriptors = []
        for kind, key in arg_spec:
            if kind == "input":
                desc = make_descriptor(self.input_shapes[key])
                self._inputs[key] = desc
            elif kind == "output":
                desc = make_descriptor(self.output_shapes[key])
                self._outputs[key] = desc
            elif kind == "const":
                array = np.ascontiguousarray(constants[key], dtype=np.float32)
                if array.ctypes.data % 64:
                    buf = aligned_buffer(array.nbytes).view(np.float32).reshape(array.shape)
                    buf[...] = array
                    array = buf
                self._keepalive.append(array)
                desc = make_descriptor(array.shape, array.ctypes.data)
            elif kind == "arena":
                arena = aligned_buffer(key)
                self._keepalive.append(arena)
                desc = make_descriptor((key,), arena.ctypes.data)
            else:
                raise ValueError(kind)
            descriptors.append(desc)

        # packed[i] points at a pointer to descriptor i (the C-interface ABI).
        self._descriptors = descriptors
        self._desc_ptrs = [ctypes.pointer(d) for d in descriptors]
        self._packed = (ctypes.c_void_p * len(descriptors))(
            *[ctypes.addressof(p) for p in self._desc_ptrs])
        self._input_desc = [self._inputs[i] for i in range(len(self.input_shapes))]
        self._output_desc = [self._outputs[i] for i in range(len(self.output_shapes))]

    def _lookup(self, name):
        address = self.engine.raw_lookup("_mlir_ciface_" + name)
        if not address:
            raise RuntimeError(f"entry point {name} not found")
        return ctypes.CFUNCTYPE(None, ctypes.c_void_p)(address)

    def profile(self, *inputs, iters=20):
        """Time each kernel separately (requires a module built with profile=True).

        Returns [(kernel name, milliseconds per call)].
        """
        import time

        self(*inputs)  # bind the input/output descriptors and warm up
        results = []
        for name in self.kernel_names:
            fn = self._lookup(f"{self.entry}__{name}")
            fn(self._packed)
            start = time.perf_counter()
            for _ in range(iters):
                fn(self._packed)
            results.append((name, (time.perf_counter() - start) / iters * 1e3))
        return results

    def __call__(self, *inputs):
        keep = []
        for desc, t, shape in zip(self._input_desc, inputs, self.input_shapes):
            if t.dtype != torch.float32 or not t.is_contiguous() or t.device.type != "cpu":
                t = t.detach().to("cpu", torch.float32).contiguous()
            if tuple(t.shape) != shape:
                raise ValueError(f"Kiln was compiled for input shape {shape}, got {tuple(t.shape)}")
            keep.append(t)
            ptr = t.data_ptr()
            desc.allocated = desc.aligned = ptr
        outputs = [torch.empty(shape, dtype=torch.float32) for shape in self.output_shapes]
        for desc, t in zip(self._output_desc, outputs):
            desc.allocated = desc.aligned = t.data_ptr()
        self._fn(self._packed)
        return outputs