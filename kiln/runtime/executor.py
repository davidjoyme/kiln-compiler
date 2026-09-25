"""Runtime: JIT-compile a lowered module and call it with PyTorch tensors."""

import ctypes

import numpy as np
import torch
from mlir.execution_engine import ExecutionEngine
from mlir.ir import Context
from mlir.runtime import get_ranked_memref_descriptor

from kiln.frontend.fx_to_mlir import KilnProgram
from kiln.passes.pipeline import lower_to_llvm


def _memref_arg(array: np.ndarray):
    # The C interface takes a pointer to a memref descriptor, and invoke()
    # expects a pointer to each argument, hence the double pointer.
    return ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(array)))


class KilnExecutable:
    """A compiled graph: callable with the same tensors as the FX graph."""

    def __init__(self, program: KilnProgram, opt_level: int = 3, dump_ir: bool = False):
        self.program = program
        self.ctx = Context()  # must outlive the engine
        self.module = lower_to_llvm(program.mlir, self.ctx, dump_stages=dump_ir)
        self.engine = ExecutionEngine(self.module, opt_level=opt_level)

    def __call__(self, *inputs):
        # Keep every numpy array alive until the call returns.
        in_arrays = [
            np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32) for t in inputs
        ]
        out_arrays = [np.empty(shape, dtype=np.float32) for shape in self.program.output_shapes]
        args = [_memref_arg(a) for a in in_arrays + out_arrays]
        self.engine.invoke(self.program.func_name, *args)
        return tuple(torch.from_numpy(a) for a in out_arrays)