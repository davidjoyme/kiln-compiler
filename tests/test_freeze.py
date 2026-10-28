# File: tests/test_freeze.py
"""Freezing and constant folding."""

import numpy as np
import torch

from kiln.frontend.fx_import import import_exported
from kiln.ir import interpret, parse
from kiln.passes.freeze import ConstantFold, FreezeParameters
from kiln.passes.manager import PassManager
from kiln.passes.simplify import CSE, DCE, Canonicalize
from models.transformer import Block


def run(g, *passes):
    PassManager(list(passes)).run(g)
    return g


def test_linear_weight_transpose_is_folded():
    model = torch.nn.Linear(16, 32).eval()
    x = torch.randn(4, 16)
    g = import_exported(torch.export.export(model, (x,)).run_decompositions())
    assert any(n.op == "transpose" for n in g.nodes)

    run(g, FreezeParameters(), ConstantFold(), DCE())
    assert not any(n.op == "transpose" for n in g.nodes)
    assert len(g.args) == 1  # only the user input is left
    folded = g.constants["weight.transpose"]
    np.testing.assert_array_equal(folded, model.weight.detach().numpy().T)
    with torch.no_grad():
        np.testing.assert_allclose(interpret(g, x.numpy())[0], model(x).numpy(), rtol=1e-5, atol=1e-6)


def test_freezing_the_transformer_block_preserves_semantics():
    model = Block().eval()
    x = torch.randn(16, 64)
    g = import_exported(torch.export.export(model, (x,)).run_decompositions())
    pm = PassManager([Canonicalize(), CSE(), FreezeParameters(), ConstantFold(), Canonicalize(), CSE(), DCE()])
    pm.run(g)
    assert pm.stats["freeze"]["frozen"] == 13          # 12 parameters + the mask buffer
    assert pm.stats["const-fold"]["folded"] == 4       # one transpose per Linear
    with torch.no_grad():
        np.testing.assert_allclose(interpret(g, x.numpy())[0], model(x).numpy(), rtol=1e-4, atol=1e-5)


def test_folding_never_grows_constants():
    g = parse("""
    graph @f(%arg0: f32[64, 64]) -> (f32[64, 64]) {
      %0 = const {value = 2.0} : f32[]
      %1 = expand %0 {shape = [64, 64]} : f32[64, 64]
      %2 = mul %arg0, %1 : f32[64, 64]
      return %2
    }""")
    run(g, ConstantFold())
    assert any(n.op == "expand" for n in g.nodes)  # broadcasting a scalar stays a free view


def test_scalar_arithmetic_is_folded():
    g = parse("""
    graph @f(%arg0: f32[4]) -> (f32[4]) {
      %0 = const {value = 2.0} : f32[]
      %1 = const {value = 3.0} : f32[]
      %2 = mul %0, %1 : f32[]
      %3 = add %arg0, %2 : f32[4]
      return %3
    }""")
    run(g, ConstantFold())
    consts = [n.attrs.get("value") for n in g.nodes if n.op == "const"]
    assert consts == [6.0]