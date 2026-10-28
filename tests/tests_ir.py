# File: tests/test_ir.py
"""The Kiln IR: printing, parsing, verification and the reference interpreter."""

import numpy as np
import pytest

from kiln.ir import Graph, TensorType, interpret, parse, verify

SRC = """
graph @forward(%arg0: f32[2, 4], %arg1: f32[8, 4] {param = "w"}) -> (f32[2, 8]) {
  %0 = transpose %arg1 {perm = [1, 0]} : f32[4, 8]
  %1 = matmul %arg0, %0 : f32[2, 8]
  %2 = const {value = 0.0} : f32[]
  %3 = maximum %1, %2 : f32[2, 8]
  return %3
}
"""


def test_print_parse_roundtrip():
    g = parse(SRC)
    assert str(parse(str(g))) == str(g)


def test_interpret_matches_numpy():
    g = parse(SRC)
    x = np.random.randn(2, 4).astype(np.float32)
    (out,) = interpret(g, x)
    np.testing.assert_allclose(out, np.maximum(x @ g.constants["w"].T, 0), rtol=1e-6)


def test_builder_infers_types():
    g = Graph()
    x = g.add_arg(TensorType((3, 1)), "x")
    y = g.add_arg(TensorType((1, 5)), "y")
    s = g.add_node("add", [x, y])
    r = g.add_node("reduce_sum", [s], {"dims": [1]})
    g.results = [r]
    verify(g)
    assert s.shape == (3, 5) and r.shape == (3, 1)


def test_verifier_rejects_wrong_type():
    bad = SRC.replace("%1 = matmul %arg0, %0 : f32[2, 8]", "%1 = matmul %arg0, %0 : f32[2, 9]")
    with pytest.raises(ValueError, match="does not match"):
        parse(bad)


def test_parser_reports_undefined_values():
    with pytest.raises(ValueError, match="undefined"):
        parse(SRC.replace("%3 = maximum %1, %2", "%3 = maximum %9, %2"))