# File: kiln/codegen/scalar.py
"""Scalar bodies of elementwise ops, emitted as arith/math dialect operations.

Transcendental functions are expanded here into polynomial approximations
built only from arithmetic, FMA, floor and bit manipulation. MLIR's
``math.exp`` would otherwise lower to a libm call per element, which LLVM
cannot vectorise; these expansions vectorise like any other arithmetic, which
made the fused softmax kernel about 4x faster during development.

  exp(x):  Cody-Waite range reduction x = n*ln2 + r, then a degree-6
           polynomial for exp(r) on [-ln2/2, ln2/2] (Cephes expf
           coefficients), then scaling by 2^n built directly in the exponent
           bits. Max relative error ~2 ulp; flushes to 0 below -87.3.
  tanh(x): sign(x) * (1 - e) / (1 + e) with e = exp(-2|x|), stable for large |x|.
  erf(x):  Abramowitz & Stegun 7.1.26, max absolute error 1.5e-7.
"""

from __future__ import annotations


def f32(x: float) -> str:
    if x == float("inf"):
        return "0x7F800000"
    if x == float("-inf"):
        return "0xFF800000"
    return repr(float(x))


class ScalarBuilder:
    """Emits scalar f32 code into a linalg.generic body."""

    def __init__(self, fresh, emit):
        self.fresh = fresh
        self.emit = emit
        self.consts = {}

    def const(self, x: float) -> str:
        key = f32(x)
        if key not in self.consts:
            name = self.fresh("c")
            self.emit(f"{name} = arith.constant {key} : f32")
            self.consts[key] = name
        return self.consts[key]

    def _op(self, text) -> str:
        name = self.fresh()
        self.emit(f"{name} = {text}")
        return name

    def add(self, a, b):
        return self._op(f"arith.addf {a}, {b} : f32")

    def sub(self, a, b):
        return self._op(f"arith.subf {a}, {b} : f32")

    def mul(self, a, b):
        return self._op(f"arith.mulf {a}, {b} : f32")

    def div(self, a, b):
        return self._op(f"arith.divf {a}, {b} : f32")

    def fma(self, a, b, c):
        return self._op(f"math.fma {a}, {b}, {c} : f32")

    def maximum(self, a, b):
        return self._op(f"arith.maximumf {a}, {b} : f32")

    def minimum(self, a, b):
        return self._op(f"arith.minimumf {a}, {b} : f32")

    def neg(self, a):
        return self._op(f"arith.negf {a} : f32")

    def abs(self, a):
        return self._op(f"math.absf {a} : f32")

    def copysign(self, a, b):
        return self._op(f"math.copysign {a}, {b} : f32")

    # -- transcendental expansions ----------------------------------------------------

    def exp(self, x):
        c = self.const
        xc = self._op(f"arith.minnumf {x}, {c(88.3762626647949)} : f32")
        xc = self._op(f"arith.maxnumf {xc}, {c(-87.3365478515625)} : f32")
        n = self._op(f"math.floor {self.fma(xc, c(1.44269504088896341), c(0.5))} : f32")
        r = self.sub(self.sub(xc, self.mul(n, c(0.693359375))), self.mul(n, c(-2.12194440e-4)))
        p = c(1.9875691500e-4)
        for coef in (1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1):
            p = self.fma(p, r, c(coef))
        y = self.add(self.fma(p, self.mul(r, r), r), c(1.0))
        ni = self._op(f"arith.fptosi {n} : f32 to i32")
        bias = self._op("arith.constant 127 : i32")
        shift = self._op("arith.constant 23 : i32")
        bits = self._op(f"arith.shli {self._op(f'arith.addi {ni}, {bias} : i32')}, {shift} : i32")
        scale = self._op(f"arith.bitcast {bits} : i32 to f32")
        result = self.mul(y, scale)
        under = self._op(f"arith.cmpf olt, {x}, {c(-87.3365478515625)} : f32")
        return self._op(f"arith.select {under}, {c(0.0)}, {result} : f32")

    def tanh(self, x):
        e = self.exp(self.mul(self.abs(x), self.const(-2.0)))
        t = self.div(self.sub(self.const(1.0), e), self.add(self.const(1.0), e))
        return self.copysign(t, x)

    def sigmoid(self, x):
        return self.div(self.const(1.0), self.add(self.const(1.0), self.exp(self.neg(x))))

    def erf(self, x):
        c = self.const
        ax = self.abs(x)
        t = self.div(c(1.0), self.fma(ax, c(0.3275911), c(1.0)))
        p = c(1.061405429)
        for coef in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):
            p = self.fma(p, t, c(coef))
        p = self.mul(p, t)
        e = self.exp(self.neg(self.mul(ax, ax)))
        y = self.sub(c(1.0), self.mul(p, e))
        return self.copysign(y, x)

    def gelu(self, x, approximate):
        c = self.const
        if approximate == "tanh":
            x3 = self.mul(self.mul(x, x), x)
            inner = self.mul(self.fma(x3, c(0.044715), x), c(0.7978845608028654))
            g = self.add(c(1.0), self.tanh(inner))
        else:
            g = self.add(c(1.0), self.erf(self.mul(x, c(0.7071067811865476))))
        return self.mul(self.mul(x, c(0.5)), g)

    # -- dispatch -------------------------------------------------------------------

    def apply(self, op: str, args: list, attrs: dict) -> str:
        if op in ("add", "sub", "mul", "div", "maximum", "minimum"):
            return getattr(self, op)(*args)
        if op == "neg":
            return self.neg(args[0])
        if op == "relu":
            return self.maximum(args[0], self.const(0.0))
        if op in ("exp", "tanh", "sigmoid", "erf"):
            return getattr(self, op)(args[0])
        if op == "gelu":
            return self.gelu(args[0], attrs.get("approximate", "none"))
        if op == "rsqrt":
            return self._op(f"math.rsqrt {args[0]} : f32")
        if op == "sqrt":
            return self._op(f"math.sqrt {args[0]} : f32")
        if op == "copy":
            return args[0]
        raise NotImplementedError(f"no scalar lowering for '{op}'")