# File: kiln/ir/parser.py
"""Parser for the textual Kiln IR (see printer.py for the format).

Constants written without data (``const {name = "w"}``) get deterministic
pseudo-random data seeded from their name, so hand-written test IR can be
compiled and executed without shipping weight files.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

from kiln.ir.core import Graph, Kernel, Node, TensorType

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<comment>//[^\n]*)
      | (?P<value>%[A-Za-z0-9_.]+)
      | (?P<symbol>@[A-Za-z0-9_.]+)
      | (?P<string>"(?:[^"\\]|\\.)*")
      | (?P<number>[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+|inf|nan)(?![A-Za-z_]))
      | (?P<arrow>->)
      | (?P<ident>[A-Za-z_][A-Za-z0-9_.]*)
      | (?P<punct>[{}()\[\],:=])
    )""",
    re.VERBOSE,
)


class ParseError(ValueError):
    pass


def _tokenize(text):
    pos, out = 0, []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            line = text.count("\n", 0, pos) + 1
            raise ParseError(f"line {line}: unexpected character {text[pos]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "comment":
            continue
        out.append((kind, m.group(kind), text.count("\n", 0, m.start()) + 1))
    out.append(("eof", None, text.count("\n") + 1))
    return out


class _Parser:
    def __init__(self, text):
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self, k=0):
        return self.toks[self.i + k]

    def next(self):
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def expect(self, value=None, kind=None):
        k, v, line = self.next()
        if (value is not None and v != value) or (kind is not None and k != kind):
            want = value if value is not None else kind
            raise ParseError(f"line {line}: expected {want!r}, got {v!r}")
        return v

    def accept(self, value):
        if self.peek()[1] == value:
            self.i += 1
            return True
        return False

    # -- pieces -----------------------------------------------------------------

    def attr_value(self):
        k, v, line = self.next()
        if k == "number":
            if any(c in v for c in ".eEn") or v in ("inf", "-inf", "+inf"):
                return float(v)
            return int(v)
        if k == "string":
            return bytes(v[1:-1], "utf-8").decode("unicode_escape")
        if k == "ident" and v in ("true", "false"):
            return v == "true"
        if v == "[":
            items = []
            if not self.accept("]"):
                while True:
                    items.append(self.attr_value())
                    if self.accept("]"):
                        break
                    self.expect(",")
            return items
        if v == "{":
            self.i -= 1
            return self.attrs()
        raise ParseError(f"line {line}: bad attribute value {v!r}")

    def attrs(self):
        out = {}
        if self.peek()[1] != "{":
            return out
        self.next()
        if self.accept("}"):
            return out
        while True:
            key = self.expect(kind="ident")
            self.expect("=")
            out[key] = self.attr_value()
            if self.accept("}"):
                return out
            self.expect(",")

    def type(self):
        dtype = self.expect(kind="ident")
        self.expect("[")
        dims = []
        if not self.accept("]"):
            while True:
                dims.append(int(self.expect(kind="number")))
                if self.accept("]"):
                    break
                self.expect(",")
        return TensorType(tuple(dims), dtype)

    # -- graph ----------------------------------------------------------------

    def graph(self) -> Graph:
        self.expect("graph")
        g = Graph(self.expect(kind="symbol")[1:])
        env = {}
        self.expect("(")
        if not self.accept(")"):
            while True:
                name = self.expect(kind="value")
                self.expect(":")
                t = self.type()
                attrs = self.attrs()
                env[name] = g.add_arg(t, attrs.get("param", name[1:]), **attrs)
                if self.accept(")"):
                    break
                self.expect(",")
        self.expect("->")
        self.expect("(")
        if not self.accept(")"):
            while True:
                self.type()
                if self.accept(")"):
                    break
                self.expect(",")
        if self.accept("attributes"):
            g.attrs = self.attrs()
        self.expect("{")

        while True:
            k, v, line = self.peek()
            if v == "kernel":
                self.next()
                name = self.expect(kind="symbol")[1:]
                kind = self.expect(kind="ident")
                g.kernels.append(Kernel(name, kind, [], self.attrs()))
            elif v == "return":
                self.next()
                results = []
                if self.peek()[0] == "value":
                    while True:
                        results.append(self._lookup(env, self.expect(kind="value"), line))
                        if not self.accept(","):
                            break
                g.results = results
                self.expect("}")
                break
            elif k == "value":
                self._node(g, env)
            else:
                raise ParseError(f"line {line}: unexpected {v!r}")

        # Attach nodes to kernels named in their attributes.
        by_name = {k.name: k for k in g.kernels}
        for n in g.nodes:
            kname = n.attrs.get("kernel")
            if kname is not None:
                if kname not in by_name:
                    raise ParseError(f"node refers to undeclared kernel @{kname}")
                by_name[kname].nodes.append(n)
        for a, attrs in zip(g.args, g.arg_attrs):
            if "param" in attrs and attrs["param"] not in g.constants:
                g.constants[attrs["param"]] = _fake_data(attrs["param"], a.shape)
        return g

    def _lookup(self, env, name, line):
        if name not in env:
            raise ParseError(f"line {line}: use of undefined value {name}")
        return env[name]

    def _node(self, g, env):
        name = self.expect(kind="value")
        line = self.peek()[2]
        self.expect("=")
        op = self.expect(kind="ident")
        operands = []
        if self.peek()[0] == "value":
            while True:
                operands.append(self._lookup(env, self.expect(kind="value"), line))
                if not self.accept(","):
                    break
        attrs = self.attrs()
        self.expect(":")
        t = self.type()
        node = Node(op, operands, attrs, t)
        g.nodes.append(node)
        if op == "const" and "name" in attrs and attrs["name"] not in g.constants:
            g.constants[attrs["name"]] = _fake_data(attrs["name"], t.shape)
        env[name] = node.out


def _fake_data(name, shape):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return rng.standard_normal(shape).astype(np.float32) * 0.1


def parse(text: str) -> Graph:
    from kiln.ir.core import verify

    g = _Parser(text).graph()
    verify(g)
    return g