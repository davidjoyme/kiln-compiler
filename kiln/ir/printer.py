# File: kiln/ir/printer.py
"""Textual form of the Kiln IR.

    graph @forward(%arg0: f32[2, 4]) -> (f32[2, 8]) {
      kernel @k0 matmul {tile = [2, 8]}
      %0 = const {name = "fc.weight"} : f32[4, 8]
      %1 = matmul %arg0, %0 {kernel = "k0"} : f32[2, 8]
      return %1
    }

The format round-trips through ``kiln.ir.parser.parse``, which is what lets
``kiln-opt`` run individual passes on hand-written IR for FileCheck tests.
"""

from __future__ import annotations


def format_attr(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(format_attr(x) for x in v) + "]"
    if isinstance(v, dict):
        return format_attrs(v)
    raise TypeError(f"cannot print attribute value {v!r}")


def format_attrs(attrs: dict) -> str:
    if not attrs:
        return ""
    return "{" + ", ".join(f"{k} = {format_attr(attrs[k])}" for k in attrs) + "}"


def print_graph(graph) -> str:
    names = {}
    for i, a in enumerate(graph.args):
        names[a] = f"%arg{i}"
    for i, n in enumerate(graph.nodes):
        names[n.out] = f"%{i}"

    def name(v):
        return names.get(v, "%<undefined>")

    args = []
    for a, attrs in zip(graph.args, graph.arg_attrs):
        s = f"{name(a)}: {a.type}"
        if attrs:
            s += " " + format_attrs(attrs)
        args.append(s)
    results = ", ".join(str(v.type) for v in graph.results)
    header = f"graph @{graph.name}({', '.join(args)}) -> ({results})"
    if graph.attrs:
        header += " attributes " + format_attrs(graph.attrs)
    lines = [header + " {"]

    for k in graph.kernels:
        line = f"  kernel @{k.name} {k.kind}"
        if k.attrs:
            line += " " + format_attrs(k.attrs)
        lines.append(line)

    for n in graph.nodes:
        operands = ", ".join(name(v) for v in n.inputs)
        line = f"  {name(n.out)} = {n.op}"
        if operands:
            line += " " + operands
        if n.attrs:
            line += " " + format_attrs(n.attrs)
        line += f" : {n.out.type}"
        lines.append(line)

    lines.append("  return " + ", ".join(name(v) for v in graph.results))
    lines.append("}")
    return "\n".join(lines)