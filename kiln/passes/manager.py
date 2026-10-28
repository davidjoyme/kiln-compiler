# File: kiln/passes/manager.py
"""Pass infrastructure: a Pass base class and a PassManager with timing and dumps."""

from __future__ import annotations

import time

from kiln.ir.core import Graph, verify


class Pass:
    """A transformation on a Kiln graph. Subclasses set ``name`` and implement run()."""

    name = "pass"

    def run(self, graph: Graph) -> dict | None:
        """Transform ``graph`` in place. May return a dict of statistics."""
        raise NotImplementedError


class PassManager:
    def __init__(self, passes, print_after_all=False, verify_each=True):
        self.passes = list(passes)
        self.print_after_all = print_after_all
        self.verify_each = verify_each
        self.timings: dict[str, float] = {}
        self.stats: dict[str, dict] = {}

    def run(self, graph: Graph) -> Graph:
        for p in self.passes:
            start = time.perf_counter()
            stats = p.run(graph)
            self.timings[p.name] = self.timings.get(p.name, 0.0) + time.perf_counter() - start
            if stats:
                self.stats[p.name] = stats
            if self.verify_each:
                verify(graph)
            if self.print_after_all:
                print(f"// ----- IR after {p.name} -----")
                print(graph)
        return graph