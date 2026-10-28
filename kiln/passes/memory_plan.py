# File: kiln/passes/memory_plan.py
"""Static memory planning: pack every intermediate buffer into one arena.

After buffer assignment, each value that crosses a kernel boundary has its
own buffer. Kiln allocates all of them once, at load time, inside a single
block of memory, the *arena*, so a call does no allocation at all. This pass
decides where in the arena each buffer lives.

For inference everything is static: every shape is known, and the kernels
run in a fixed order. So the planner can compute each buffer's *live range*,
the span of kernels during which its contents are still needed:

    first = the kernel that writes it
    last  = the last kernel that reads it, directly or through a view
            (a transpose or slice of a buffer reads the buffer)

Two buffers whose live ranges don't overlap can share memory. Ranges are
closed intervals: a kernel that reads buffer A and writes buffer B needs both
at once, so A = [.., i] and B = [i, ..] overlap and never share.

Assigning offsets is then a 2-D packing problem (time on one axis, address
on the other), the same shape as register allocation with variable-size
registers. Finding the smallest arena is NP-hard in general, so Kiln uses the
greedy-by-size heuristic from TensorFlow Lite's arena planner:

    sort buffers by size, largest first
    for each buffer:
        place it at the lowest offset that doesn't collide with any
        already-placed buffer whose live range overlaps its own

Largest-first leaves the small buffers to fill the gaps between big ones.
The result is checked against a lower bound, the peak number of bytes live
at any single kernel: no plan can use less than that.
"""

from __future__ import annotations

from kiln.ir.core import Graph
from kiln.passes.buffers import ALIGNMENT, compute_layouts, kernel_inputs
from kiln.passes.manager import Pass


def align(nbytes):
    return (nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def live_ranges(graph: Graph):
    """{buffer value: [first, last]} in kernel indices, for every arena buffer."""
    layouts = compute_layouts(graph)
    kernel_index = {n: i for i, k in enumerate(graph.kernels) for n in k.nodes}
    ranges = {}
    for v, lay in layouts.items():
        if lay.base[0] == "arena" and lay.base[1] is v:
            i = kernel_index[v.node]
            ranges[v] = [i, i]
    for i, k in enumerate(graph.kernels):
        for v in kernel_inputs(graph, k):
            lay = layouts.get(v)
            if lay is not None and lay.base[0] == "arena":
                owner = lay.base[1]          # a view reads the buffer it views
                ranges[owner][1] = max(ranges[owner][1], i)
    return ranges


def overlaps(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def peak_live_bytes(ranges):
    """Lower bound on the arena: the most bytes live during any one kernel."""
    if not ranges:
        return 0
    last = max(r[1] for r in ranges.values())
    return max(sum(align(v.type.nbytes) for v, r in ranges.items() if r[0] <= t <= r[1])
               for t in range(last + 1))


class MemoryPlan(Pass):
    """Assign arena offsets. ``reuse=False`` gives every buffer its own slot."""

    name = "memory-plan"

    def __init__(self, reuse=True):
        self.reuse = reuse

    def run(self, graph: Graph):
        ranges = live_ranges(graph)
        # Largest first; ties broken by start time so the plan is deterministic.
        order = sorted(ranges, key=lambda v: (-v.type.nbytes, ranges[v][0]))
        placed = []   # (offset, size, live range)

        for v in order:
            size = align(v.type.nbytes)
            if self.reuse:
                busy = sorted((off, off + sz) for off, sz, r in placed if overlaps(r, ranges[v]))
                offset = 0
                for start, end in busy:            # first gap that fits
                    if offset + size <= start:
                        break
                    offset = max(offset, end)
            else:
                offset = sum(sz for _, sz, _ in placed)
            placed.append((offset, size, ranges[v]))
            v.node.attrs["offset"] = offset

        arena = max((off + sz for off, sz, _ in placed), default=0)
        graph.attrs["arena_bytes"] = arena
        return {
            "buffers": len(placed),
            "arena_bytes": arena,
            "naive_bytes": sum(sz for _, sz, _ in placed),
            "lower_bound_bytes": peak_live_bytes(ranges),
        }