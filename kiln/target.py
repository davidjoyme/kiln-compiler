# File: kiln/target.py
"""The CPU Kiln compiles for: the host, unless overridden.

The scheduling cost model needs only a few numbers: how many f32 lanes a
vector register holds, how many vector registers there are, and the L1 and
L2 data cache sizes. They're read from /proc/cpuinfo and sysfs on Linux.

Override with KILN_TARGET=avx512 | avx2 | neon | generic to schedule for a
different vector width (the code is still compiled for the host CPU, so
this is for experiments, e.g. checking AVX2 schedules on an AVX-512 machine).
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Target:
    name: str
    vector_lanes: int        # f32 elements per vector register
    vector_registers: int
    l1_bytes: int
    l2_bytes: int

    def __str__(self):
        return (f"{self.name}: {self.vector_lanes} x f32 per vector, {self.vector_registers} "
                f"vector registers, L1d {self.l1_bytes // 1024} KiB, L2 {self.l2_bytes // 1024} KiB")


ISAS = {
    "avx512": ("x86-64 AVX-512", 16, 32),
    "avx2": ("x86-64 AVX2", 8, 16),
    "neon": ("AArch64 NEON", 4, 32),
    "generic": ("generic", 4, 16),
}


def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(("flags", "Features")):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _cache_bytes(level, types):
    base = "/sys/devices/system/cpu/cpu0/cache"
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return None
    for entry in entries:
        path = os.path.join(base, entry)
        try:
            with open(os.path.join(path, "level")) as f:
                lvl = int(f.read())
            with open(os.path.join(path, "type")) as f:
                typ = f.read().strip()
            with open(os.path.join(path, "size")) as f:
                size = f.read().strip()
        except (OSError, ValueError):
            continue
        if lvl == level and typ in types:
            mult = {"K": 1024, "M": 1024 * 1024}.get(size[-1], 1)
            return int(size.rstrip("KM")) * mult
    return None


def detect_isa():
    flags = _cpu_flags()
    if "avx512f" in flags:
        return "avx512"
    if "avx2" in flags:
        return "avx2"
    if platform.machine().lower() in ("aarch64", "arm64"):
        return "neon"
    return "generic"


@lru_cache(maxsize=None)
def host() -> Target:
    isa = os.environ.get("KILN_TARGET") or detect_isa()
    if isa not in ISAS:
        raise ValueError(f"KILN_TARGET must be one of {', '.join(ISAS)}, got {isa!r}")
    name, lanes, registers = ISAS[isa]
    l1 = _cache_bytes(1, ("Data", "Unified")) or 32 * 1024
    l2 = _cache_bytes(2, ("Unified", "Data")) or 1024 * 1024
    return Target(name, lanes, registers, l1, l2)