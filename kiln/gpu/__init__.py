# File: kiln/gpu/__init__.py
"""Kiln's GPU backend: Triton kernels generated from the optimised Kiln graph."""

from kiln.gpu.runtime import device, interpreting

__all__ = ["device", "interpreting"]