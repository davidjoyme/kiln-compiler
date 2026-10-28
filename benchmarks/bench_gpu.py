# File: benchmarks/bench_gpu.py
"""Kiln's GPU backend (generated Triton) vs eager PyTorch and torch.compile, on one GPU.

Three tables:
  1. matmul TFLOP/s at GPT-2 small shapes: Kiln (default configs), Kiln
     autotuned, and cuBLAS through eager PyTorch, at matched precision;
  2. memory-bound fused ops (LayerNorm, softmax, bias + GELU): Kiln vs eager
     vs torch.compile (TorchInductor, which also generates Triton);
  3. a GPT-2 small transformer block at several sequence lengths: latency and
     peak GPU memory, Kiln vs eager vs torch.compile.

Tables 1 and 2 report GPU kernel time: many calls are captured in one CUDA
graph and the replay is timed, for Kiln, eager and Inductor alike, because
these kernels take microseconds, less than Python needs to launch them.
Table 3 reports the end-to-end latency of a call, as a user sees it; there
Kiln replays its whole launch sequence as a CUDA graph, so it's compared with
torch.compile(mode="reduce-overhead"), which does the same.

Sizes that don't fit in the GPU's *free* memory (with headroom) are skipped,
so it's safe to run next to other jobs, though their load will skew timings.

    python benchmarks/bench_gpu.py                 # exact fp32 (no TF32 on either side)
    python benchmarks/bench_gpu.py --tf32          # TF32 tensor cores for Kiln and PyTorch
    python benchmarks/bench_gpu.py --no-inductor   # skip torch.compile (slow to compile)
    python benchmarks/bench_gpu.py --smoke         # tiny sizes; also runs on the Triton interpreter
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from models.transformer import Block  # noqa: E402

import kiln  # noqa: E402
from kiln.gpu import device, interpreting  # noqa: E402
from kiln.gpu.runtime import kernel_seconds  # noqa: E402


def kernel_ms(fn, *xs):
    """GPU time per call (CUDA-graph replay of many calls), in ms; nan if it can't be captured."""
    try:
        with torch.no_grad():
            return kernel_seconds(lambda: fn(*xs)) * 1e3
    except Exception as err:   # noqa: BLE001
        print(f"   (couldn't time {getattr(fn, '__name__', type(fn).__name__)}: {str(err).splitlines()[0][:80]})")
        return float("nan")


def ms_per_call(fn, *xs, rounds=5, min_time=0.05):
    """Best-of-rounds milliseconds per call (CUDA events on a GPU)."""
    with torch.no_grad():
        for _ in range(3):
            fn(*xs)
        if device().type != "cuda":
            best = float("inf")
            for _ in range(rounds):
                t0 = time.perf_counter()
                fn(*xs)
                best = min(best, time.perf_counter() - t0)
            return best * 1e3
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*xs)
        end.record()
        torch.cuda.synchronize()
        calls = max(1, int(min_time * 1e3 / max(start.elapsed_time(end), 1e-3)))
        best = float("inf")
        for _ in range(rounds):
            start.record()
            for _ in range(calls):
                fn(*xs)
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end) / calls)
        return best


def fits(nbytes, headroom=0.5):
    if device().type != "cuda":
        return True
    free, _ = torch.cuda.mem_get_info()
    return nbytes < free * headroom


def peak_mib(fn, *xs):
    if device().type != "cuda":
        return float("nan")
    with torch.no_grad():
        fn(*xs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn(*xs)
        torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def inductor(fn, enabled, mode=None):
    if not enabled or device().type != "cuda":
        return None
    try:
        return torch.compile(fn, mode=mode)
    except Exception as err:   # noqa: BLE001
        print(f"  (torch.compile unavailable: {err})")
        return None


def matmul_table(args, dev):
    print("\n1. matmul (nn.Linear, 128 tokens unless noted): TFLOP/s of GPU kernel time")
    shapes = [("attention qkv", 128, 2304, 768), ("attention out", 128, 768, 768),
              ("MLP up", 128, 3072, 768), ("MLP down", 128, 768, 3072),
              ("MLP up, 2048 tokens", 2048, 3072, 768), ("decode, 1 token", 1, 3072, 768)]
    if args.smoke:
        shapes = [("small", 16, 64, 32)]
    print(f"   {'workload':<22}{'M x N x K':>16}{'kiln':>9}{'kiln tuned':>12}{'cuBLAS':>9}{'tuned/cuBLAS':>14}")
    for name, m, n, k in shapes:
        if not fits(4 * (m * k + k * n + 2 * m * n) * 2):
            print(f"   {name:<22} skipped: not enough free GPU memory")
            continue
        linear = torch.nn.Linear(k, n).eval().to(dev)
        x = torch.randn(m, k, device=dev)
        flops = 2 * m * n * k
        row = []
        for flags in ({}, {"autotune": True}):
            c = kiln.compile(linear, (x,), device="cuda", allow_tf32=args.tf32, cuda_graphs=False, **flags)
            with torch.no_grad():
                tol = 1e-2 if args.tf32 else 1e-3
                torch.testing.assert_close(c(x), linear(x), rtol=tol, atol=tol)
            row.append(flops / kernel_ms(c, x) / 1e9)
        eager = flops / kernel_ms(linear, x) / 1e9
        print(f"   {name:<22}{f'{m}x{n}x{k}':>16}{row[0]:>9.1f}{row[1]:>12.1f}{eager:>9.1f}{row[1] / eager:>13.2f}x")


def memory_bound_table(args, dev):
    print("\n2. memory-bound ops, fused: GPU milliseconds per call")
    rows, d = (8, 64) if args.smoke else (2048, 768)
    bias = torch.randn(4 * d, device=dev)
    ln = torch.nn.LayerNorm(d).eval().to(dev)
    work = [
        (f"LayerNorm [{rows}, {d}]", ln, torch.randn(rows, d, device=dev)),
        (f"softmax [12, {rows // 4}, {rows // 4}]", lambda t: F.softmax(t, -1),
         torch.randn(12, rows // 4, rows // 4, device=dev)),
        (f"bias + GELU [{rows}, {4 * d}]", lambda t: F.gelu(t + bias, approximate="tanh"),
         torch.randn(rows, 4 * d, device=dev)),
    ]
    print(f"   {'workload':<26}{'kiln':>9}{'eager':>9}{'inductor':>10}{'kiln vs eager':>15}")
    for name, fn, x in work:
        c = kiln.compile(fn, (x,), device="cuda", cuda_graphs=False)
        with torch.no_grad():
            torch.testing.assert_close(c(x), fn(x), rtol=1e-4, atol=1e-5)
        t_kiln, t_eager = kernel_ms(c, x), kernel_ms(fn, x)
        ind = inductor(fn, args.inductor)
        t_ind = kernel_ms(ind, x) if ind is not None else float("nan")
        print(f"   {name:<26}{t_kiln:>9.3f}{t_eager:>9.3f}{t_ind:>10.3f}{t_eager / t_kiln:>14.2f}x")


def block_table(args, dev):
    print("\n3. GPT-2 small transformer block (d_model 768, 12 heads): end-to-end ms and peak MiB per call")
    lengths = [16] if args.smoke else [128, 512, 1024, 2048]
    d, heads = (64, 4) if args.smoke else (768, 12)
    print(f"   {'tokens':>7}{'kiln ms':>9}{'eager ms':>10}{'inductor* ms':>14}{'kiln vs eager':>15}"
          f"{'kiln MiB':>10}{'eager MiB':>11}")
    for t in lengths:
        # attention scores dominate: ~2 x heads x t^2 floats, plus activations and weights
        if not fits(4 * (3 * heads * t * t + 16 * t * d) + 30 * 2**20):
            print(f"   {t:>7} skipped: not enough free GPU memory")
            continue
        model = Block(d, heads, t).eval().to(dev)
        x = torch.randn(t, d, device=dev)
        c = kiln.compile(model, (x,), device="cuda", autotune=not args.smoke, allow_tf32=args.tf32)
        with torch.no_grad():
            tol = 1e-2 if args.tf32 else 1e-3
            torch.testing.assert_close(c(x), model(x), rtol=tol, atol=tol)
        t_kiln, t_eager = ms_per_call(c, x), ms_per_call(model, x)
        ind = inductor(model, args.inductor, mode="reduce-overhead")
        try:
            t_ind = ms_per_call(ind, x) if ind is not None else float("nan")
        except Exception as err:   # noqa: BLE001
            print(f"   (torch.compile failed: {str(err).splitlines()[0][:80]})")
            t_ind = float("nan")
        # Kiln's arena is allocated once at compile time, so a call's peak alone would hide it:
        # count arena + per-call allocations, against eager's per-call peak (weights excluded on both)
        kiln_mib = c.report.arena_bytes / 2**20 + peak_mib(c, x)
        print(f"   {t:>7}{t_kiln:>9.3f}{t_eager:>10.3f}{t_ind:>14.3f}{t_eager / t_kiln:>14.2f}x"
              f"{kiln_mib:>10.1f}{peak_mib(model, x):>11.1f}")


def footnote():
    print("   * torch.compile(mode=\"reduce-overhead\"): Inductor's Triton kernels, replayed as a CUDA graph")
    print("   kiln MiB = planned arena + per-call allocations; eager MiB = peak allocated during a call")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tf32", action="store_true", help="TF32 tensor cores for Kiln and PyTorch")
    p.add_argument("--no-inductor", dest="inductor", action="store_false")
    p.add_argument("--smoke", action="store_true", help="tiny sizes (works on the Triton interpreter)")
    args = p.parse_args()
    if not torch.cuda.is_available() and not interpreting():
        sys.exit("No CUDA GPU found. To check correctness without one: TRITON_INTERPRET=1 pytest tests/test_gpu.py")
    torch.manual_seed(0)
    dev = device()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"{torch.cuda.get_device_name()}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free, "
              f"precision: {'TF32' if args.tf32 else 'fp32'} (Kiln and PyTorch alike)")
        if free < 0.8 * total:
            print("note: other processes are using this GPU; their load will skew timings")
    else:
        print("Triton interpreter on the CPU: checks the script runs; timings are meaningless")
    matmul_table(args, dev)
    memory_bound_table(args, dev)
    block_table(args, dev)
    footnote()


if __name__ == "__main__":
    main()