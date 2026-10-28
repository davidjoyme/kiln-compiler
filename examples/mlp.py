# File: examples/mlp.py
"""Compile a small MLP with Kiln and compare it against eager PyTorch.

    python examples/mlp.py              # result + compile report
    python examples/mlp.py --ir         # also print the Kiln IR after every pass
    python examples/mlp.py --mlir       # also print the generated MLIR
"""

import sys

import torch

import kiln


def main():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    ).eval()
    x = torch.randn(2, 4)

    compiled = kiln.compile(model, (x,), print_ir="--ir" in sys.argv, dump_mlir="--mlir" in sys.argv)

    with torch.no_grad():
        expected = model(x)
    actual = compiled(x)

    print("eager:\n", expected)
    print("kiln:\n", actual)
    print("match:", torch.allclose(expected, actual, atol=1e-5))
    print(compiled.report)


if __name__ == "__main__":
    main()