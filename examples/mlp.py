"""Compile a small MLP with Kiln and compare it against eager PyTorch.

Run with KILN_DUMP_IR=1 to see the MLIR at every stage of the pipeline.
"""

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

    # dynamic=False: Kiln only handles static shapes, so recompile per shape.
    compiled = torch.compile(model, backend=kiln.backend, dynamic=False)

    # Kiln only handles inference graphs for now, so disable autograd.
    with torch.no_grad():
        expected = model(x)
        actual = compiled(x)

    print("eager:\n", expected)
    print("kiln:\n", actual)
    print("match:", torch.allclose(expected, actual, atol=1e-5))
    print("stats:", kiln.stats)


if __name__ == "__main__":
    main()