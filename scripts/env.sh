# Point Python and your shell at the MLIR build.  Usage:  source scripts/env.sh
_KILN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_KILN_BUILD="${LLVM_BUILD:-$_KILN_ROOT/build}"

export PYTHONPATH="$_KILN_BUILD/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$_KILN_BUILD/bin:$PATH"

python -c "import mlir.ir" 2>/dev/null \
  && echo "MLIR Python bindings found in $_KILN_BUILD" \
  || echo "warning: could not import mlir; has the build finished?"
unset _KILN_ROOT _KILN_BUILD