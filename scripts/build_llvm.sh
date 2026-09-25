#!/usr/bin/env bash
# Build LLVM + MLIR (with Python bindings) at the commit pinned in llvm-version.txt.
# Run from anywhere, inside the activated conda env:  ./scripts/build_llvm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_SRC="${LLVM_SRC:-$ROOT/llvm-project}"
LLVM_BUILD="${LLVM_BUILD:-$ROOT/build}"
COMMIT="$(tr -d '[:space:]' < "$ROOT/llvm-version.txt")"

# 1. Get the LLVM source at exactly the pinned commit (shallow fetch).
if [ ! -d "$LLVM_SRC/.git" ]; then
  git init -q "$LLVM_SRC"
  git -C "$LLVM_SRC" remote add origin https://github.com/llvm/llvm-project.git
fi
if [ "$(git -C "$LLVM_SRC" rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
  echo "Fetching llvm-project @ $COMMIT"
  git -C "$LLVM_SRC" fetch --depth 1 origin "$COMMIT"
  git -C "$LLVM_SRC" checkout -q FETCH_HEAD
fi

# 2. Python packages the MLIR bindings need.
python -m pip install -r "$LLVM_SRC/mlir/python/requirements.txt"

# 3. Configure. Use lld if it is installed: much faster, lighter linking.
EXTRA_FLAGS=()
if command -v ld.lld >/dev/null 2>&1; then
  EXTRA_FLAGS+=(-DLLVM_USE_LINKER=lld)
fi

cmake -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD" -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DLLVM_PARALLEL_LINK_JOBS="${LLVM_LINK_JOBS:-2}" \
  "${EXTRA_FLAGS[@]}"

# 4. Build.
ninja -C "$LLVM_BUILD"

echo
echo "LLVM/MLIR built in $LLVM_BUILD"
echo "Next: source scripts/env.sh"