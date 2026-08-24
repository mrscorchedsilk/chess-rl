#!/usr/bin/env bash
# Build the native extension for THIS worktree and stage it into _pkg/ so the
# worktree shadows the shared venv's installed chess_rl_native.  The shared
# venv is never written to, so the live v2 lineage keeps its own pinned build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CHESS_RL_VENV:-$HOME/chess-rl/.venv}"
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
PB="$("$PY" -m pybind11 --cmakedir)"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE="$PY" -Dpybind11_DIR="$PB" "$@"
cmake --build "$ROOT/build" -j"$(nproc)"
mkdir -p "$ROOT/_pkg/chess_rl_native/third_party/chess-library/include"
cp "$ROOT/src/chess_rl_native/__init__.py" "$ROOT/_pkg/chess_rl_native/__init__.py"
cp "$ROOT/LICENSE" "$ROOT/_pkg/chess_rl_native/LICENSE"
cp "$ROOT/native/third_party/chess-library/LICENSE" "$ROOT/_pkg/chess_rl_native/third_party/chess-library/LICENSE"
cp "$ROOT/native/third_party/chess-library/include/chess.hpp" "$ROOT/_pkg/chess_rl_native/third_party/chess-library/include/" 2>/dev/null || true
cp "$ROOT/build/"_chess_rl_native*.so "$ROOT/_pkg/chess_rl_native/"
cp "$ROOT/native/third_party/chess-library/PROVENANCE.json" \
   "$ROOT/_pkg/chess_rl_native/third_party/chess-library/" 2>/dev/null || true
echo "staged -> $ROOT/_pkg/chess_rl_native"
