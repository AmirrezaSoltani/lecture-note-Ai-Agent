#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if command -v ruff >/dev/null 2>&1; then
  echo "[lint] running ruff"
  ruff check app tests
else
  echo "[lint] ruff not found, running stdlib syntax checks"
  "$PYTHON_BIN" -m compileall app tests >/dev/null
fi

echo "[lint] running pytest"
PYTHONPATH=. "$PYTHON_BIN" -m pytest -q
