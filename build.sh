#!/usr/bin/env bash
# Build a single-file executable zipapp for the llm CLI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${LMM_BUILD_DIR:-$ROOT/build}"
DIST_DIR="${LMM_DIST_DIR:-$ROOT/dist}"
STAGE="$BUILD_DIR/app"
ARTIFACT="$DIST_DIR/llm"

if ! command -v python3 >/dev/null 2>&1; then
    echo "build.sh: python3: command not found on PATH" >&2
    exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE" "$DIST_DIR"
rm -f "$ARTIFACT"

if [ "${LMM_SKIP_TESTS:-0}" != "1" ]; then
    echo "Running unit tests..."
    (cd "$ROOT" && python3 -m unittest discover -s "${LMM_TEST_DIR:-$ROOT/tests}")
fi

cp -R "$ROOT/llm" "$STAGE/llm"
find "$STAGE" -type d -name __pycache__ -prune -exec rm -rf {} +

cat > "$STAGE/__main__.py" <<'PY'
import sys

from llm.cli import main

if __name__ == "__main__":
    sys.exit(main())
PY

python3 -m compileall -b -q "$STAGE"
python3 -m zipapp -p "/usr/bin/env python3" -c -o "$ARTIFACT" "$STAGE"
chmod +x "$ARTIFACT"

state="$(mktemp -d)"
trap 'rm -rf "$state"' EXIT
LMM_STATE_DIR="$state" "$ARTIFACT" help >/dev/null

echo "Built $ARTIFACT"
