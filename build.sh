#!/usr/bin/env bash
# Build a single-file executable zipapp for the llm CLI.
set -euo pipefail

detect_os() {
    local value="${OSTYPE:-}"
    if [[ -z "$value" ]]; then
        case "$(uname -s)" in
            Darwin) value=darwin ;;
            Linux) value=linux-gnu ;;
        esac
    fi
    case "$value" in
        darwin*) OS=macos ;;
        linux-gnu*) OS=arch ;;
        *)
            echo "unsupported OS: ${value:-$(uname -s)} (supported: macOS, Arch Linux)" >&2
            exit 1
            ;;
    esac
}
detect_os

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${LMM_BUILD_DIR:-$ROOT/build}"
DIST_DIR="${LMM_DIST_DIR:-$ROOT/dist}"
STAGE="$BUILD_DIR/app"
ARTIFACT="$DIST_DIR/llm"

if ! command -v python3 >/dev/null 2>&1; then
    echo "build.sh: python3: command not found on PATH" >&2
    exit 1
fi

if [ "${LMM_SKIP_LLAMA_CPP_UPGRADE:-0}" != "1" ]; then
    if [ "$OS" = "macos" ] && command -v brew >/dev/null 2>&1; then
        if brew list --versions llama.cpp >/dev/null 2>&1; then
            echo "Updating Homebrew llama.cpp..."
            if ! brew upgrade llama.cpp; then
                echo "warning: brew upgrade llama.cpp failed; continuing." >&2
            fi
        else
            echo "Homebrew llama.cpp is not installed; skipping upgrade." >&2
        fi
    elif [ "$OS" = "arch" ]; then
        if command -v llama-server >/dev/null 2>&1; then
            echo "llama-server: $(llama-server --version 2>&1 | head -n 1)"
        else
            echo "llama-server not found; install llama.cpp (AUR, e.g. yay -S llama.cpp-cuda-git for CUDA or yay -S llama.cpp-git for CPU) or set LMM_LLAMA_SERVER." >&2
        fi
    fi
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
