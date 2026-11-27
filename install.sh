#!/usr/bin/env bash
# Install the single-file CLI produced by build.sh into ~/.local/bin.
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
DIST="${LMM_DIST_DIR:-$ROOT/dist}/llm"
DEST="${LMM_BIN_DIR:-$HOME/.local/bin}"

if [ ! -f "$DIST" ]; then
    echo "install.sh: $DIST not found; run ./build.sh first" >&2
    exit 1
fi

mkdir -p "$DEST"

for name in llm lmm; do
    target="$DEST/$name"
    if [ -d "$target" ] && [ ! -L "$target" ]; then
        echo "install.sh: $target is a directory; refusing to replace it" >&2
        exit 1
    fi
done

tmp="$(mktemp "$DEST/.llm.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
cp "$DIST" "$tmp"
chmod 755 "$tmp"
# mv descends into a symlink to a directory, so replace the link itself.
if [ -L "$DEST/llm" ]; then
    rm -f "$DEST/llm"
fi
mv -f "$tmp" "$DEST/llm"
trap - EXIT
if [ ! -f "$DEST/llm" ] || [ -L "$DEST/llm" ]; then
    echo "install.sh: failed to install $DEST/llm" >&2
    exit 1
fi

ln -sfn llm "$DEST/lmm"

if [ "$OS" = "arch" ] && ! command -v llama-server >/dev/null 2>&1; then
    echo "warning: llama-server not found; install llama.cpp (AUR) or set LMM_LLAMA_SERVER." >&2
fi

case ":$PATH:" in
    *":$DEST:"*) ;;
    *) echo "warning: $DEST is not on PATH" >&2 ;;
esac

echo "Installed $DEST/llm"
