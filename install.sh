#!/usr/bin/env bash
# Install the single-file CLI produced by build.sh into ~/.local/bin.
set -euo pipefail

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

case ":$PATH:" in
    *":$DEST:"*) ;;
    *) echo "warning: $DEST is not on PATH" >&2 ;;
esac

echo "Installed $DEST/llm"
