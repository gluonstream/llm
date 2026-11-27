#!/usr/bin/env bash
# Fetch, build, and install the latest llm from source.
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
if [ "$OS" = "macos" ]; then
    default_dir="/private/tmp/llm-update"
else
    default_dir="/tmp/llm-update"
fi
export LMM_UPDATE_DIR="${LMM_UPDATE_DIR:-$default_dir}"
cd "$ROOT"
exec python3 -m llm.update "$@"
