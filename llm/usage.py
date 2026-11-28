"""Read llama.cpp token usage from the server log."""
import re
from pathlib import Path

TAIL_BYTES = 2 * 1024 * 1024

_N_TOKENS_RE = re.compile(r"n_tokens = (\d+)")
_PROMPT_EVAL_RE = re.compile(r"prompt eval time\b.*?/\s*(\d+)\s+tokens")


def _tail_text(path, tail_bytes):
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - tail_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def peak_tokens(log_path, model_path=None, tail_bytes=TAIL_BYTES):
    """Return the largest token count seen in the log tail, or None."""
    text = _tail_text(log_path, tail_bytes)
    if model_path is not None:
        marker = "loading model '%s'" % str(model_path)
        index = text.rfind(marker)
        if index >= 0:
            text = text[index + len(marker):]
    counts = [int(value) for value in _N_TOKENS_RE.findall(text)]
    counts.extend(int(value) for value in _PROMPT_EVAL_RE.findall(text))
    counts = [count for count in counts if count > 0]
    return max(counts) if counts else None
