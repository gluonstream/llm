"""Sync the selected model into a Codex ``config.toml``.

Codex config is TOML with a single top-level model.  We edit the file with a
targeted line editor so comments, order, and unrelated sections (such as
``[marketplaces.*]`` and ``[plugins.*]``) survive byte-for-byte.
"""
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config

_LLAMACPP = "[model_providers.llamacpp]"
_KEY_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*(.*)$')
_MODEL_RE = re.compile(r'^\s*model\s*=')
_PROVIDER_RE = re.compile(r'^\s*model_provider\s*=')


@dataclass
class SyncResult:
    """Outcome of a sync run against the Codex config file."""
    path: Path
    updated: int = 0
    added: int = 0
    unchanged: int = 0
    skipped: list = field(default_factory=list)
    backup: Optional[Path] = None


def config_path():
    """Return the Codex config path from environment or the platform default."""
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def read_config(path=None):
    """Return the raw config text, or None when the file is missing/invalid."""
    path = config_path() if path is None else Path(path)
    try:
        return path.read_text()
    except (OSError, ValueError):
        return None


def _header_name(line):
    """Return a normalized ``[table]``/``[[array]]`` header, else None."""
    stripped = line.strip()
    if not stripped.startswith("["):
        return None
    if "#" in stripped:
        stripped = stripped.split("#", 1)[0].rstrip()
    return stripped if stripped.endswith("]") else None


def _unquote(value):
    """Strip a trailing comment and surrounding quotes from a TOML value."""
    in_string = False
    for index, char in enumerate(value):
        if char == '"':
            in_string = not in_string
        elif char == "#" and not in_string:
            value = value[:index]
            break
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _top_level_value(text, key):
    """Return the value of a top-level key (before the first section)."""
    for line in text.split("\n"):
        if _header_name(line) is not None:
            break
        match = _KEY_RE.match(line)
        if match and match.group(1) == key:
            return _unquote(match.group(2))
    return None


def _section_value(text, key):
    """Return the value of ``key`` within the llamacpp section, else None."""
    in_llamacpp = False
    for line in text.split("\n"):
        header = _header_name(line)
        if header is not None:
            in_llamacpp = (header == _LLAMACPP)
            continue
        if in_llamacpp:
            match = _KEY_RE.match(line)
            if match and match.group(1) == key:
                return _unquote(match.group(2))
    return None


def _section_keys(base_url):
    return [
        (re.compile(r'^\s*name\s*='), 'name = "llama.cpp"'),
        (re.compile(r'^\s*base_url\s*='), 'base_url = "%s"' % base_url),
        (re.compile(r'^\s*wire_api\s*='), 'wire_api = "responses"'),
        (re.compile(r'^\s*requires_openai_auth\s*='),
         'requires_openai_auth = false'),
        (re.compile(r'^\s*supports_websockets\s*='),
         'supports_websockets = false'),
    ]


def _edit(text, alias, base_url):
    """Return ``(new_text, section_added)`` for the given alias and URL."""
    lines = text.split("\n")

    # Top-level ``model``/``model_provider`` live before the first section.
    first_section = None
    for index, line in enumerate(lines):
        if _header_name(line) is not None:
            first_section = index
            break
    top_end = first_section if first_section is not None else len(lines)

    model_found = False
    provider_found = False
    for index in range(top_end):
        line = lines[index]
        if _MODEL_RE.match(line):
            new_line = 'model = "%s"' % alias
            if line != new_line:
                lines[index] = new_line
            model_found = True
        elif _PROVIDER_RE.match(line):
            new_line = 'model_provider = "llamacpp"'
            if line != new_line:
                lines[index] = new_line
            provider_found = True

    insert = []
    if not model_found:
        insert.append('model = "%s"' % alias)
    if not provider_found:
        insert.append('model_provider = "llamacpp"')
    if insert:
        at = first_section if first_section is not None else len(lines)
        lines[at:at] = insert

    # The ``[model_providers.llamacpp]`` section.
    section_index = None
    for index, line in enumerate(lines):
        if _header_name(line) == _LLAMACPP:
            section_index = index
            break

    if section_index is None:
        block = [
            _LLAMACPP,
            'name = "llama.cpp"',
            'base_url = "%s"' % base_url,
            'wire_api = "responses"',
            'requires_openai_auth = false',
            'supports_websockets = false',
        ]
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(block)
        return "\n".join(lines), True

    section_end = len(lines)
    for index in range(section_index + 1, len(lines)):
        if _header_name(lines[index]) is not None:
            section_end = index
            break

    for key_re, canonical in _section_keys(base_url):
        found = False
        for index in range(section_index + 1, section_end):
            if key_re.match(lines[index]):
                if lines[index] != canonical:
                    lines[index] = canonical
                found = True
                break
        if not found:
            lines.insert(section_end, canonical)
            section_end += 1

    return "\n".join(lines), False


def _write(path, text):
    """Back up path, then atomically rewrite it; return the backup path."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(path.name + ".bak-" + timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(text)
        if path.exists():
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return backup


def sync(profiles, selected_alias, path=None, dry_run=False):
    """Point Codex at the selected model; return a ``SyncResult``."""
    path = config_path() if path is None else Path(path)
    result = SyncResult(path=path)
    if not selected_alias:
        result.skipped.append("no selected model alias")
        return result
    text = read_config(path)
    if text is None:
        result.skipped.append("config missing")
        return result
    base_url = config.URL.rstrip("/") + "/v1"
    new_text, section_added = _edit(text, selected_alias, base_url)
    if new_text == text:
        result.unchanged = 1
        return result
    if dry_run:
        if section_added:
            result.added = 1
        else:
            result.updated = 1
        return result
    result.backup = _write(path, new_text)
    if section_added:
        result.added = 1
    else:
        result.updated = 1
    return result


def mismatches(selected_alias, path=None):
    """Return ``(field, actual, expected)`` differences for model/provider/url."""
    path = config_path() if path is None else Path(path)
    text = read_config(path)
    if text is None:
        return []
    base_url = config.URL.rstrip("/") + "/v1"
    result = []
    model = _top_level_value(text, "model")
    if model != selected_alias:
        result.append(("model", model, selected_alias))
    provider = _top_level_value(text, "model_provider")
    if provider != "llamacpp":
        result.append(("model_provider", provider, "llamacpp"))
    base = _section_value(text, "base_url")
    if base != base_url:
        result.append(("base_url", base, base_url))
    return result
