"""Sync llama.cpp context limits into an OpenCode config file."""
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SyncResult:
    """Outcome of a sync run against one config file."""
    path: Path
    updated: int = 0
    added: int = 0
    unchanged: int = 0
    skipped: list = field(default_factory=list)
    backup: Optional[Path] = None


def config_path():
    """Return the OpenCode config path from environment or the platform default."""
    explicit = os.environ.get("LMM_OPENCODE_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg).expanduser() / "opencode" / "opencode.json"
    return Path.home() / ".config" / "opencode" / "opencode.json"


def read_config(path=None):
    path = config_path() if path is None else Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _models_map(data):
    provider = data.get("provider")
    if not isinstance(provider, dict):
        return None
    llamacpp = provider.get("llamacpp")
    if not isinstance(llamacpp, dict):
        return None
    models = llamacpp.get("models")
    return models if isinstance(models, dict) else None


def _alias_contexts(profiles):
    """Map each profile's alias to its positive integer context."""
    result = {}
    for profile in profiles.values():
        if not isinstance(profile, dict):
            continue
        alias = str(profile.get("alias") or "").strip()
        context = profile.get("context")
        if alias and type(context) is int and context > 0:
            result[alias] = context
    return result


def _write(path, data):
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
            json.dump(data, out, indent=2)
            out.write("\n")
        if path.exists():
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return backup


def sync(profiles, path=None, dry_run=False):
    path = config_path() if path is None else Path(path)
    result = SyncResult(path=path)
    data = read_config(path)
    if data is None:
        result.skipped.append("config missing or invalid")
        return result
    models = _models_map(data)
    if models is None:
        result.skipped.append("provider.llamacpp.models missing")
        return result

    changed = False
    for alias, context in sorted(_alias_contexts(profiles).items()):
        entry = models.get(alias)
        if entry is None:
            models[alias] = {"name": alias, "limit": {"context": context}}
            result.added += 1
            changed = True
            continue
        if not isinstance(entry, dict):
            models[alias] = {"name": alias, "limit": {"context": context}}
            result.updated += 1
            changed = True
            continue
        limit = entry.get("limit")
        if not isinstance(limit, dict):
            entry["limit"] = {"context": context}
            result.updated += 1
            changed = True
            continue
        if limit.get("context") == context:
            result.unchanged += 1
            continue
        limit["context"] = context
        result.updated += 1
        changed = True

    if not changed or dry_run:
        return result
    result.backup = _write(path, data)
    return result


def mismatches(profiles, path=None):
    """Return ``(alias, opencode_context_or_None, llm_context)`` differences."""
    data = read_config(path)
    if data is None:
        return []
    models = _models_map(data)
    if models is None:
        return []
    result = []
    for alias, context in sorted(_alias_contexts(profiles).items()):
        entry = models.get(alias)
        current = None
        if isinstance(entry, dict):
            limit = entry.get("limit")
            if isinstance(limit, dict) and type(limit.get("context")) is int:
                current = limit["context"]
        if current != context:
            result.append((alias, current, context))
    return result


def unknown_entries(profiles, path=None):
    """Return ``llamacpp`` model aliases not present in ``profiles``."""
    data = read_config(path)
    if data is None:
        return []
    models = _models_map(data)
    if models is None:
        return []
    known = set(_alias_contexts(profiles))
    return [alias for alias in sorted(models) if alias not in known]
