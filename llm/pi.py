"""Sync llama.cpp context windows into a Pi coding-agent config file."""
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config


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
    """Return the Pi models path from environment or the platform default."""
    pi_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if pi_dir:
        return Path(pi_dir).expanduser() / "models.json"
    return Path.home() / ".pi" / "agent" / "models.json"


def read_config(path=None):
    path = config_path() if path is None else Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _local_provider(data):
    """Return the local llama.cpp provider dict, or None when absent."""
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return None
    wanted = config.URL.rstrip("/") + "/v1"
    for provider in providers.values():
        if not isinstance(provider, dict):
            continue
        if str(provider.get("baseUrl") or "").rstrip("/") == wanted:
            return provider
    local = providers.get("llamacpp-local")
    return local if isinstance(local, dict) else None


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
    provider = _local_provider(data)
    if provider is None:
        result.skipped.append("no local llamacpp provider")
        return result
    models = provider.get("models")
    if not isinstance(models, list):
        result.skipped.append("provider models is not a list")
        return result

    by_id = {str(entry["id"]): entry for entry in models
             if isinstance(entry, dict) and entry.get("id")}
    changed = False
    for alias, context in sorted(_alias_contexts(profiles).items()):
        entry = by_id.get(alias)
        if entry is None:
            models.append({"id": alias, "name": alias,
                           "contextWindow": context, "maxTokens": 2048})
            result.added += 1
            changed = True
            continue
        if entry.get("contextWindow") == context:
            result.unchanged += 1
            continue
        entry["contextWindow"] = context
        result.updated += 1
        changed = True

    if not changed or dry_run:
        return result
    result.backup = _write(path, data)
    return result


def mismatches(profiles, path=None):
    """Return ``(alias, contextWindow_or_None, llm_context)`` differences."""
    data = read_config(path)
    if data is None:
        return []
    provider = _local_provider(data)
    if provider is None:
        return []
    models = provider.get("models")
    if not isinstance(models, list):
        return []
    by_id = {str(entry["id"]): entry for entry in models
             if isinstance(entry, dict) and entry.get("id")}
    result = []
    for alias, context in sorted(_alias_contexts(profiles).items()):
        entry = by_id.get(alias)
        current = None
        if entry is not None and type(entry.get("contextWindow")) is int:
            current = entry["contextWindow"]
        if current != context:
            result.append((alias, current, context))
    return result


def unknown_entries(profiles, path=None):
    """Return local-provider model ids not present in ``profiles``."""
    data = read_config(path)
    if data is None:
        return []
    provider = _local_provider(data)
    if provider is None:
        return []
    models = provider.get("models")
    if not isinstance(models, list):
        return []
    known = set(_alias_contexts(profiles))
    result = []
    for entry in models:
        if isinstance(entry, dict) and entry.get("id"):
            model_id = str(entry["id"])
            if model_id not in known:
                result.append(model_id)
    return sorted(result)
