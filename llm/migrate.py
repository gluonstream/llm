"""Migrate legacy graphify model files into the Hugging Face cache."""
import json
import os
import shutil
import urllib.parse
from pathlib import Path

from . import hfcache, hub

KNOWN_MIGRATIONS = [
    {
        "source": "~/.local/share/graphify-qwen2.5-7b-instruct",
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "commit": "bb5d59e06d9551d752d08b292a50eb208b07ab1f",
        "revision": "main",
        "files": [
            {
                "filename": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                "sha256": "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db",
                "size": 3993201344,
            },
            {
                "filename": "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
                "sha256": "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a",
                "size": 689872288,
            },
        ],
    },
    {
        "source": "~/.local/share/graphify-inkling-2-mini",
        "repo": "unsloth/Inkling-2-Mini-GGUF",
        "commit": "8078d149286eec7193795413b9e0643da6d11fa1",
        "revision": "main",
        "files": [
            {
                "filename": "Inkling-2-Mini-UD-Q4_K_XL.gguf",
                "sha256": "e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3",
                "size": 17629760736,
            },
        ],
    },
    {
        "source": "~/.local/share/graphify-glm-4.7-flash",
        "repo": "unsloth/GLM-4.7-Flash-GGUF",
        "commit": "0d32489ecb9db6d2a4fc93bd27ef01519f95474d",
        "revision": "0d32489ecb9db6d2a4fc93bd27ef01519f95474d",
        "files": [
            {
                "filename": "GLM-4.7-Flash-UD-Q4_K_XL.gguf",
                "sha256": "b0d4fbc1211f891b4cfbf2a497160bfe06a49412420068904d426b7a13f4ba7f",
                "size": 17520169312,
            },
        ],
    },
]


def legacy_root_default():
    return Path.home() / ".local" / "share"


def _finalize(entry):
    """Add resolved paths and presence flags to a planned entry."""
    entry = dict(entry)
    source = Path(entry["source"])
    files = []
    for item in entry.get("files", []):
        item = dict(item)
        item.setdefault("path", str(source / item["filename"]))
        item["present"] = Path(item["path"]).is_file()
        files.append(item)
    entry["files"] = files
    entry["status"] = ("ready" if files and all(item["present"] for item in files)
                       else "missing")
    return entry


def _manifest_entry(path, source):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    url_repo = url_revision = url_filename = None
    parts = [urllib.parse.unquote(part) for part
             in urllib.parse.urlsplit(str(data.get("url") or "")).path
             .strip("/").split("/")]
    if "resolve" in parts:
        index = parts.index("resolve")
        if len(parts) >= index + 3:
            url_repo = "/".join(parts[:index])
            url_revision = parts[index + 1]
            url_filename = "/".join(parts[index + 2:])

    repo = data.get("repo") or url_repo
    filename = data.get("filename") or url_filename
    revision = str(data.get("revision") or url_revision or "main")
    commit = str(data.get("commit") or revision)
    sha256 = data.get("sha256")
    size = data.get("bytes") or data.get("size")
    if not (repo and filename and sha256 and size is not None):
        return None
    try:
        size = int(size)
    except (TypeError, ValueError):
        return None
    return {
        "source": str(source),
        "repo": repo,
        "commit": commit,
        "revision": revision,
        "origin": "manifest",
        "files": [
            {"filename": filename, "sha256": sha256, "size": size,
             "path": str(Path(source) / filename)},
        ],
    }


def plan(legacy_root=None, root=None):
    """Describe the legacy models that can be moved into the cache."""
    base = (Path(legacy_root).expanduser() if legacy_root is not None
            else legacy_root_default())
    entries = []
    covered = set()
    for mapping in KNOWN_MIGRATIONS:
        name = Path(mapping["source"]).name
        source = base / name
        covered.add(name)
        entry = dict(mapping)
        entry["source"] = str(source)
        entry["origin"] = "known"
        entry["files"] = [dict(item) for item in mapping["files"]]
        entries.append(_finalize(entry))

    for manifest in sorted(base.glob("graphify-*/download-manifest.json")):
        if manifest.parent.name in covered:
            continue
        entry = _manifest_entry(manifest, manifest.parent)
        if entry is not None:
            entries.append(_finalize(entry))
    return entries


def _unrelated_files(source, migrated_paths):
    source = Path(source)
    if not source.is_dir():
        return []
    covered = {Path(path).resolve() for path in migrated_paths}
    leftovers = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "download-manifest.json" and path.parent == source:
            continue
        if path.resolve() not in covered:
            leftovers.append(path)
    return leftovers


def _write_ref(repo, commit, root):
    refs = hfcache.repo_dir(repo, root) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(commit)


def migrate(entry, root=None, dry_run=False, verify=False, keep_old=False):
    """Move one planned entry into the cache and remove its old directory."""
    root = Path(hfcache.cache_root() if root is None else root)
    source_dir = Path(entry.get("source", ""))
    repo = entry["repo"]
    commit = str(entry.get("commit") or entry.get("revision") or "")
    revision = str(entry.get("revision") or commit)
    migrated_paths = []
    results = []

    for item in entry.get("files", []):
        filename = item["filename"]
        expected_sha = item["sha256"]
        expected_size = int(item.get("size") or 0)
        source = Path(item.get("path") or (source_dir / filename))
        blob = hfcache.repo_dir(repo, root) / "blobs" / expected_sha
        record = {"filename": filename, "status": "missing",
                  "method": None, "message": ""}
        results.append(record)

        if blob.is_file():
            if blob.stat().st_size != expected_size:
                record["status"] = "error"
                record["message"] = ("existing blob has a different size: %s"
                                     % blob)
                continue
            if verify and hub.sha256_file(blob) != expected_sha:
                record["status"] = "error"
                record["message"] = "blob checksum mismatch: %s" % blob
                continue
            record["status"] = "already present"
            record["method"] = "reused"
            migrated_paths.append(str(source))
            continue

        if blob.exists():
            record["status"] = "error"
            record["message"] = "existing blob is not a file: %s" % blob
            continue

        if not source.is_file():
            record["message"] = "source file not found: %s" % source
            continue

        actual_size = source.stat().st_size
        if expected_size and actual_size != expected_size:
            record["status"] = "error"
            record["message"] = ("source has %d bytes, expected %d: %s"
                                 % (actual_size, expected_size, source))
            continue

        if dry_run:
            record["status"] = "would migrate"
            continue

        if verify and hub.sha256_file(source) != expected_sha:
            record["status"] = "error"
            record["message"] = "checksum mismatch for %s" % source
            continue

        blob.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, blob)
            record["method"] = "hardlink"
        except OSError:
            shutil.copy2(source, blob)
            record["method"] = "copy"

        snapshot = hfcache.expected_snapshot(repo, filename, commit, root=root)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.is_symlink() or snapshot.exists():
            if snapshot.is_dir() and not snapshot.is_symlink():
                shutil.rmtree(snapshot)
            else:
                snapshot.unlink()
        os.symlink(os.path.relpath(blob, snapshot.parent), snapshot)
        record["status"] = "migrated"
        migrated_paths.append(str(source))

    statuses = [record["status"] for record in results]
    if "error" in statuses:
        overall = "error"
    elif dry_run and "would migrate" in statuses:
        overall = "dry-run"
    elif "migrated" in statuses:
        overall = "migrated"
    elif statuses and all(status == "already present" for status in statuses):
        overall = "already present"
    else:
        overall = "missing"

    leftover = []
    complete = bool(statuses) and all(
        status in ("migrated", "already present") for status in statuses)
    if complete and not dry_run and not keep_old:
        leftover = _unrelated_files(source_dir, migrated_paths)
        if not leftover:
            shutil.rmtree(str(source_dir), ignore_errors=True)

    if (overall == "migrated" and not dry_run and revision == "main"):
        _write_ref(repo, commit, root)

    return {
        "source": str(source_dir),
        "repo": repo,
        "commit": commit,
        "status": overall,
        "files": results,
        "leftover": [str(path) for path in leftover],
    }
