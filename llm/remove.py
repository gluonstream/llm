"""Remove a model's files from the Hugging Face cache."""
import os
from pathlib import Path

from . import config, hfcache, service


def _source(profile):
    if profile.get("id") in config.BUILTIN_PROFILES:
        return "built-in"
    if profile.get("source") == "cache":
        return "cache"
    return "user"


def _choices(profiles):
    names = set()
    for profile_id, profile in profiles.items():
        names.add(str(profile_id))
        if profile.get("alias"):
            names.add(str(profile["alias"]))
    return sorted(names)


def resolve(name, profiles=None):
    """Resolve an exact model id or alias (case-insensitive) only."""
    profiles = config.PROFILES if profiles is None else profiles
    query = str(name).strip().lower()
    if not query:
        choices = _choices(profiles)
        raise config.ModelError(
            "No model name given; choose one of: " + ", ".join(choices),
            candidates=choices)
    matches = []
    for profile_id, profile in profiles.items():
        if query in (str(profile_id).lower(),
                     str(profile.get("alias") or "").lower()):
            matches.append(profile_id)
    if len(matches) == 1:
        return profiles[matches[0]]
    choices = _choices(profiles)
    raise config.ModelError(
        "No exact model %r; choose one of: %s" % (name, ", ".join(choices)),
        candidates=choices)


def _within(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _snapshot_dir(entry, snapshots):
    """Return the ``snapshots/<commit>`` directory containing ``entry``."""
    try:
        relative = entry.relative_to(snapshots)
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    return snapshots / relative.parts[0]


def _referrers(repo_dir):
    """Map each blob path to the snapshot symlinks that resolve to it."""
    referrers = {}
    snapshots = repo_dir / "snapshots"
    try:
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir():
                continue
            for link in snapshot.rglob("*"):
                if not link.is_symlink():
                    continue
                try:
                    target = link.resolve()
                except OSError:
                    continue
                referrers.setdefault(target, []).append(link)
    except OSError:
        pass
    return referrers


def _resolve_target(link, blobs):
    """Resolve a snapshot symlink to its blob, refusing out-of-cache targets.

    Returns ``None`` for a plain (non-symlink) snapshot entry, which is treated
    as its own data file.
    """
    link = Path(link)
    if not link.is_symlink():
        return None
    try:
        target = link.resolve()
    except (OSError, RuntimeError):
        raise config.ModelError("cannot resolve snapshot target: %s" % link)
    if not _within(target, blobs):
        raise config.ModelError(
            "snapshot %s points outside the cache blobs: %s" % (link, target))
    return target


def _model_relative_names(entry, snapshot_dir, profile):
    """Return the relative filename(s) of the model inside its snapshot dir."""
    relative_entry = entry.relative_to(snapshot_dir)
    parts = (hfcache.split_set(entry.name)
             or hfcache.split_set(profile.get("filename") or ""))
    if parts is None:
        return [relative_entry.as_posix()]
    stem, _index, total = parts
    return [(relative_entry.parent / ("%s-%05d-of-%05d.gguf"
                                      % (stem, index, total))).as_posix()
            for index in range(1, total + 1)]


def plan(profile, root=None):
    """Describe the cache files that would be removed for ``profile``."""
    root = Path(hfcache.cache_root() if root is None else root).resolve()
    alias = profile.get("alias") or profile.get("id") or "model"
    path = profile.get("path")
    if not path:
        raise config.ModelError("%s has no cached path to remove" % alias)
    repo = profile.get("repo")
    if not repo:
        raise config.ModelError(
            "%s has no repo; cannot locate its cache" % alias)

    repo_dir = hfcache.repo_dir(repo, root).resolve()
    snapshots = (repo_dir / "snapshots").resolve()
    blobs = (repo_dir / "blobs").resolve()

    # Locate the snapshot entry without following its own symlink, so a path
    # whose directory escapes the cache (or whose target escapes blobs) is
    # refused rather than followed.
    entry = Path(path)
    entry = entry.parent.resolve() / entry.name
    if not _within(entry, snapshots):
        raise config.ModelError(
            "%s path is outside the cache snapshots: %s" % (alias, entry))
    if not entry.is_file():
        raise config.ModelError(
            "%s is not installed (no file at %s)" % (alias, entry))

    revision = profile.get("revision")
    safe_revision = None
    if revision is not None:
        try:
            safe_revision = hfcache._safe_component(revision, "revision")
        except ValueError as exc:
            raise config.ModelError(str(exc))

    snapshot_dir = _snapshot_dir(entry, snapshots)
    if snapshot_dir is None or not snapshot_dir.is_dir():
        raise config.ModelError(
            "%s is not under a cache snapshot: %s" % (alias, entry))
    commit = snapshot_dir.name

    # The same model may live under several snapshot commits; collect every
    # snapshot directory that holds the same relative filename(s).
    rel_names = _model_relative_names(entry, snapshot_dir, profile)
    collected = []
    for snapshot in sorted(snapshots.iterdir(), key=lambda item: item.name):
        if not snapshot.is_dir():
            continue
        for rel in rel_names:
            candidate = snapshot / rel
            if not _within(candidate, snapshots):
                raise config.ModelError(
                    "candidate %s escapes the cache snapshots" % candidate)
            if not candidate.is_file():
                continue
            target = _resolve_target(candidate, blobs) or candidate
            try:
                size = target.stat().st_size
            except OSError:
                size = 0
            collected.append({
                "path": candidate, "rel": rel, "commit": snapshot.name,
                "target": target, "size": size,
            })

    if not collected:
        raise config.ModelError(
            "no cached files found for %s" % alias)

    # A blob is shared only when some symlink outside the removal set still
    # refers to it; copies of the model across snapshots do not count.
    referrers = _referrers(repo_dir)
    removal_counts = {}
    for item in collected:
        target = item["target"]
        removal_counts[target] = removal_counts.get(target, 0) + 1
    shared_targets = {
        target for target, count in removal_counts.items()
        if len(referrers.get(target, [])) > count
    }

    files = []
    freed = 0
    seen = set()
    for item in collected:
        shared = item["target"] in shared_targets
        files.append({
            "path": str(item["path"]),
            "filename": item["rel"],
            "commit": item["commit"],
            "size": item["size"],
            "shared": shared,
        })
        if not shared and item["target"] not in seen:
            seen.add(item["target"])
            freed += item["size"]

    return {
        "repo": repo,
        "commit": commit,
        "revision": safe_revision,
        "source": _source(profile),
        "files": files,
        "freed": freed,
    }


def _stop_if_running(profile):
    pid = service.running()
    if not pid:
        return
    actual = service.identity(pid)
    alias = profile.get("alias")
    if alias and "--alias %s" % alias in actual:
        service.stop()
        return
    path = profile.get("path")
    if path and (path in actual or os.path.realpath(path) in actual):
        service.stop()


def remove(profile, root=None, apply=False):
    """Remove a model's cache files; a dry run unless ``apply`` is true."""
    root = Path(hfcache.cache_root() if root is None else root).resolve()
    result = plan(profile, root)
    result["dry_run"] = not apply
    result["deleted"] = []
    result["kept"] = []
    result["errors"] = []
    if not apply:
        return result

    _stop_if_running(profile)

    repo_dir = hfcache.repo_dir(result["repo"], root).resolve()
    snapshots = (repo_dir / "snapshots").resolve()
    blobs = (repo_dir / "blobs").resolve()

    deleted = []
    kept = []
    errors = []
    freed = 0
    seen = set()
    for item in result["files"]:
        link = Path(item["path"])
        is_link = link.is_symlink()
        blob = _resolve_target(link, blobs) if is_link else link
        try:
            if is_link or link.exists():
                link.unlink()
                deleted.append(str(link))
        except OSError as exc:
            errors.append("could not remove %s: %s" % (link, exc))
            continue
        if item["shared"]:
            if blob not in seen:
                seen.add(blob)
                kept.append(str(blob))
            continue
        if blob in seen:
            continue
        seen.add(blob)
        if is_link:
            try:
                if blob.exists():
                    blob.unlink()
            except OSError as exc:
                errors.append("could not remove blob %s: %s" % (blob, exc))
                continue
            deleted.append(str(blob))
        freed += item["size"]

    # Prune now-empty directories up to (but not including) snapshots/.
    for item in result["files"]:
        directory = Path(item["path"]).parent
        while directory != snapshots and directory != directory.parent:
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent

    # Remove refs/<revision> and the repo dir only when no snapshots remain.
    try:
        remaining = any(snapshots.iterdir())
    except OSError:
        remaining = False
    if not remaining:
        revision = result["revision"]
        if revision:
            ref = repo_dir / "refs" / revision
            if ref.is_symlink() or ref.exists():
                try:
                    ref.unlink()
                    deleted.append(str(ref))
                except OSError as exc:
                    errors.append("could not remove %s: %s" % (ref, exc))
        # Only empty directories collapse; a shared blob keeps blobs/ (and the
        # repo) in place.
        for directory in (repo_dir / "refs", repo_dir / "snapshots",
                          repo_dir / "blobs", repo_dir):
            try:
                directory.rmdir()
            except OSError:
                pass

    if result["source"] in ("user", "built-in"):
        config.remove_model(profile["id"])

    result["deleted"] = deleted
    result["kept"] = kept
    result["errors"] = errors
    result["freed"] = freed
    return result
