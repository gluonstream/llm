"""Remove a model's files from the Hugging Face cache."""
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


def _snapshot_dir(entry):
    """Return the ``snapshots/<commit>`` directory that contains ``entry``."""
    entry = Path(entry)
    for parent in entry.parents:
        if parent.parent.name == "snapshots":
            return parent
    return None


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


def plan(profile, root=None):
    """Describe the cache files that would be removed for ``profile``."""
    root = Path(hfcache.cache_root() if root is None else root)
    alias = profile.get("alias") or profile.get("id") or "model"
    path = profile.get("path")
    if not path:
        raise config.ModelError("%s has no cached path to remove" % alias)
    entry = Path(path)
    if not entry.exists():
        raise config.ModelError(
            "%s is not installed (no file at %s)" % (alias, entry))
    repo = profile.get("repo")
    if not repo:
        raise config.ModelError(
            "%s has no repo; cannot locate its cache" % alias)
    repo_dir = hfcache.repo_dir(repo, root)
    snapshot_dir = _snapshot_dir(entry)
    if snapshot_dir is None:
        raise config.ModelError(
            "%s is not under a cache snapshot: %s" % (alias, entry))
    commit = snapshot_dir.name

    parts = (hfcache.split_set(entry.name)
             or hfcache.split_set(profile.get("filename") or ""))
    if parts is not None:
        stem, _index, total = parts
        try:
            links = sorted(
                entry.parent.glob("%s-*-of-%05d.gguf" % (stem, total)),
                key=lambda item: item.name)
        except OSError:
            links = []
        if not links:
            raise config.ModelError(
                "no shards found for split set %s" % entry)
    else:
        links = [entry]

    referrers = _referrers(repo_dir)
    files = []
    freed = 0
    for link in links:
        link = Path(link)
        try:
            blob = link.resolve()
        except OSError:
            blob = link
        try:
            size = blob.stat().st_size
        except OSError:
            size = 0
        shared = len(referrers.get(blob, [])) > 1
        if not shared:
            freed += size
        try:
            filename = str(link.relative_to(snapshot_dir))
        except ValueError:
            filename = link.name
        files.append({
            "path": str(link),
            "filename": filename,
            "size": size,
            "shared": shared,
        })

    return {
        "repo": repo,
        "commit": commit,
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
    if alias and alias in actual:
        service.stop()


def remove(profile, root=None, apply=False):
    """Remove a model's cache files; a dry run unless ``apply`` is true."""
    root = Path(hfcache.cache_root() if root is None else root)
    result = plan(profile, root)
    result["dry_run"] = not apply
    result["deleted"] = []
    result["kept"] = []
    result["errors"] = []
    if not apply:
        return result

    _stop_if_running(profile)

    repo_dir = hfcache.repo_dir(result["repo"], root)
    snapshots = repo_dir / "snapshots"

    deleted = []
    kept = []
    errors = []
    freed = 0
    for item in result["files"]:
        link = Path(item["path"])
        is_link = link.is_symlink()
        blob = link.resolve() if is_link else link
        try:
            if is_link:
                link.unlink()
                deleted.append(str(link))
            elif link.exists():
                link.unlink()
                deleted.append(str(link))
        except OSError as exc:
            errors.append("could not remove %s: %s" % (link, exc))
            continue
        if item["shared"]:
            kept.append(str(blob))
            continue
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
        revision = profile.get("revision")
        if revision:
            ref = repo_dir / "refs" / str(revision)
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
