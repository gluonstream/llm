"""Read the Hugging Face cache layout and synthesize model profiles."""
import os
import re
from pathlib import Path
from typing import Optional

DEFAULT_SAMPLING = {
    "temperature": 0.7,
    "top_p": 1.0,
    "min_p": 0.01,
    "repeat_penalty": 1.0,
}

SPLIT_RE = re.compile(
    r"^(?P<stem>.*)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SPLIT_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)


def cache_root(env=None) -> Path:
    """Return the HF hub cache directory, preferring the environment."""
    env = os.environ if env is None else env
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = env.get(key)
        if value:
            return Path(value).expanduser().resolve()
    home = env.get("HF_HOME")
    if home:
        return (Path(home).expanduser() / "hub").resolve()
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_dir(repo, root=None) -> Path:
    root = Path(cache_root() if root is None else root)
    return root / ("models--" + repo.replace("/", "--"))


def read_ref(repo, revision, root=None) -> Optional[str]:
    path = repo_dir(repo, root) / "refs" / revision
    try:
        commit = path.read_text().strip()
    except OSError:
        return None
    return commit or None


def expected_snapshot(repo, filename, revision, root=None) -> Path:
    revision = str(revision)
    if COMMIT_RE.fullmatch(revision):
        commit = revision
    else:
        commit = read_ref(repo, revision, root) or revision
    return repo_dir(repo, root) / "snapshots" / commit / filename


def _safe_component(value, label):
    text = str(value)
    if (not text or text in (".", "..") or "/" in text or "\\" in text
            or "\x00" in text):
        raise ValueError("Unsafe %s: %r" % (label, text))
    return text


def _safe_relative_filename(filename):
    relative = Path(str(filename))
    if (relative.is_absolute() or not relative.parts
            or any(part in ("", "..") for part in relative.parts)):
        raise ValueError("Unsafe model filename: %r" % filename)
    return relative


def snapshot_path(repo, filename, commit, root=None) -> Path:
    """Return snapshots/<commit>/<filename> without touching the filesystem."""
    commit = _safe_component(commit, "commit")
    relative = _safe_relative_filename(filename)
    return repo_dir(repo, root) / "snapshots" / commit / relative


def link_snapshot(repo, filename, commit, blob, root=None) -> Path:
    """Symlink a blob into snapshots/<commit>/<filename>; return the link."""
    snapshot = snapshot_path(repo, filename, commit, root)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.is_symlink() or snapshot.exists():
        if snapshot.is_dir() and not snapshot.is_symlink():
            raise ValueError("Snapshot path is a directory: %s" % snapshot)
        snapshot.unlink()
    os.symlink(os.path.relpath(blob, snapshot.parent), snapshot)
    return snapshot


def write_ref(repo, revision, commit, root=None) -> Path:
    """Write refs/<revision> = commit and return the ref file."""
    revision = _safe_component(revision, "revision")
    refs = repo_dir(repo, root) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    path = refs / revision
    path.write_text(str(commit))
    return path


def resolve_snapshot(repo, filename, revision, root=None) -> Optional[Path]:
    expected = expected_snapshot(repo, filename, revision, root)
    if expected.exists():
        return expected
    snapshots = repo_dir(repo, root) / "snapshots"
    try:
        candidates = [path for path in snapshots.glob("*") if path.is_dir()]
    except OSError:
        return None

    def modified(path):
        try:
            return path.stat().st_mtime
        except OSError:
            return 0

    for candidate in sorted(candidates, key=modified, reverse=True):
        path = candidate / filename
        if path.is_file():
            return path
    return None


def split_set(filename) -> Optional[tuple]:
    match = SPLIT_RE.match(Path(str(filename)).name)
    if not match:
        return None
    return (match.group("stem"), int(match.group("index")),
            int(match.group("total")))


def group_ggufs(paths) -> list:
    """Group plain GGUF files and complete multi-shard sets."""
    groups = {}
    plain = []
    for path in paths:
        path = Path(path)
        parts = split_set(path.name)
        if parts is None:
            plain.append(path)
            continue
        stem, index, total = parts
        groups.setdefault((path.parent, stem, total), []).append(
            (index, path))

    found = []
    for (parent, _stem, total), shards in groups.items():
        shards.sort(key=lambda item: item[0])
        indices = [index for index, _path in shards]
        if indices != list(range(1, total + 1)):
            continue
        files = [path for _index, path in shards]
        try:
            size = sum(path.stat().st_size for path in files)
        except OSError:
            continue
        found.append({
            "entry": files[0],
            "files": files,
            "size": size,
            "split": True,
        })

    for path in plain:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        found.append({
            "entry": path,
            "files": [path],
            "size": size,
            "split": False,
        })

    return sorted(found, key=lambda group: str(group["entry"]))


def _usable(path, snapshot):
    name = path.name.lower()
    if name.startswith("mmproj-") or name.startswith("mtp-"):
        return False
    if name.endswith(".part"):
        return False
    try:
        relative = path.relative_to(snapshot)
    except ValueError:
        relative = path
    if any(part.startswith(".") for part in relative.parts):
        return False
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _unique(base, taken):
    base = base or "model"
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = "%s-%d" % (base, suffix)
        suffix += 1
    return candidate


def _snapshot_location(repo, entry, root):
    """Return the snapshot directory containing entry and its relative path."""
    snapshots = repo_dir(repo, root) / "snapshots"
    try:
        relative = entry.relative_to(snapshots)
    except ValueError:
        relative = None
    if relative is not None and len(relative.parts) >= 2:
        snapshot = snapshots / relative.parts[0]
        return snapshot, entry.relative_to(snapshot)
    for parent in entry.parents:
        grandparent = parent.parent
        if grandparent is not None and grandparent.name == "snapshots":
            return parent, entry.relative_to(parent)
    snapshot = entry.parent
    return snapshot, entry.relative_to(snapshot)


def profile_for(repo, revision, group, root=None, sampling=None) -> dict:
    entry = Path(group["entry"])
    _snapshot, relative = _snapshot_location(repo, entry, root)
    stem = entry.stem
    if group.get("split"):
        stem = SPLIT_SUFFIX_RE.sub("", stem)
    model_id = _slug(repo + "-" + stem)
    return {
        "id": model_id,
        "alias": _slug(stem),
        "repo": repo,
        "filename": relative.as_posix(),
        "revision": revision,
        "path": str(entry),
        "context": 8192,
        "sampling": dict(sampling) if sampling else dict(DEFAULT_SAMPLING),
        "source": "cache",
    }


def downloaded_profile(repo, filename, revision, commit, model_id=None,
                       alias=None, root=None, context=8192, sampling=None):
    """Build the profile for a file `get` just placed in the cache."""
    snapshot = snapshot_path(repo, filename, commit, root)
    profile = {
        "repo": repo,
        "filename": Path(str(filename)).as_posix(),
        "revision": str(revision),
        "path": str(snapshot),
        "context": int(context),
        "sampling": dict(sampling) if sampling else dict(DEFAULT_SAMPLING),
    }
    if model_id is not None:
        profile["id"] = model_id
    if alias is not None:
        profile["alias"] = alias
    return profile


def discover(root=None, env=None, sampling=None) -> dict:
    """Return synthesized profiles for complete GGUF models in the cache."""
    root = Path(cache_root(env) if root is None else root)
    try:
        repos = sorted((path for path in root.iterdir()
                        if path.is_dir() and path.name.startswith("models--")),
                       key=lambda path: path.name)
    except OSError:
        return {}

    profiles = {}
    taken = set()
    for path in repos:
        repo = path.name[len("models--"):].replace("--", "/")
        if not repo:
            continue
        snapshots = path / "snapshots"
        main = read_ref(repo, "main", root)
        names = []
        if main and (snapshots / main).is_dir():
            names.append(main)
        try:
            names.extend(sorted(other.name for other in snapshots.iterdir()
                                if other.is_dir() and other.name != main))
        except OSError:
            pass

        seen = set()
        for name in names:
            snapshot = snapshots / name
            if not snapshot.is_dir():
                continue
            revision = "main" if main and name == main else name
            paths = [candidate for candidate
                     in sorted(snapshot.rglob("*.gguf"),
                               key=lambda item: str(item))
                     if _usable(candidate, snapshot)]
            for group in group_ggufs(paths):
                try:
                    relative = group["entry"].relative_to(snapshot)
                except ValueError:
                    continue
                filename = relative.as_posix()
                if filename in seen:
                    continue
                seen.add(filename)

                profile = profile_for(repo, revision, group, root=root,
                                      sampling=sampling)
                model_id = _unique(profile["id"], taken)
                taken.add(model_id)
                alias = _unique(profile["alias"], taken)
                taken.add(alias)
                profile["id"] = model_id
                profile["alias"] = alias
                profiles[model_id] = profile
    return profiles
