"""Hugging Face search, GGUF selection and downloads."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://huggingface.co"
USER_AGENT = "llm/1.0"
CHUNK = 1024 * 1024
class HubError(RuntimeError):
    def __init__(self, message, candidates=()):
        super().__init__(message)
        self.candidates = list(candidates)


def _headers():
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _get_json(url):
    request = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code in (401, 403):
            hint = " \u2014 set HF_TOKEN or choose a public repo"
        raise HubError("Hugging Face HTTP %s for %s%s" % (exc.code, url, hint))
    except urllib.error.URLError as exc:
        raise HubError("Could not reach Hugging Face (%s): %s" % (url, exc.reason))
    except ValueError:
        raise HubError("Hugging Face returned invalid JSON for %s" % url)


def parse_spec(spec):
    """Split 'owner/repo:selector' into (repo, selector-or-None)."""
    repo, _, selector = spec.rpartition(":")
    if not repo:
        repo, selector = spec, ""
    if "/" not in repo or repo.startswith("/") or repo.endswith("/"):
        raise HubError("Expected a Hugging Face repo like 'owner/name', got %r" % spec)
    selector = selector.strip() or None
    return repo, selector


def _is_gated(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "none", "no")
    return bool(value)


def _normalize_search(item):
    return {
        "id": item.get("id", ""),
        "downloads": int(item.get("downloads") or 0),
        "likes": int(item.get("likes") or 0),
        "gated": _is_gated(item.get("gated")),
        "tags": list(item.get("tags") or []),
    }


def search_models(query, limit=30):
    if not query.strip():
        raise HubError("Search query must not be empty")
    params = urllib.parse.urlencode({
        "search": query.strip(),
        "limit": str(min(100, max(limit, 1))),
        "sort": "downloads",
        "direction": "-1",
    })
    data = _get_json(BASE_URL + "/api/models?" + params)
    if not isinstance(data, list):
        raise HubError("Hugging Face search returned an unexpected response")
    return [_normalize_search(item) for item in data]


def _is_gguf(result):
    if "gguf" in str(result.get("id", "")).lower():
        return True
    return any("gguf" in str(tag).lower() for tag in result.get("tags") or [])


def filter_search(results, include_all=False):
    if include_all:
        return list(results), 0
    shown = [result for result in results if _is_gguf(result)]
    return shown, len(results) - len(shown)


def gguf_files(siblings):
    files = []
    for item in siblings or []:
        name = item.get("rfilename") or ""
        if not name.lower().endswith(".gguf"):
            continue
        if name.rsplit("/", 1)[-1].lower().startswith("mmproj-"):
            continue
        lfs = item.get("lfs") or {}
        size = lfs.get("size", item.get("size"))
        sha256 = lfs.get("sha256") or item.get("sha256")
        files.append({"name": name,
                      "size": int(size) if size is not None else None,
                      "sha256": sha256})
    return files


def model_files(repo, revision="main"):
    url = BASE_URL + "/api/models/" + urllib.parse.quote(repo, safe="/") + "?blobs=true"
    data = _get_json(url)
    if not isinstance(data, dict):
        raise HubError("Hugging Face returned an unexpected model response for %s" % repo)
    return gguf_files(data.get("siblings"))


def select_file(files, selector):
    if not files:
        raise HubError("No GGUF files found in this repo")
    if selector:
        query = selector.strip().lower()
        exact = [item for item in files
                 if item["name"].lower() in (query, query + ".gguf")]
        matches = exact or [item for item in files if query in item["name"].lower()]
        if not matches:
            raise HubError("No GGUF file matches %r; choices: %s"
                           % (selector, ", ".join(item["name"] for item in files)),
                           candidates=files)
        if len(matches) > 1:
            raise HubError("%r matches %d files; choose one with --file"
                           % (selector, len(matches)), candidates=matches)
        chosen = matches[0]
    else:
        if len(files) > 1:
            raise HubError("This repo has %d GGUF files; choose one with :QUANT or --file"
                           % len(files), candidates=files)
        chosen = files[0]
    if "-00001-of-" in chosen["name"]:
        raise HubError("Split GGUF sets are not supported yet: %s" % chosen["name"])
    return chosen


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_url(repo, filename, revision, base_url):
    return "%s/%s/resolve/%s/%s" % (
        base_url.rstrip("/"), urllib.parse.quote(repo, safe="/"),
        urllib.parse.quote(revision, safe=""),
        urllib.parse.quote(filename, safe=""))


def safe_destination(dest_dir, filename):
    """Resolve a remote filename inside dest_dir, refusing escapes."""
    relative = Path(filename)
    if (relative.is_absolute() or not relative.parts
            or any(part == ".." for part in relative.parts)):
        raise HubError("Unsafe model filename: %r" % filename)
    root = Path(dest_dir).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HubError("Unsafe model filename: %r" % filename)
    return candidate


def download_file(repo, file, dest_dir, base_url=BASE_URL, revision="main",
                  force=False, progress=None):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = file["name"]
    final = safe_destination(dest_dir, filename)
    part = safe_destination(dest_dir, filename + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    expected_size = file.get("size")
    expected_sha = file.get("sha256")

    if final.exists() and not force:
        ok = expected_size is None or final.stat().st_size == expected_size
        if ok and expected_sha:
            ok = sha256_file(final) == expected_sha
        if ok:
            return final
        raise HubError("%s already exists but does not match the remote file; "
                       "use --force to replace it" % final)
    if force:
        for path in (final, part):
            path.unlink(missing_ok=True)

    offset = part.stat().st_size if part.exists() else 0
    headers = _headers()
    if offset:
        headers["Range"] = "bytes=%d-" % offset
    request = urllib.request.Request(
        _download_url(repo, filename, revision, base_url), headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        hint = " \u2014 set HF_TOKEN or choose a public repo" if exc.code in (401, 403) else ""
        raise HubError("Hugging Face HTTP %s downloading %s%s"
                       % (exc.code, filename, hint))
    except urllib.error.URLError as exc:
        raise HubError("Could not reach Hugging Face: %s" % exc.reason)

    with response:
        status = response.getcode()
        if offset and status == 206:
            mode = "ab"
            done = offset
            length = response.headers.get("Content-Length")
            total = int(length) + offset if length else expected_size
        else:
            mode = "wb"
            done = 0
            length = response.headers.get("Content-Length")
            total = int(length) if length else expected_size
        started = time.monotonic()
        last_report = 0.0
        with part.open(mode) as output:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                output.write(block)
                done += len(block)
                now = time.monotonic()
                if progress and now - last_report >= 2:
                    last_report = now
                    progress(done, total, now - started)

    size = part.stat().st_size
    if expected_size is not None and size != expected_size:
        raise HubError("Downloaded %d bytes, expected %d; rerun to resume"
                       % (size, expected_size))
    if expected_sha:
        actual = sha256_file(part)
        if actual != expected_sha:
            part.unlink(missing_ok=True)
            raise HubError("Checksum mismatch for %s; run `lmm get` again" % filename)
    os.replace(part, final)
    return final


def write_manifest(dest_dir, repo, filename, revision, size, sha256):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "url": _download_url(repo, filename, revision, BASE_URL),
        "revision": revision,
        "bytes": size,
        "sha256": sha256,
        "repo": repo,
        "filename": filename,
    }
    path = dest_dir / "download-manifest.json"
    temporary = dest_dir / "download-manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, path)
    return path
