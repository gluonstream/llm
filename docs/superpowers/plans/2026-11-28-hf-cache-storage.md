# HF Cache Storage and Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every GGUF model `llm` manages into `~/.cache/huggingface/hub/`, discover all complete GGUF sets already there, and make them first-class `llm` models.

**Architecture:** A stdlib `hfcache` module owns cache-root resolution, snapshot refs, split-set grouping, discovery, and profile synthesis. `config.load_profiles` resolves built-ins from cache coordinates and merges discovered profiles before user models and overrides. `llm migrate-cache` hardlinks legacy files into `blobs/` and writes `refs/` + `snapshots/`; `llm get` writes the same layout.

**Tech Stack:** Python 3.9+ standard library only; `unittest`; existing `llm` modules. No `huggingface_hub` import.

**Spec:** `docs/superpowers/specs/2026-11-28-hf-cache-storage-design.md`

## Global Constraints

- Python 3.9+, standard library only. No third-party imports.
- Avoid `X | Y` runtime annotations; use `Optional`/`Union` or no annotations.
- Tests run with `python3 -m unittest discover -s tests`; no test touches the real HF cache or runs a download.
- Cache root resolution: `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub`, `~/.cache/huggingface/hub`.
- Blob name is the file's SHA-256 (LFS sha); snapshot entries are relative symlinks to `../../blobs/<sha>`.
- Discovery skips `mmproj-*`, `mtp-*`, `*.part`, dotfiles, broken/zero-byte files, and incomplete split sets.
- User models always win alias/id collisions; discovered models never shadow built-ins.
- `LMM_NO_CACHE=1` disables discovery (used by tests).
- Commits use `feat:`, `fix:`, `test:`, or `docs:` prefixes.

## Review Focus

1. A cache with several snapshots/commits for one repo — must not list the same file twice or pick an older commit over `refs/main`.
2. An incomplete split set or a broken symlink — must be skipped, never offered as a runnable model.
3. A legacy migration across filesystems or with an existing blob — must not lose data, duplicate 36 GB, or leave a half-written snapshot.
4. Alias collisions between cache models, built-ins, and user models — user wins, built-ins never shadowed, deterministic suffixes.
5. `get` writing into a read-only or missing cache — must fail with an actionable message and leave no partial snapshot.

---

### Task 1: `llm/hfcache.py` — cache layout, discovery, profiles

**Files:**
- Create: `llm/hfcache.py`
- Create: `tests/test_hfcache.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `cache_root(env=None) -> Path`
  - `repo_dir(repo, root=None) -> Path`
  - `read_ref(repo, revision, root=None) -> Optional[str]`
  - `expected_snapshot(repo, filename, revision, root=None) -> Path`
  - `resolve_snapshot(repo, filename, revision, root=None) -> Optional[Path]`
  - `split_set(filename) -> Optional[tuple]` → `(stem, index, total)`
  - `group_ggufs(paths) -> list[dict]` → `{"entry": Path, "files": [Path], "size": int, "split": bool}`
  - `discover(root=None, env=None, sampling=None) -> dict` → `{profile_id: profile}`
  - `profile_for(repo, revision, group, root=None, sampling=None) -> dict`

- [ ] **Step 1: Write the failing tests**

`tests/test_hfcache.py` builds synthetic trees under a temp dir with real files and relative symlinks.

```python
def test_cache_root_precedence(self)
    # HF_HUB_CACHE wins; then HUGGINGFACE_HUB_CACHE; then HF_HOME/hub; then ~/.cache/huggingface/hub

def test_read_ref_and_expected_snapshot(self)
    # refs/main = "a"*40; expected path ends snapshots/<a*40>/q4/file.gguf

def test_split_set_parsing(self)
    # "x-00001-of-00002.gguf" -> ("x", 1, 2); "plain.gguf" -> None

def test_group_ggufs_complete_and_incomplete_split(self)
    # two shards -> one group with size sum; missing shard -> no group

def test_discover_prefers_ref_main_and_dedupes(self)
    # same filename in two snapshots -> one profile, path under refs/main commit

def test_discover_skips_mmproj_part_and_broken(self)
    # mmproj-x.gguf, y.gguf.part, broken symlink -> none

def test_discover_synthesizes_deterministic_ids_and_aliases(self)
    # alias from stem; split suffix removed from BOTH alias and id:
    # id == slug(repo + "-" + stem), never contains "-00001-of-00002"

def test_profile_for_shape(self)
    # id, alias, repo, filename (relative), revision, path, context 8192, sampling, source "cache"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_hfcache -v`
Expected: `ModuleNotFoundError: No module named 'llm.hfcache'`.

- [ ] **Step 3: Implement `llm/hfcache.py`**

`cache_root`: check env in order, `Path(value).expanduser().resolve()`; default `Path.home()/".cache/huggingface/hub"`.

`repo_dir`: `<root>/models--<repo.replace("/", "--")>`.

`read_ref`: `(repo_dir/refs/revision).read_text().strip()`; `None` on `OSError`/empty.

`expected_snapshot`:
- if `revision` matches `^[0-9a-f]{40}$` → commit = revision;
- else `commit = read_ref(...)`; if `None`, use `revision` as the directory name (best effort).
- return `repo_dir/snapshots/<commit>/<filename>`.

`resolve_snapshot`: `expected_snapshot(...).exists()` → return it; else scan `snapshots/*/<filename>` for an existing regular file, newest first; `None` otherwise.

`split_set`: regex `^(?P<stem>.*)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$` (case-insensitive); return `(stem, int(index), int(total))`.

`group_ggufs`:
- partition paths into split and plain;
- split groups keyed by `(parent, stem, total)`; keep only groups whose indices are exactly `1..total`; entry = index 1; size = sum;
- each plain path is its own group;
- return sorted by `str(entry)`.

`discover`:
- if `root/models--*` has no dirs → `{}`.
- for each repo dir sorted: main commit = `read_ref(repo, "main")`; snapshots in `[main] + sorted(other snapshots)`; walk each for `*.gguf`; skip `mmproj-`/`mtp-` prefixes, `.part`, dotfiles, and `not path.exists()`/`size == 0`; group only the first snapshot that yields each `(repo, relative filename)`.
- build profiles with `profile_for`; de-dupe ids by appending `-2`, `-3`; aliases likewise, but skip any id/alias already used in this discovery pass.

`profile_for`:
- `relative = group["entry"].relative_to(snapshot_dir)`; `stem = group["entry"].stem`; if split, strip `-?????-of-?????`.
- `alias = slug(stem)`, `id = slug(repo + "-" + stem)` — the same shard-free stem, so the id is stable regardless of how many shards the set has.
- return the profile shape from the spec; `context=8192`; `sampling = sampling or DEFAULT_SAMPLING` where `DEFAULT_SAMPLING` is a literal dict in `hfcache.py`. Task 2's `config` passes its own `DEFAULT_SAMPLING` into `discover`, avoiding a `config` ↔ `hfcache` import cycle.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_hfcache -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/hfcache.py tests/test_hfcache.py
git commit -m "feat: discover GGUF models in the Hugging Face cache"
```

---

### Task 2: Cache-aware profiles and sizes

**Files:**
- Modify: `llm/config.py` (built-ins, `DEFAULT_SAMPLING`, `load_profiles`, `reload_profiles`)
- Modify: `llm/cli.py` (`_local_size`)
- Modify: `tests/__init__.py` (set `LMM_NO_CACHE=1` before any `llm` import, so the suite never reads the real cache)
- Test: `tests/test_config.py`, `tests/test_cli.py`, `tests/test_hfcache.py`

**Interfaces:**
- Consumes: `llm.hfcache.cache_root`, `expected_snapshot`, `resolve_snapshot`, `discover`, `split_set`.
- Produces:
  - `config.DEFAULT_SAMPLING`
  - `config.HF_CACHE`
  - `load_profiles(path=None, builtin=None, warnings=None, discovered=None)`
  - `reload_profiles(...)` forwards `discovered`
  - `cli._local_size(profile)` sums only the matching split set

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
def test_builtin_profiles_resolve_from_cache(self)
    # temp HF cache with a qwen snapshot; BUILTIN_PROFILES resolve to it

def test_load_profiles_merges_discovered_before_user_models(self)
    # discovered "qwen2.5-coder..." appears; a user model with the same alias wins

def test_discovered_does_not_shadow_builtin(self)
    # a discovered alias equal to a builtin alias is skipped

def test_lmm_no_cache_disables_discovery(self)
    # env LMM_NO_CACHE=1 -> no discovered profiles

# tests/test_cli.py
def test_local_size_sums_only_the_shard_set(self)
    # snapshot with q4 shards and an unrelated q8 file -> q4 total only
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_config tests.test_cli -v`
Expected: assertions fail (built-ins still point at `~/.local/share`).

- [ ] **Step 3: Implement**

`tests/__init__.py`: add `os.environ.setdefault("LMM_NO_CACHE", "1")` so discovery is off for the whole suite unless a test passes `discovered=` explicitly. Tests that exercise discovery call `hfcache.discover(root=tmp)` directly.

`config.py`:
- `DEFAULT_SAMPLING = {"temperature": 0.7, "top_p": 1.0, "min_p": 0.01, "repeat_penalty": 1.0}`; cli imports it instead of redefining.
- `HF_CACHE = hfcache.cache_root()`.
- Built-ins: replace `path` with `repo`, `filename`, `revision`; keep `alias`, `context`, `sampling`; qwen uses `Qwen/Qwen2.5-7B-Instruct-GGUF`/`main`/shard 1; inkling uses `unsloth/Inkling-2-Mini-GGUF`/`main`; glm keeps its pinned commit.
- Helper `_resolve_builtin(profile)`:
  - `resolved = hfcache.resolve_snapshot(...)`; `expected = hfcache.expected_snapshot(...)` if missing;
  - set `path = str(resolved or expected)`.
- `load_profiles`:
  1. builtins copy, each `_resolve_builtin`;
  2. `discovered = hfcache.discover(root=HF_CACHE, sampling=DEFAULT_SAMPLING) if (discovered is None and not os.environ.get("LMM_NO_CACHE")) else (discovered or {})`;
  3. add discovered if its id and alias are unused;
  4. merge user models as today (user id replaces discovered; a user alias collision drops the discovered one);
  5. apply overrides if present (none yet);
  6. append `warnings`.
- `reload_profiles` forwards `discovered`.
- `selected_profile`/`resolve_model` unchanged.

`cli._local_size`:
- if `profile.get("filename")` or `"path"` ends in a split name: detect `hfcache.split_set(Path(path).name)`; sum `Path(path).parent.glob(stem + "-*-of-" + total + ".gguf")`; return the sum.
- else existing behavior.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_config tests.test_cli tests.test_hfcache -v`
Expected: all pass; existing 147 tests still pass.

- [ ] **Step 5: Commit**

```bash
git add llm/config.py llm/cli.py tests/test_config.py tests/test_cli.py tests/test_hfcache.py
git commit -m "feat: resolve profiles and split sizes from the HF cache"
```

---

### Task 3: `llm migrate-cache`

**Files:**
- Create: `llm/migrate.py`
- Modify: `llm/cli.py` (parser + `cmd_migrate_cache`)
- Test: `tests/test_migrate.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `llm.hfcache.cache_root`, `repo_dir`, `expected_snapshot`; `llm.hub.sha256_file`.
- Produces:
  - `KNOWN_MIGRATIONS` list of dicts with `source`, `repo`, `commit`, `files` (`filename`, `sha256`, `size`)
  - `plan(legacy_root=None, root=None) -> list[dict]`
  - `migrate(entry, root=None, dry_run=False, verify=False, keep_old=False) -> dict`
  - `cmd_migrate_cache(args)`

- [ ] **Step 1: Write the failing tests**

```python
def test_known_mapping_and_manifest_scan(self)
    # three known dirs + an extra manifest -> planned entries

def test_migrate_hardlinks_and_writes_layout(self)
    # temp source + temp cache -> blob content, snapshot symlink target ../../blobs/<sha>,
    # refs/main commit, source removed

def test_migrate_copy_fallback(self)
    # mock os.link to raise OSError -> shutil.copy2 used, source removed

def test_migrate_is_idempotent(self)
    # second run reports "already present", no duplicate

def test_migrate_dry_run_changes_nothing(self)
    # source and cache unchanged

def test_migrate_verify_detects_bad_sha(self)
    # --verify on a corrupted source fails, no snapshot written

def test_migrate_cli_output(self)
    # ["migrate-cache", "--dry-run"] prints a table with repo/commit/files
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_migrate -v`
Expected: `ModuleNotFoundError: No module named 'llm.migrate'`.

- [ ] **Step 3: Implement**

`KNOWN_MIGRATIONS`:
- qwen: source `~/.local/share/graphify-qwen2.5-7b-instruct`, repo `Qwen/Qwen2.5-7B-Instruct-GGUF`, commit `bb5d59e06d9551d752d08b292a50eb208b07ab1f`, files:
  - `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`, sha `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db`, size `3993201344`
  - `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`, sha `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a`, size `689872288`
- inkling: source `~/.local/share/graphify-inkling-2-mini`, repo `unsloth/Inkling-2-Mini-GGUF`, commit `8078d149286eec7193795413b9e0643da6d11fa1`, file `Inkling-2-Mini-UD-Q4_K_XL.gguf`, sha `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3`, size `17629760736`
- glm: source `~/.local/share/graphify-glm-4.7-flash`, repo `unsloth/GLM-4.7-Flash-GGUF`, commit `0d32489ecb9db6d2a4fc93bd27ef01519f95474d`, file `GLM-4.7-Flash-UD-Q4_K_XL.gguf`, sha `b0d4fbc1211f891b4cfbf2a497160bfe06a49412420068904d426b7a13f4ba7f`, size `17520169312`

`plan`:
- start from `KNOWN_MIGRATIONS`; for each `download-manifest.json` under `graphify-*/` not already covered, parse `url/revision/bytes/sha256` and add.
- keep entries whose source files exist; report missing sources as `missing`.

`migrate` per file:
1. blob = `repo_dir/blobs/<sha>`; `mkdir(parents=True)`.
2. if blob exists and size matches: reuse; if `verify`, compare sha.
3. else if `dry_run`: skip.
4. else `os.link(source, blob)`; on `OSError` `shutil.copy2`; remove source only after success.
5. snapshot = `expected_snapshot(repo, filename, commit)` (commit is pinned, so no ref needed); `mkdir(parents=True)`; `os.symlink(os.path.relpath(blob, snapshot.parent), snapshot)`.
6. write `refs/main` with the commit only when the mapping's `revision` is `main`.
7. after all files: if `keep_old` is false, `shutil.rmtree(source_dir)`. When deciding whether the directory is fully migrated, ignore the known metadata/lock files `download-manifest.json`, `download.lock`, `.DS_Store`, and any `*.part`; any other leftover keeps the directory and is reported.

`cmd_migrate_cache`: parser flags `--dry-run`, `--verify`, `--keep-old`; print `SOURCE → REPO @ commit` rows and a summary; return 0 unless an error.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_migrate tests.test_cli -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/migrate.py llm/cli.py tests/test_migrate.py tests/test_cli.py
git commit -m "feat: migrate legacy models into the HF cache"
```

---

### Task 4: `get` writes the cache; docs

**Files:**
- Modify: `llm/hub.py` (repo commit + blob download)
- Modify: `llm/cli.py` (`cmd_get`)
- Modify: `README.md`
- Modify: `llm/hfcache.py` (helper to register a downloaded profile)
- Test: `tests/test_hub.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `llm.hfcache.cache_root`, `repo_dir`, `expected_snapshot`.
- Produces:
  - `hub.repo_info(repo) -> tuple[str, list]` → `(commit, files)`
  - `hub.download_blob(repo, file, blob_dir, ...) -> Path`
  - `cmd_get` registers `repo`/`filename`/`revision`/`path`
  - `render.cache_line(root)`? optional

- [ ] **Step 1: Write the failing tests**

```python
def test_repo_info_returns_commit_and_files(self)
    # mocked API -> (sha, gguf files)

def test_download_blob_writes_sha_named_file(self)
    # mocked URL -> blobs/<sha>, no remote filename in the cache

def test_get_writes_snapshot_and_profile(self)
    # temp cache + mock HTTP -> models--.../snapshots/<commit>/<file> symlink,
    # refs/main, models.json profile with repo/filename/revision/path

def test_get_never_writes_local_share(self)
    # assert no path under ~/.local/share/llm is created
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_hub tests.test_cli -v`
Expected: assertions fail.

- [ ] **Step 3: Implement**

`hub.repo_info(repo)`:
- `GET /api/models/<repo>?blobs=true`; return `(data["sha"], gguf_files(...))`.
- keep `model_files(repo, revision)` as a thin wrapper for compatibility.

`hub.download_blob(repo, file, blob_dir, ...)`:
- target = `blob_dir/<sha256>`; reuse `download_file` machinery but write `.part` next to the blob and `os.replace` to the sha name; verify size/sha; return target.

`cmd_get`:
- `commit, files = hub.repo_info(repo)`; select the file; download to `repo_dir/blobs`;
- symlink `snapshots/<commit>/<filename>` → `os.path.relpath(blob, snapshot.parent)`;
- write `refs/<revision>` = commit;
- profile: `repo`, `filename`, `revision`, `path` (snapshot), `context` (`--context` or 8192), `sampling` (`DEFAULT_SAMPLING`);
- `config.register_model(profile)`;
- provenance manifest at `config.STATE/manifests/<id>.json`;
- never create `MODELS_DIR`; use `hardware.preflight(..., models_dir=hfcache.cache_root())` for the fit estimate.

README: document the cache root, `migrate-cache`, discovery, and `get`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_hub tests.test_cli -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/hub.py llm/cli.py llm/hfcache.py README.md tests/test_hub.py tests/test_cli.py
git commit -m "feat: download models into the HF cache"
```

---

### Task 5: Dedupe cache discovery against built-ins and user models

**Files:**
- Modify: `llm/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `load_profiles` (built-ins → discovered → user), `hfcache.discover`.
- Produces: discovery merge skips a discovered profile whose `(repo, filename)` or resolved `path` matches an already-merged profile.

- [ ] **Step 1: Write the failing test**

```python
def test_discovery_does_not_duplicate_a_builtin(self)
    # discovered qwen with the same repo/filename as the built-in qwen
    # -> only the built-in alias appears; no qwen2-5-7b-instruct-q4-k-m duplicate

def test_discovery_skips_same_resolved_path(self)
    # discovered profile pointing at the same file as a built-in/user model
    # -> skipped even when repo/filename metadata differs

def test_discovery_keeps_distinct_cache_models(self)
    # a different repo/file still appears
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_config -v`
Expected: duplicates appear, so the assertions fail.

- [ ] **Step 3: Implement**

In `load_profiles`, while adding discovered profiles, build a set of `(repo, filename)` and resolved `path` values from the built-ins and user models merged so far. Skip a discovered profile when either key matches; otherwise add it and register its keys. Keep the existing id/alias collision rules.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_config -v`
Expected: all pass; `llm list` shows exactly the three built-ins plus the two distinct cache models.

- [ ] **Step 5: Commit**

```bash
git add llm/config.py tests/test_config.py
git commit -m "fix: dedupe discovered cache models against known profiles"
```

---

### Task 6: Quiet expected dedupe warnings

**Files:**
- Modify: `llm/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `load_profiles` warning population.
- Produces: no warning for a discovered profile skipped because it duplicates an already-registered profile by `(repo, filename)` or resolved path; warnings remain for dropped user models and for discovered id/alias collisions that skip a *different* model.

- [ ] **Step 1: Write the failing test**

```python
def test_expected_dedupe_does_not_warn(self)
    # built-in qwen + discovered qwen with the same repo/filename
    # -> warnings list is empty

def test_dropped_user_model_still_warns(self)
    # user alias collides with a built-in -> one warning remains
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_config -v`
Expected: the dedupe warning is present, so the assertions fail.

- [ ] **Step 3: Implement**

Remove the `warnings.append(... "duplicate of an already-registered model")` path for identity dedupe (repo/filename/path). Keep the warnings for dropped user models and for discovered profiles skipped by id/alias collision.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_config -v`
Expected: all pass; `llm list` is quiet on a normal cache.

- [ ] **Step 5: Commit**

```bash
git add llm/config.py tests/test_config.py
git commit -m "fix: do not warn for expected cache dedupe"
```

---

### Task 7: `llm remove`

**Files:**
- Create: `llm/remove.py`
- Modify: `llm/cli.py` (parser + `cmd_remove`)
- Modify: `llm/config.py` (remove a user entry and any override)
- Test: `tests/test_remove.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `hfcache.repo_dir`, `hfcache.resolve_snapshot`, `hfcache.split_set`, `config.resolve_model`, `config.remove_model`, `service.running`, `service.identity`, `service.stop`, `render.human_size`.
- Produces:
  - `remove.plan(profile, root=None) -> dict` with repo, commit, files (`path`, `filename`, `sha256`, `size`), shared blobs, profile id/source.
  - `remove.remove(profile, root=None, dry_run=False, force=False) -> dict` with removed/deleted/freed/errors.
  - `config.remove_model(model_id, path=None) -> None` (delete the user entry and any override, preserving the rest).
  - `cmd_remove(args)` and the `remove` subcommand.

- [ ] **Step 1: Write the failing tests**

```python
def test_plan_identifies_split_set_and_shared_blobs(self)
    # temp cache with two quants; plan(q4 split) lists both shards, not q8

def test_remove_deletes_only_the_named_quant(self)
    # q4 split removed; q8 snapshot and blob remain; repo dir remains

def test_remove_keeps_blob_referenced_by_another_snapshot(self)
    # same blob linked from two snapshots; removing one keeps the blob

def test_remove_is_dry_run_by_default_without_force(self)
    # no force -> refused, nothing deleted, message names --force

def test_remove_builtin_requires_force_and_keeps_profile(self)
    # built-in file deleted with force; built-in profile still resolves and shows missing

def test_remove_user_model_deletes_entry_and_override(self)
    # models.json entry and override removed; other models preserved

def test_remove_stops_the_running_model(self)
    # running alias matches -> service.stop called before deletion

def test_remove_cli_output_and_exit_code(self)
    # ["remove", alias, "--force"] prints freed size; dry-run prints planned deletions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_remove -v`
Expected: `ModuleNotFoundError: No module named 'llm.remove'`.

- [ ] **Step 3: Implement**

`remove.plan(profile)`:
- require `path`; derive `repo`, `commit`/`revision`.
- if the filename is a split set, collect all shards from the same snapshot dir; otherwise the single file.
- resolve each snapshot path to its blob; mark a blob `shared` when any other symlink in `snapshots/*/**` resolves to it.
- return the file list, total size, and profile source (`built-in`, `user`, `cache`).

`remove.remove(profile, dry_run=False, force=False)`:
- if not `force` and not `dry_run`: return a refusal result (`needs_force=True`), deleting nothing.
- if the profile is the running model, `service.stop()` first.
- dry-run: return the plan with `would delete` entries and the freed total.
- otherwise: unlink each snapshot symlink; unlink a blob only when it has no other referrer; remove empty snapshot dirs; remove `refs/<revision>` only when its commit snapshot dir is gone; remove the repo dir only when it has no snapshots left.
- for a user model: `config.remove_model(id)`.
- for a built-in: keep the profile (files are now missing).
- return deleted paths, kept/shared blobs, freed bytes, and errors.

`config.remove_model(model_id)`:
- load the registry; `pop(models, model_id)`, `pop(overrides, model_id)`; save atomically; `reload_profiles()`.

`cmd_remove`:
- resolve the model; print the plan (files, sizes, shared blobs); require `--force` unless `--dry-run`; call `remove`; print freed size and any kept blobs; return 1 on errors.

Parser: `add("remove", cmd_remove, "delete a model's files from the HF cache")`, `model`, `--dry-run`, `--force`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_remove tests.test_cli -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/remove.py llm/cli.py llm/config.py tests/test_remove.py tests/test_cli.py
git commit -m "feat: add llm remove for cached models"
```
