# LLM — HF cache storage and discovery

Date: 2026-11-28
Status: approved (direction agreed by the human)

## Intent

`llm` currently keeps models in custom directories (`~/.local/share/graphify-*`,
`~/.local/share/llm/models`). Unsloth GUI, `huggingface_hub`, and the user's
other tools use `~/.cache/huggingface/hub/`. The result is duplicate downloads
and models `llm` cannot see.

Goal: one model store in the standard Hugging Face cache. `llm` migrates the
legacy files, discovers every complete GGUF set already in the cache, and makes
them selectable, startable, and sizeable without re-downloading.

## Goals

- Resolve the cache root from `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`,
  `HF_HOME/hub`, then `~/.cache/huggingface/hub`.
- Discover complete GGUF sets in `models--*/snapshots/*/` (including subdirs),
  group split shards, skip `mmproj-*`, `.part`, and broken symlinks.
- Synthesize a deterministic profile for every discovered model so it appears
  in `list`, `status`, `use`, `apply`, and `context`.
- `llm migrate-cache` moves the three legacy models into the cache layout using
  their known repo/commit/SHA, hardlinking on the same filesystem, then removes
  the old directories.
- Built-in profiles stop hardcoding `~/.local/share/graphify-*`; they store
  `repo`/`filename`/`revision` and resolve `path` from the cache.
- `llm get` downloads into the same layout (`blobs`, `refs`, `snapshots`).
- `_local_size` sums only the matching split set, never every GGUF in a
  snapshot directory.

## Non-goals

- Downloading or managing non-GGUF models (safetensors, MLX, whisper).
- Reading or writing Unsloth GUI configuration.
- Deleting cache entries the user did not ask to migrate.
- Reworking the context/GPU features already planned; this is a storage layer.
- Multi-file non-GGUF model loading.

## Cache layout

```
<root>/models--<org>--<repo>/
  blobs/<sha256>                 # actual bytes
  refs/<revision>                # commit SHA, e.g. refs/main
  snapshots/<commit>/<path>      # symlink to ../../blobs/<sha256>
```

`<root>` resolution, first match:

1. `HF_HUB_CACHE`
2. `HUGGINGFACE_HUB_CACHE`
3. `HF_HOME` + `/hub`
4. `~/.cache/huggingface/hub`

All values expand `~` and are absolute.

Snapshot resolution for a profile:

- `revision` that is a 40-char hex string → that commit.
- otherwise read `refs/<revision>`; if missing, use the only snapshot directory
  or the newest one (by directory mtime) and report that the ref was missing.
- `path` = `snapshots/<commit>/<filename>`.

## Discovery

For each `models--<org>--<repo>` directory:

1. Choose candidate snapshots: the `refs/main` commit first, then every other
   snapshot directory sorted by name. Process in that order and dedupe by
   `(repo, relative filename)` so older commits do not duplicate a model.
2. Walk each snapshot for `*.gguf`, skipping:
   - names starting with `mmproj-` or `mtp-`;
   - `*.part` and dotfiles;
   - files whose resolved target does not exist or is 0 bytes.
3. Group split sets by `(parent, stem, total)` from
   `^(.*)-(\d{5})-of-(\d{5})\.gguf$`. A group is complete only when indices
   `1..total` are all present; otherwise skip it.
4. A split group is one model whose entry path is the `00001` shard and whose
   size is the sum of the group. A non-split file is one model.
5. Derive:
   - `filename` = relative path inside the snapshot;
   - `id` = `slug(repo + "-" + filename-without-shard-suffix)`;
   - `alias` = `slug(stem-without-shard-suffix)`;
   - on collision with a built-in/user profile or another discovered model,
     append `-2`, `-3`, … deterministically (sorted by repo then filename).
6. Synthesized profile:
   ```json
   {
     "id": "...",
     "alias": "...",
     "repo": "org/repo",
     "filename": "q4/file-00001-of-00002.gguf",
     "revision": "main",
     "path": "/.../snapshots/<commit>/q4/file-00001-of-00002.gguf",
     "context": 8192,
     "sampling": {"temperature": 0.7, "top_p": 1.0, "min_p": 0.01, "repeat_penalty": 1.0},
     "source": "cache"
   }
   ```
   Discovered profiles are read-only derivations; context overrides from
   `models.json` still apply by id.

## Built-in profiles

Built-ins keep curated aliases/sampling/context but store cache coordinates:

- qwen2.5-7b: `Qwen/Qwen2.5-7B-Instruct-GGUF`, `main`,
  `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`.
- inkling-2-mini: `unsloth/Inkling-2-Mini-GGUF`, `main`,
  `Inkling-2-Mini-UD-Q4_K_XL.gguf`.
- glm-4.7-flash: `unsloth/GLM-4.7-Flash-GGUF`,
  `0d32489ecb9db6d2a4fc93bd27ef01519f95474d`,
  `GLM-4.7-Flash-UD-Q4_K_XL.gguf`.

`load_profiles` resolves a built-in `path` from `repo`/`filename`/`revision`
when the file exists; when it does not, it keeps the expected snapshot path so
`doctor` can report it missing. A migrated legacy file is therefore visible
immediately.

`DEFAULT_SAMPLING` moves into `config` so discovery and `get` share it.

## Migration

`llm migrate-cache [--dry-run] [--verify] [--keep-old]`:

Known mappings (from the manifests/API):

| source | repo | commit | files |
|---|---|---|---|
| `~/.local/share/graphify-qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | `bb5d59e06d9551d752d08b292a50eb208b07ab1f` | q4_k_m shards 1-2 |
| `~/.local/share/graphify-inkling-2-mini` | `unsloth/Inkling-2-Mini-GGUF` | `8078d149286eec7193795413b9e0643da6d11fa1` | UD-Q4_K_XL |
| `~/.local/share/graphify-glm-4.7-flash` | `unsloth/GLM-4.7-Flash-GGUF` | `0d32489ecb9db6d2a4fc93bd27ef01519f95474d` | UD-Q4_K_XL |

Also scan `~/.local/share/graphify-*/download-manifest.json` for
`url/revision/bytes/sha256` and migrate any file not in the known table.

Per file:

1. `blob = <root>/models--org--repo/blobs/<sha256>`.
2. If `blob` exists: verify size (and SHA with `--verify`); reuse it.
3. Else hardlink source → blob (`os.link`); on `OSError` (cross-device),
   `shutil.copy2`; remove the source only after the link/copy succeeds. With
   hardlinks the 36 GB is not duplicated.
4. Create `snapshots/<commit>/<filename>` as a relative symlink to the blob.
5. Write `refs/<revision>` = commit (for `main`; pinned commits need no ref).
6. Remove the old `graphify-*` directory when every file in it migrated and
   `--keep-old` is absent.
7. Idempotent: rerunning reports `already present` and does nothing.

`--dry-run` prints the table and changes nothing. `--verify` hashes the source
and the blob (slow, 36 GB).

## `llm get`

`get` writes the same layout:

1. Fetch the repo commit (`/api/models/<repo>` `sha`; existing
   `model_files` gains the repo commit).
2. Download each selected file to `blobs/<sha256>` via a `.part` temporary,
   verifying size and SHA.
3. Symlink into `snapshots/<commit>/<filename>`.
4. Write `refs/<revision>` = commit.
5. Register a profile with `repo`, `filename`, `revision`, `path`, `context`,
   `sampling`.
6. Keep a provenance manifest in `STATE/manifests/<id>.json` (not in the
   snapshot directory).

No more `~/.local/share/llm/models`.

## Visibility

- `llm list`/`doctor` show the cache root once, and each model's source
  (`built-in`, `cache`, `user`).
- `doctor` reports missing cache files and incomplete split sets.
- `status`/`start` use the resolved snapshot path.

## Edge cases

- Multiple snapshots for one repo: prefer `refs/main`, dedupe by repo+filename.
- Incomplete split set: skipped, reported by `doctor` when it matches a known
  profile.
- Alias collision with a built-in: discovered model gets a `-2` suffix; user
  models always win.
- Cache root missing: discovery returns empty; built-ins resolve to expected
  paths and show as missing.
- Cache root not writable: `get`/migrate fail with an actionable message; reads
  still work.
- Cross-filesystem migration: copy, then remove the source.
- Existing blob with a different size: fail unless `--force`/`--verify` passes.
- Symlinked snapshots on a different volume: resolve normally; `path` works.
- 40-char pinned revision: no `refs` file required.
- Deep subdirectories: symlink target computed with `os.path.relpath`.
- `HF_HOME` unset but `HF_HUB_CACHE` set: use it; expand `~`.

## Testing

- `tests/test_hfcache.py`: root precedence; ref resolution; snapshot selection;
  single/split discovery; incomplete split; mmproj/part skip; dedupe across
  commits; alias collisions; profile synthesis; `_local_size` split sum.
- `tests/test_config.py`: built-ins resolve cache paths; discovered profiles
  merge after built-ins/user models; user models win; missing cache files keep
  expected paths.
- `tests/test_migrate.py`: known mapping; manifest scan; hardlink path; copy
  fallback; symlink target; refs; remove old; idempotency; dry-run; verify.
- `tests/test_cli.py`: `migrate-cache` dry-run/apply output; list shows cache
  models; doctor reports missing/incomplete.
- `tests/test_hub.py`: `get` creates blobs/refs/snapshots and a profile; no
  writes under `~/.local/share/llm`.
- Manual: `llm migrate-cache --dry-run`, then apply on this machine; `llm list`
  shows the three migrated models plus `qwen2.5-coder-14b-instruct-q4_k_m` and
  `qwen3.8-flash-next-gsq-rco-q2_0`; `llm use qwen2.5-7b-instruct` still starts.

## Execution order

This plan executes after context-plan Task 3 (GPU stats) and before context
Task 4 (registry overrides), because it changes `load_profiles` and profile
paths. Context Task 4 must preserve cache discovery when adding overrides.
