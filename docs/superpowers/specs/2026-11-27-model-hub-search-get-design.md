# Model Hub — search and get — design

Date: 2026-11-27
Status: approved for implementation (user: "yes implement phase 2")

## Intent

`lmm` can manage models it already knows, but adding one still means finding a
GGUF on Hugging Face by hand, downloading it with curl, and editing a profile.
This phase adds two commands:

- `lmm search <query>` — find Hugging Face repos by name.
- `lmm get <repo>[:<quant>]` — download one GGUF, verify it, and register it so
  `lmm list`, `lmm use` and `lmm start` see it immediately.

## Goals

- Search the Hugging Face model API by name, GGUF repos first by default.
- Resolve a repo's GGUF files, with `:QUANT` / `--file` selection and a clear
  candidate list when ambiguous.
- Download with resume, progress, sha256 verification and atomic rename.
- Write the same `download-manifest.json` shape used by the existing models.
- Register the model in `~/.local/state/llm/models.json`, merged
  with the built-in profiles at import.
- Keep the CLI dependency-free (Python standard library only).

## Non-goals

- Split GGUF sets (`-00001-of-00002`) — detect and refuse with a clear message.
- Auto-starting the model; `get` ends with `lmm use <alias>`.
- OAuth / gated-repo login flows; a public repo or `HF_TOKEN` is enough.
- Resumable downloads across a changing remote file (size/sha from the API).
- Searching datasets, spaces, or non-model repos.

## Command surface

```
lmm search <query> [--limit N] [--all]
lmm get <repo>[:<quant-or-file>] [--file NAME] [--revision REV]
        [--name ALIAS] [--context N] [--requires-llama-cpp VER] [--force]
```

Examples:

```
lmm search inkling
lmm search --all qwen
lmm get unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_XL
lmm get unsloth/Inkling-2-Mini-GGUF --file Inkling-2-Mini-UD-Q4_K_XL.gguf
```

- `search`: `GET /api/models?search=…&sort=downloads&direction=-1&limit=…`.
  Default filters to repos whose id or tags mark them GGUF; `--all` disables the
  filter. Output: repo id, downloads, likes, `gated` marker.
- `get`: `GET /api/models/<repo>?blobs=true`, keep `.gguf` siblings, select by
  case-insensitive exact name, name without `.gguf`, or substring. No selector
  with one GGUF picks it; more than one lists candidates with sizes.

## Registry

- `models.json` is a flat object `{ "<id>": <profile> }`.
- `load_profiles()` merges built-in profiles with the registry; invalid entries
  are skipped rather than crashing `lmm`.
- `register_model(profile)` validates required keys (`id`, `alias`, `path`,
  `context`, `sampling`), refuses to overwrite a built-in id, refuses an alias
  already used by a different id, writes atomically, and updates `PROFILES`.
- IDs and aliases are derived from repo + filename and made unique with `-2`,
  `-3`, … suffixes.

## Download behaviour

- Destination: `~/.local/share/llm/models/<id>/<filename>`.
- Partial file: `<filename>.part`; resume with `Range: bytes=<size>-` when the
  server answers 206, restart cleanly on 200.
- Progress to stderr at most once every 2 seconds: percent, done/total, speed,
  ETA.
- Verify the LFS `sha256` when the API provides it; on mismatch remove the
  partial and fail with a `--force`/retry hint.
- `os.replace` the verified part into place; write `download-manifest.json`
  with `url`, `revision`, `bytes`, `sha256`, `repo`, `filename`.
- Existing valid destination is a no-op; existing mismatched destination fails
  unless `--force`.
- `HF_TOKEN`, when set, is sent as `Authorization: Bearer …`.
- 401/403 adds "set HF_TOKEN or choose a public repo" to the error.

## Decisions

- Standard library (`urllib`, `json`, `hashlib`) rather than `huggingface_hub`
  or the `hf` CLI, matching the project's zero-dependency constraint.
- Default context 8192 and sampling `{temperature 0.7, top_p 1.0, min_p 0.01,
  repeat_penalty 1.0}`, matching the other general-purpose profiles.
- `--requires-llama-cpp` stores `min_llama_cpp` so `doctor` warns before a load.
- `search` hides non-GGUF results by default and reports how many were hidden.
- `get` does not mutate the running service; switching stays `lmm use`.

## Error handling

- Network/HTTP/JSON failures raise one `HubError` with the URL and status.
- Ambiguous or missing selectors raise `HubError` with candidate files/sizes.
- Checksum/size mismatch leaves no half-valid model in place.
- Ctrl-C leaves the `.part` file for the next run.

## Testing

- Registry merge/validation/uniqueness with temp files.
- Spec parsing, file filtering and selection (exact, quant, ambiguous, none).
- Download engine against a local HTTP server, including 206 resume, 200
  restart, checksum mismatch, and already-complete no-op.
- CLI dispatch with the hub mocked; live `lmm search inkling` as a network smoke
  test (download of a 16 GB quant is left to the user).

## Left to the builder

- Exact table widths and number formatting.
- Progress formatting details.
- Whether to print the manifest path after a successful download.

## Hardware preflight (added 2026-11-27)

Before `get` downloads a model, estimate whether this machine can load it.

- Detect total RAM (`sysctl hw.memsize`), chip (`machdep.cpu.brand_string`),
  architecture (`hw.optional.arm64`) and free disk on the models volume.
- Unified-memory Macs budget `total_ram * 0.8`; the remaining RAM is left for
  macOS and other applications.
- Required memory = model size + context overhead, where overhead is
  2 GiB + 1.5 GiB per 8192 tokens of context.
- Disk needed = model size + 512 MiB.
- If either check fails, print a framed warning naming the model, the estimate,
  the machine and the disk, then exit 1 without downloading.
- `--force` bypasses the refusal and downloads anyway, printing that the model
  may fail to load.
- Already-registered models skip the preflight unless `--force` is given.
- `lmm doctor` prints chip, RAM, usable budget and free disk.
