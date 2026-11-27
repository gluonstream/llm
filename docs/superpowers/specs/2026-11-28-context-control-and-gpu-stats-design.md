# LLM — context control, `apply`, and real GPU stats

Date: 2026-11-28
Status: draft (awaiting approval)

## Intent

`llm` manages a persistent llama.cpp server, but it hides the single most
important setting: the context window. The server can start at 8,192 tokens
while another tool (OpenCode, graphify) believes the window is 64K or larger.
The result is a loop of `request (N tokens) exceeds the available context size
(8192 tokens)`, failed compaction, and a `doctor` command that neither shows the
context nor explains the failure.

The same gap exists for memory: `apply` should choose 64K or 128K per model
from the model's real KV-cache size and the machine's real memory, and Linux
users should get precise GPU usage instead of a coarse name/size probe.

Goals:

- `llm` is the single source of truth for each model's context.
- Context is visible everywhere it matters: `list`, `status`, `doctor`.
- `llm context` reads or sets one model; `llm apply` sizes every installed
  model in one step.
- `apply` chooses 128K when it fits with headroom, otherwise 64K when it fits,
  otherwise leaves the current value and says why.
- Sizing uses real GGUF metadata (layers, KV heads, head dims) for the KV
  cache, not the current `1.5 GiB per 8K` guess.
- On Linux, `gpustat --json` supplies precise VRAM/ utilisation/process data,
  with `nvidia-smi`, AMD sysfs, and `lspci` fallbacks.
- `doctor` explains context-overflow failures and warns when the running
  server, the stored profile, and OpenCode disagree.
- `apply --sync-opencode` rewrites the local `llamacpp` provider limits so
  OpenCode follows `llm`.

Non-goals:

- No model downloads, no change to the graph-analysis CLI.
- No multi-model concurrency; still one model on port 18081.
- No quantization of the KV cache; sizing assumes the f16 default.
- No automatic installation of `gpustat` or any package (it needs sudo).
- No change to sampling or other profile fields in this iteration.

Anti-goals (would make this a failure even if it works):

- Writing a context that then fails to load or pushes the machine into swap.
- Silently rewriting another tool's config.
- Reporting a "precise" number that is actually a guess.
- Breaking existing flat `models.json` files or existing commands.

## Current state (evidence)

- `llm/cli.py:cmd_doctor` prints model size and `min_llama_cpp`, never
  `context`. `render.format_list` and `render.format_status` are the same.
- The running Inkling server uses `-c 8192` (`process.json`, `/props`
  `n_ctx = 8192`). Its smallest rejected request in `server.log` is 8,904
  tokens, i.e. OpenCode's fixed prompt already exceeds 8,192; compaction cannot
  recover.
- `llm/diagnose.py:explain_failure` only recognises unknown architectures and
  `failed to load model`; `service.last_failure()` returns `None` for the
  context-overflow log.
- Built-in profiles are immutable (`config.register_model` refuses their IDs).
  `--context` exists only on `get`; there is no way to change a built-in's
  context.
- OpenCode's config declares 65,536 for qwen, 32,768 for glm, 8,192 for
  Inkling; `llm`'s built-in qwen is 8,192. A deleted `models.json` (backed up
  as `models.json.backup-20261127-234725`) removed the old 64K custom qwen.
- `llm/hardware.py` probes `nvidia-smi`, AMD sysfs, and `lspci`; it never reads
  used/free VRAM or utilisation, and it ignores `gpustat`.

## Approaches considered

### A. Where context overrides live

1. **Overrides inside `models.json` (chosen).** Built-ins stay code constants;
   user models keep their full profiles; a new top-level `overrides` map holds
   only `{"context": N}` for any model id. `load_profiles` merges overrides on
   top of both. One file, no duplicated paths or sampling, user edits survive.
2. **Materialise every model into `models.json`.** Easy to read, but duplicates
   paths/sampling, drifts from built-ins, and clobbers hand edits. Rejected.
3. **Separate `overrides.json`.** Clean separation but two files and the user
   asked for `models.json`. Rejected for now.

### B. How `apply` sizes context

1. **Largest of {128K, 64K} that fits, using real GGUF KV metadata (chosen).**
   Read the model's metadata, compute f16 KV bytes exactly, add weights and a
   compute margin, compare to the platform budget. Conservative and explainable.
2. **Static table per model.** Simple but wrong on unfamiliar models and on
   machines with different memory. Rejected.
3. **Current crude estimate (`1.5 GiB / 8K`).** Overestimates GQA models by
   roughly 3x (Qwen2.5 7B needs ~7 GiB of KV at 128K, not ~24 GiB), so it would
   wrongly refuse 128K. Rejected.

### C. GPU probing

1. **`gpustat --json` first, then `nvidia-smi`, AMD sysfs, `lspci` (chosen).**
   Gives VRAM used/free, utilisation, temperature and per-process residency.
   Optional external command; no Python dependency.
2. **Keep `nvidia-smi` only.** Fewer dependencies but no utilisation or process
   attribution and no benefit on AMD. Rejected.
3. **Require `gpustat`.** Breaks macOS and non-NVIDIA Linux. Rejected.

### D. Keeping OpenCode in sync

1. **Opt-in `apply --sync-opencode` rewrites the local provider (chosen).**
   `llm` never edits another tool's config unless asked. `doctor` warns about
   mismatches either way.
2. **Always sync.** Surprising side effects. Rejected.
3. **Print a snippet only.** Keeps drift. Rejected.

## Design

### 1. Registry format and overrides

`~/.local/state/llm/models.json` keeps its current compatibility:

- flat `{"<id>": profile}` (legacy), or
- `{"models": {"<id>": profile}, "overrides": {"<id>": {"context": N}}}`.

New helpers in `llm/config.py`:

- `load_registry(path=None)` → `Registry(models, overrides, wrapped)`, accepting
  both shapes. Invalid entries are ignored, as today.
- `save_registry(registry, path=None)` → atomic write, `0600`, preserving the
  wrapped shape. If the file was legacy-flat and overrides are needed, it is
  converted to the wrapped shape once.
- `load_profiles(path=None, builtin=None)` applies user models first (existing
  shadow rules), then overrides by model id. An override may set `context`
  only; a positive integer is required, anything else is ignored with a
  `doctor`-visible warning.
- `register_model` reads and writes the full registry so it never drops the
  `overrides` map.
- `config.PROFILES` stays a module-level cache; add `reload_profiles()` that
  rebuilds it from disk. `apply` and `context` call it after writing.

`apply` writes an override for a built-in and updates the profile entry in
place for a user model (that entry is already the user's own definition).

### 2. GGUF sizing

New `llm/gguf.py`, standard library only:

- `read_metadata(path)` parses the GGUF header (`GGUF`, version, tensor count,
  metadata count), then reads metadata key/value pairs using the little-endian
  type table (uint8..uint64, int8..int64, float32/64, bool, string, array).
  Arrays are consumed; only scalar integers and integer arrays are retained.
  Reads are bounded (header size, metadata count, string length) and any
  malformed file returns `None`, never raises.
- `kv_cache_bytes(meta, context)`:
  - architecture = `general.architecture`;
  - `n_layer = <arch>.block_count`;
  - `n_kv = <arch>.attention.head_count_kv` (may be an int or a per-layer
    array; an array is summed across layers), defaulting to
    `<arch>.attention.head_count`;
  - `d_k = <arch>.attention.key_length` or
    `<arch>.embedding_length // <arch>.attention.head_count`; same for `d_v`;
  - `n_kv_total` = `n_layer * n_kv` for a scalar, or the sum of the per-layer
    array;
  - bytes = `context * n_kv_total * (d_k + d_v) * 2` (f16 KV).
  - Returns `None` if a required field is missing.
- If `context` exceeds `<arch>.context_length` (the model's native window),
  the row is annotated `beyond native N` but not blocked: RoPE scaling and
  llama.cpp can extend it, and the user explicitly wants 64K/128K.
- `estimate_required(model_size, context, metadata=None)`:
  - with metadata: `model_size + kv_cache_bytes + max(1 GiB, 5% of model_size) +
    256 MiB`;
  - without: the existing formula, so current tests and behaviour hold.
  - A table cell marks estimates made without metadata as `approx`.

### 3. Memory budget and context choice

The budget is a **capacity** figure, not instantaneous free memory. This is
deliberate: `apply` is normally run while llama-server holds the model, so
free VRAM already excludes the weights `estimate_required` adds back. Measuring
free memory would make the result depend on whether the server is loaded and
on what other processes are using the GPU.

- Unified memory (Apple Silicon, NVIDIA Tegra/GB10): `total_ram * 0.80`, or
  `available_ram * 0.80` when a meaningful `available_ram` exists (Linux).
- Discrete GPU: `total_vram * 0.90`. If VRAM is unknown, `total_ram * 0.75`.
- Free VRAM is used only for display and for a warning when the machine cannot
  currently host the chosen size (e.g. another process is using the GPU).
  `GpuStats.processes` is display-only, never load-bearing for budgeting.
- Unified detection on Linux: `gpustat`'s `memory.unified` flag when present;
  otherwise a name match for `tegra|jetson|orin|xavier|gb10`, or
  `/proc/device-tree/compatible` containing `nvidia` and one of those. A
  machine flagged unified uses the RAM budget and ignores reported "VRAM".
- `choose_context(profile, hardware, metadata=None, candidates=(131072, 65536),
  ceiling=None)`:
  1. If `ceiling` is set, drop candidates above it.
  2. Build `[current] + candidates`, de-duplicated, largest first.
  3. Return the first whose `estimate_required` is `<= budget`.
  4. If none fit, return the current value with reason
     `"no candidate fits (needs ~X, budget ~Y)"`.
  5. If metadata is missing, use the approximate estimate and mark it.
- `apply --context N` sets `ceiling=N` for the chooser: it is an upper bound,
  never a forced value. That keeps `llm apply --context 131072` safe when it
  runs over every installed model. A hard per-model override is
  `llm context MODEL N`, which writes N and prints a fit warning.

### 4. Command surface

```
llm apply [MODEL...] [--context N] [--dry-run] [--no-restart] [--sync-opencode]
llm context [MODEL] [TOKENS|auto]
```

`llm apply`:

- Selects all installed models (file exists, size > 0) unless models are named.
- Detects hardware once; reads each model's GGUF metadata once.
- Computes the chosen context, writes overrides/user entries, then calls
  `config.reload_profiles()`.
- Restarts the service when the running model's effective context changed
  *or* the server's actual `n_ctx` from `/props` differs from the effective
  profile (drift repair). If `/props` is unavailable, fall back to the
  "value changed" test. `--no-restart` skips this and prints `llm restart`.
  Restart reuses the normal `stop`/`start`/`wait_ready` path and reports the
  new PID. If stopped, it does not start anything.
- `--dry-run` performs no writes, no OpenCode edit, no restart.
- Prints:

```
MODEL                     FILE       CONTEXT            EST. NEEDED   ACTION
qwen2.5-7b-instruct       4.4 GiB    8,192 → 128K       12.9 GiB      updated
inkling-2-mini           16.4 GiB    8,192 → 64K        27.9 GiB      updated
glm-4.7-flash            16.3 GiB    32,768             31.2 GiB      kept (no candidate fits)
```

- Exit code 1 if a write fails; 0 otherwise, including "nothing to change".

`llm context`:

- No arguments: table of every model with its effective context.
- `llm context MODEL`: show that model's value and source (built-in, user,
  override).
- `llm context MODEL N`: write an explicit value (override for built-ins,
  entry for user models), warn if it does not fit, do not restart.
- `llm context MODEL auto`: run the sizing logic for one model and write the
  chosen value; `llm context MODEL auto --dry-run` only prints it. Does not
  restart.

### 5. Visibility and diagnosis

`list` gains a `CONTEXT` column. `status` gains
`Context: 65,536` and, when the service is healthy, the server's real value
from `/props` (`default_generation_settings.n_ctx`), plus a warning if they
differ.

`doctor`:

- Per model: `name`, size, context, `min_llama_cpp` status.
- GPU block:
  ```
  GPU:          NVIDIA RTX 4090 · 24.0 GiB · 3.2 GiB used · 20.8 GiB free · 35% util
                PID 4242 llama-server 12.1 GiB
  ```
  Multiple GPUs print one block each; totals are summed for budgeting.
  A unified-memory GPU is labelled `unified` and its `VRAM` is not treated as
  dedicated.
- If Linux + NVIDIA hardware is detected and `gpustat` is missing:
  `install gpustat for precise VRAM stats: sudo pacman -S python-pipx &&
  pipx ensurepath && pipx install gpustat` (pacman variant when `pacman`
  exists); otherwise a generic `pipx install gpustat` hint.
- Running-server context mismatch: `configured 65536, server 8192 → run llm
  restart`.
- OpenCode mismatch: for each alias in the local `llamacpp` provider,
  `opencode.json says 65536, llm says 8192 → run llm apply --sync-opencode`.
  Missing entries are reported too.

`diagnose.explain_failure` learns:

- `request (N tokens) exceeds the available context size (M tokens)` and the
  JSON `exceed_context_size` form. The hint names the model when it can infer
  the source, and says:
  `request of N tokens exceeds M context → run llm apply (or llm context <model> N)`
- The last match in the tail wins, consistent with the existing load-error
  handling.

### 6. GPU probing

Extend `llm/hardware.py` with a `GpuStats` dataclass (`index`, `name`, `total`,
`used`, `free`, `utilization`, `processes`, `unified`, `source`).
`detect_hardware` keeps its existing `gpu`/`vram` fields and adds a `gpus` list
plus summed totals.

Unified detection: `gpustat`'s `memory.unified` flag when present; otherwise a
name match for `tegra|jetson|orin|xavier|gb10`, or
`/proc/device-tree/compatible` containing `nvidia` and one of those names.
A unified GPU sets `Hardware.unified = True` on Linux (today it is hardcoded
`False`) and routes budgeting through the RAM path.

Probe order:

1. `gpustat --json` (any platform; parse `gpus[]` with `memory.total`,
   `memory.used`, `utilization.gpu`, `processes[]`). MiB → bytes. Missing
   command, non-zero exit, or invalid JSON → fall through.
2. Linux: `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,
   utilization.gpu --format=csv,noheader,nounits`.
3. Linux: AMD sysfs `mem_info_vram_total` / `mem_info_vram_used`.
4. Linux: `lspci` name only.
5. macOS: `system_profiler` name/VRAM as today.

All probes accept an injected runner so tests never touch the host.

### 7. OpenCode sync

New `llm/opencode.py`:

- `config_path()` = `LMM_OPENCODE_CONFIG`, else
  `$XDG_CONFIG_HOME/opencode/opencode.json`, else
  `~/.config/opencode/opencode.json`.
- `read_config(path)` returns `None` when absent or invalid.
- `sync(profiles, path=None)`:
  - `profiles` is the set of **installed** models from the `apply` run, not
    every known profile; a model that is not installed must not appear in
    OpenCode as a selectable-but-dead entry;
  - updates `provider.llamacpp.models` entries keyed by each profile's alias;
  - sets `limit.context` to the effective context, preserving sibling keys;
  - adds a missing entry as `{"name": alias, "limit": {"context": N}}`;
  - touches nothing else (no other provider, no top-level `model`); entries in
    the `llamacpp` provider that `llm` does not know are left alone and listed
    by `doctor` as unknown;
  - writes atomically after copying to `opencode.json.bak-YYYYmmdd-HHMMSS`,
    with 2-space indent and a trailing newline;
  - if the provider or `models` map is missing, do nothing and report why.
- `mismatches(profiles, path=None)` returns
  `[(alias, opencode_value_or_None, llm_value)]` for `doctor`.

### 8. Version and docs

- Bump `__version__` and `LATEST_VERSION` to `1.2.0`.
- README: new `apply`/`context` sections, GPU stats + `gpustat` install steps,
  OpenCode sync, and the overflow diagnosis.
- Update the existing specs index if one exists.

## Edge cases

- Split GGUF (`-00001-of-`): read metadata from the first shard; size is the
  sum of shards (existing `_local_size` logic).
- Unreadable/corrupt GGUF: approximate sizing, `approx` marker, never crash.
- `models.json` legacy flat file: still loads; converted only when an override
  must be written.
- Built-in alias shadowing and unknown override ids: ignored, reported by
  `doctor`.
- Service not healthy: `/props` probe skipped; no restart attempted.
- `--sync-opencode` when OpenCode is not installed: print a note, exit 0.
- `--sync-opencode` only sees models installed during that `apply`; unknown
  entries already in the provider are left alone and reported by `doctor`.
- Multiple GPUs: sum total VRAM for the capacity budget; print each GPU
  separately; warn if live free VRAM is below the chosen estimate.
- `apply` while the server is loaded: budgeting is capacity-based, so the
  running weights are not double-counted or mistaken for missing memory.
- `gpustat` on a unified-memory GPU: use the RAM budget, not the reported
  total.
- Explicit context larger than native training context: allowed, annotated
  `beyond native N`; llama.cpp may still accept it.

## Testing

- `tests/test_gguf.py`: synthetic GGUF (v3) metadata, scalar and array KV
  heads, GQA defaults, missing fields, truncated/corrupt input.
- `tests/test_config.py`: overrides merge onto built-ins and user models;
  legacy flat read; wrapped write; `register_model` preserves overrides;
  invalid overrides ignored; `reload_profiles`.
- `tests/test_hardware.py`: gpustat JSON parsing (incl. `memory.unified`);
  probe order and fallbacks; nvidia-smi used/free/util; AMD used/free;
  capacity budget is independent of loaded weights; `choose_context` with fake
  metadata and budgets (fits 128K, fits only 64K, fits neither, `ceiling` caps).
- `tests/test_opencode.py`: sync updates/creates entries for installed models
  only, preserves other providers, leaves unknown provider entries alone,
  writes a backup, leaves invalid config untouched, reports mismatches.
- `tests/test_diagnose.py`: overflow pattern and hint text.
- `tests/test_cli.py`: `apply --dry-run` output and no writes; `apply` writes
  and reloads; `context` show/set/auto (auto writes; `--dry-run` does not);
  `--context` acts as a ceiling; restart when changed or when `/props` differs;
  `--sync-opencode`; `doctor` GPU/mismatch/overflow lines.
- Manual verification on this Mac: `apply --dry-run` shows qwen → 128K
  (exact KV), inkling/glm conservative; `apply` then `restart`; OpenCode's
  provider updated with `--sync-opencode`.

## Left to the builder

- Exact wording/column widths of the tables.
- Whether `apply` prints per-model GGUF read failures as warnings or rows.
- The precise compute-margin constants, as long as they stay conservative and
  are documented in code.
