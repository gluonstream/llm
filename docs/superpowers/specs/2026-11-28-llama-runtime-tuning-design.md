# LLM — llama.cpp runtime tuning

Date: 2026-11-28
Status: approved (balanced default agreed by the human)

## Intent

`llm apply` currently only writes `-c`. The installed llama.cpp build supports
KV-cache quantization, flash attention, prompt-cache reuse, batch sizing, and
`--fit` headroom, all of which decide whether 64K context is comfortable or
fatal on a 32–36 GiB machine. `apply` should choose those flags from the model,
the hardware, and observed usage, with conservative/balanced/extended policies.

Default policy: **balanced**. It uses `q8_0` KV (half the KV memory, near-zero
quality loss), keeps Flash Attention on, sets batch 2048 / ubatch 512, enables
prompt-cache reuse, and lets `--fit` preserve a 2 GiB margin.

## Goals

- Store a `runtime` block per model alongside `context`:
  `kv_type_k`, `kv_type_v`, `flash_attn`, `batch`, `ubatch`, `parallel`,
  `cache_prompt`, `cache_reuse`, `fit`, `fit_target`, `kv_unified`,
  `no_kv_offload`.
- `llm apply` chooses context + runtime together, writes both atomically, and
  restarts the running model when either changes.
- Context stays memory-aware but now uses the policy's KV precision:
  `q8_0` halves KV bytes, `q4_0` quarters them.
- Usage history from `server.log` right-sizes context: allocate enough for
  observed peak + 50%, rounded up to 64K/128K, not the model's native maximum.
- `doctor` and `context` show the effective runtime and observed peak.
- Old profiles without `runtime` keep today's behavior (f16 KV, 512/256, no
  cache-reuse, no fit) until `apply` writes a tuned runtime.
- Every flag is emitted only when the installed `llama-server --help` supports
  it; quantized KV always forces `-fa on`.

## Non-goals

- Warm KV save/restore (`--slot-save-path` only sets a path; it does not save
  automatically). A separate `llm slot-save/restore` can come later.
- Layer offload strategy beyond `-ngl 99`; tensor-split tuning for multi-GPU.
- Benchmarking real throughput; the policy is heuristic, with `--fit` as the
  runtime guard.
- Changing sampling, chat templates, or client sync.

## Anti-goals

- Writing a KV quantization that the server rejects (no `-fa on`).
- Choosing a context that swaps or crashes instead of letting `--fit` protect.
- Silently raising memory use without a visible policy/KV column.

## Policies

| Policy | KV K/V | Context candidates | batch/ubatch | cache-reuse | fit |
|---|---|---|---|---|---|
| `conservative` | `f16` / `f16` | 64K | 1024 / 256 | 256 | on, 2048 |
| `balanced` (default) | `q8_0` / `q8_0` | 128K, 64K | 2048 / 512 | 256 | on, 2048 |
| `extended` | `q8_0` / `q8_0` | 128K | 2048 / 1024 | 256 | on, 2048 |
| `aggressive` | `q4_0` / `q4_0` | 128K | 2048 / 1024 | 256 | on, 2048 |

All policies set `flash_attn="on"`, `parallel=1`, `cache_prompt=true`,
`kv_unified=true`, `no_kv_offload=false`.

- `conservative` is maximum accuracy / more KV RAM: f16 K and V, 64K.
- `balanced` cuts KV RAM roughly in half with near-identical perplexity:
  q8_0 K and V, 64K/128K by memory.
- `extended` keeps the balanced precision but prefers 128K and larger batches
  when memory allows.
- `aggressive` shrinks KV RAM ~70–75% for the largest context, accepting some
  precision/recall loss: q4_0 K and V. Use `--kv-k q5_0 --kv-v q4_0` for the
  slightly safer asymmetric variant.

## Sizing

KV bytes per element:

| type | bytes/element | note |
|---|---|---|
| `f16`, `bf16` | 2.0 | current default |
| `q8_0` | 1.0625 | 34 bytes / 32 values |
| `q5_1` | 0.75 | 24 / 32 |
| `q5_0` | 0.6875 | 22 / 32 |
| `q4_1` | 0.625 | 20 / 32 |
| `q4_0` | 0.5625 | 18 / 32 |

`kv_cache_bytes(metadata, context, kv_type_k, kv_type_v)` uses the existing
`n_kv_total`, `d_k`, `d_v`, with `d_k * bytes_k + d_v * bytes_v` per token.

`estimate_required(model_size, context, metadata, kv_type_k, kv_type_v)` is
weights + KV + `max(1 GiB, 5% weights) + 256 MiB`.

Budget stays capacity-based (`context_budget`): unified 0.80 × RAM, discrete
0.90 × VRAM, fallback 0.75 × RAM. `--fit-target` is a runtime margin, not
subtracted from the planning budget.

## Usage history

`usage.peak_tokens(log_path, model_path=None, tail_bytes=2 MiB)`:

- read the tail; if `model_path` appears in a `loading model '<path>'` line,
  use only the text after the last such line; otherwise use the whole tail;
- collect every `n_tokens = (\d+)` from `release`/`prompt processing` and the
  denominator of `prompt eval time = … / N tokens`;
- ignore zero; return the maximum, or `None`.

Context candidates:

- policy candidates in descending order (e.g. `[128K, 64K]`);
- if `peak` is not `None` and > 0, keep only candidates `>= peak * 1.5`, or all
  candidates when none reach it;
- `choose_context` picks the largest that fits; if none fit, keep current and
  report the reason.

## Server arguments

`service.server_args` reads `profile.get("runtime", {})` with safe defaults
(f16/512/256/no cache-reuse/no fit). It emits:

- `-ctk <type> -ctv <type>` only when the type is not `f16` and the help text
  supports `-ctk/-ctv`; this forces `-fa on`.
- `-fa on` whenever KV is quantized; otherwise the profile's `flash_attn`
  (default `on`).
- `-b`, `-ub`, `--parallel`, `-kvu`.
- `--cache-prompt` / `--no-cache-prompt` and `--cache-reuse N` when supported.
- `--fit on --fit-target <MiB>` when supported and enabled.
- `--no-kv-offload` only when `no_kv_offload` is true and supported.

`service.supported_flags(llama_server=None)` runs `llama-server --help` once and
returns a cached set of option names; tests inject it. Unknown/old builds simply
omit unsupported flags and stay on f16.

## Config

`models.json` overrides grow from `{"context": N}` to
`{"context": N, "runtime": {...}}`. Validation:

- KV types from the table above;
- `flash_attn` in `on|off|auto`;
- positive ints for `batch`, `ubatch`, `parallel`, `cache_reuse`, `fit_target`;
- booleans for `cache_prompt`, `fit`, `kv_unified`, `no_kv_offload`;
- unknown keys or bad values are ignored and reported in `Registry.invalid`.

`config.set_tuning(model_id, context, runtime, path=None)` writes a user entry
or an override atomically and preserves the rest. `load_profiles` applies the
runtime override to built-in, discovered, and user profiles.

## Commands

```
llm apply [MODEL...] [--runtime-policy conservative|balanced|extended|aggressive]
          [--kv TYPE] [--kv-k TYPE] [--kv-v TYPE] [--batch N] [--ubatch N] [--fit-target MiB]
          [--no-fit] [--no-cache-reuse]
          [--context N] [--dry-run] [--no-restart]
          [--sync-opencode] [--sync-pi] [--sync-clients]
llm context [MODEL] [TOKENS|auto]
```

`--kv TYPE` sets both K and V; `--kv-k TYPE` and `--kv-v TYPE` override
individually (for example `--kv-k q5_0 --kv-v q4_0`). `--policy` is accepted as
a hidden alias for `--runtime-policy`.

`apply` prints `MODEL`, `FILE`, `CONTEXT`, `KV`, `BATCH`, `EST. NEEDED`,
`ACTION`. Example:

```
qwen2.5-7b-instruct   4.4 GiB   8K → 128K   q8_0   2048/512   10.5 GiB   updated
inkling-2-mini       16.4 GiB   8K → 64K    q8_0   2048/512   27.4 GiB   updated
```

Restart when the running model's `context` or `runtime` changed, or `/props`
differs; `--no-restart` prints `llm restart`. `--dry-run` writes nothing.

`llm context` prints the effective context and a runtime summary
(`q8_0 KV · 2048/512 · fit 2048 · cache-reuse 256`); `context auto` uses the
profile's current KV type (f16 if unset) and writes only context.

`doctor` adds a `Runtime:` line for the selected model with KV, batch/ubatch,
fit, cache-reuse, and the observed peak from `server.log`.

## Model analysis

`llm analyze [--context N]` prints one row per model in `config.PROFILES`
(built-ins + cache-discovered + user), using the same GGUF metadata as
`apply`:

```
MODEL                  SIZE    CTX    f16      q8_0     q4_0     q8_0 SAVE   FIT
qwen2.5-7b-instruct    4.4G    64K    3.50G    1.86G    0.98G    46.9%       yes
```

- Values are at `--context` (default 64K); `fits` uses `context_budget` and the
  q8_0 estimate.
- Missing/unreadable metadata produces a row with `-` values, never a crash.
- Inkling's per-layer head counts are handled by the typed KV math.

## Generation and speculative decoding

Runtime fields extend the `runtime` block:

- `top_k`, `n_predict` — server sampling defaults (`--top-k`, `-n`).
- `spec_type` (`none`, `draft-simple`, `draft-mtp`, `draft-eagle3`),
  `draft_model` (path) — speculative decoding (`-md`, `--spec-type`).
- `cmoe` — keep MoE expert weights on CPU (`-cmoe`).
- `reasoning` (`on|off|auto`), `reasoning_budget` — `-rea`,
  `--reasoning-budget`.
- `devices` — comma-separated `-dev` list.

`apply` exposes `--spec-type`, `--draft MODEL`, `--cmoe`, `--reasoning`,
`--reasoning-budget`, `--device`, `--top-k`, `--predict`. `--draft MODEL`
resolves a profile and stores its cache path; a draft without `--spec-type`
defaults to `draft-simple`. Flags are emitted only when supported. `doctor`
and `context` include spec/CMOE/reasoning in the runtime summary.

## Codex sync

`llm apply --sync-codex` edits `~/.codex/config.toml` (or
`CODEX_HOME/config.toml`) with a targeted line editor so comments, plugins,
and marketplaces survive:

- top-level `model = "<selected alias>"`, `model_provider = "llamacpp"`;
- `[model_providers.llamacpp]` with `name = "llama.cpp"`,
  `base_url = "<llm URL>/v1"`, `wire_api = "responses"`,
  `requires_openai_auth = false`, `supports_websockets = false`;
- timestamped backup, atomic write, no write when nothing changed;
- `--sync-clients` covers OpenCode, Pi, and Codex; `doctor` warns on drift.

## Edge cases

- `-ctk/-ctv` unsupported: keep f16 and report that the policy's KV was not
  applied.
- `-fa` unsupported/off: force f16 KV.
- Model file missing: `apply` reports `missing`, writes nothing for it.
- No usage history: choose the largest candidate that fits.
- Usage peak above the largest candidate: keep the largest candidate that fits.
- Unified memory: `no_kv_offload` stays false.
- Discrete GPU where weights + KV exceed VRAM but fit RAM: the chooser may keep
  a smaller context; `no_kv_offload` remains an explicit flag for now.
- Runtime override on a built-in/discovered id: written to `models.json`.
- Old profile with no runtime: unchanged arguments.

## Testing

- `tests/test_gguf.py`: typed KV bytes (q8_0 half, q4_0 quarter, asymmetric K/V).
- `tests/test_usage.py`: last-load filtering, release/prompt lines, zero
  handling, missing log, rotated/missing model path.
- `tests/test_runtime.py`: policy table, candidate filtering by observed peak,
  `--kv/--batch` overrides, no-fit/no-cache-reuse.
- `tests/test_config.py`: runtime override validation, `set_tuning` for user and
  built-in/discovered ids, preservation of context.
- `tests/test_service.py`: flags emitted per runtime, unsupported flags omitted,
  `-fa on` forced for quantized KV, legacy defaults.
- `tests/test_analysis.py`: rows for f16/q8_0/q4_0, missing metadata, Inkling per-layer heads, table columns.
- `tests/test_codex.py`: TOML edits preserve other sections, backup, idempotency, dry-run, mismatches.
- `tests/test_cli.py`: `apply --runtime-policy` writes context+runtime, dry-run leaves
  both untouched, restart on runtime change, table columns, `context`/`doctor`
  runtime lines.
- Manual: `analyze`, `apply --dry-run`, `apply --dry-run --sync-clients` on this machine; balanced should move qwen to 128K
  and Inkling/GLM to 64K with q8_0, while the 61.9 GiB cache model is refused.
