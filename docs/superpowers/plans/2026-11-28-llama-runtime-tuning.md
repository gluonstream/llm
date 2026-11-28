# llama.cpp Runtime Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `llm apply` choose KV precision, Flash Attention, batch/ubatch, prompt-cache reuse, and `--fit` headroom from the model, hardware, and observed usage, with conservative/balanced/extended/aggressive policies.

**Architecture:** Typed KV math in `gguf`, a usage parser, and a runtime policy layer feed a richer `runtime` block stored beside `context` in `models.json`. `service.server_args` emits only supported flags and forces `-fa on` for quantized KV. `apply` writes context + runtime atomically, restarts on change, and shows both in `apply`/`context`/`doctor`.

**Tech Stack:** Python 3.9+ standard library; `unittest`; existing `llm` modules. llama.cpp flags are external, detected from `llama-server --help`.

**Spec:** `docs/superpowers/specs/2026-11-28-llama-runtime-tuning-design.md`

## Global Constraints

- Python 3.9+, standard library only; no third-party imports; no `X | Y` runtime annotations.
- Tests via `python3 -m unittest discover -s tests`; never run a real server in tests.
- Balanced default: `q8_0` K+V, 2048/512, cache-reuse 256, fit on target 2048.
- Quantized KV requires `-fa on`; if the flag is unsupported, keep f16.
- Emit optional flags only when `llama-server --help` supports them.
- Old profiles without `runtime` keep today's args (f16, 512/256, no fit/cache-reuse).
- `models.json` stays backward compatible; runtime overrides are validated and atomic.
- Version becomes `1.3.0`.

## Review Focus

1. A llama.cpp build without `-ctk/-ctv` or `--fit` — must omit the flags, not fail to start.
2. A profile with no runtime or no GGUF metadata — must keep safe defaults and not crash.
3. `--dry-run`/`--no-restart` — must write nothing and restart nothing.
4. Usage history from another model or from a failed request — must not inflate the chosen context.
5. Asymmetric or invalid KV types / small VRAM — must stay valid and documented.

---

### Task 1: Typed KV math and usage history

**Files:**
- Modify: `llm/gguf.py`
- Create: `llm/usage.py`
- Test: `tests/test_gguf.py`, `tests/test_usage.py`

**Interfaces:**
- Produces:
  - `gguf.KV_BYTES_PER_ELEMENT: dict`
  - `gguf.kv_cache_bytes(metadata, context, kv_type_k="f16", kv_type_v="f16") -> Optional[int]`
  - `usage.peak_tokens(log_path, model_path=None, tail_bytes=2*1024*1024) -> Optional[int]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gguf.py
def test_kv_cache_q8_0_is_half_f16(self)
    # same metadata: q8_0/q8_0 == f16/f16 * 1.0625/2.0
def test_kv_cache_q4_0_is_quarter_f16(self)
    # q4_0/q4_0 == f16/f16 * 0.5625/2.0
def test_kv_cache_asymmetric_types(self)
    # q8_0 K + f16 V == context * n_kv * (d_k*1.0625 + d_v*2.0)
def test_kv_cache_unknown_type_returns_none(self)

# tests/test_usage.py
def test_peak_uses_last_model_load(self)
    # two model blocks; returns the last model's max n_tokens
def test_peak_reads_release_and_prompt_lines(self)
    # n_tokens from release and prompt processing; max wins
def test_peak_ignores_zero_and_missing_log(self)
    # "n_tokens = 0" ignored; missing log -> None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_gguf tests.test_usage -v`
Expected: missing `KV_BYTES_PER_ELEMENT` / `llm.usage`.

- [ ] **Step 3: Implement**

`gguf.KV_BYTES_PER_ELEMENT = {"f16": 2.0, "bf16": 2.0, "q8_0": 1.0625,
"q5_1": 0.75, "q5_0": 0.6875, "q4_1": 0.625, "q4_0": 0.5625}`.

`kv_cache_bytes` keeps the existing `n_kv_total`, `d_k`, `d_v`; returns
`int(context * n_kv_total * (d_k * bytes_k + d_v * bytes_v))`; `None` for an
unknown type or missing metadata. The default (`f16/f16`) matches today's
result exactly.

`usage.peak_tokens`:
- read the last `tail_bytes` of the log (decode errors replaced);
- if `model_path` is given, keep only text after the last
  `loading model '<model_path>'`; if that marker is missing, use the tail;
- regex all `n_tokens = (\d+)` and the `/ N tokens` in `prompt eval time`;
- ignore `0`; return `max(...)` or `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_gguf tests.test_usage -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/gguf.py llm/usage.py tests/test_gguf.py tests/test_usage.py
git commit -m "feat: add typed KV math and usage-history parsing"
```

---

### Task 2: Runtime policies and config storage

**Files:**
- Create: `llm/runtime.py`
- Modify: `llm/hardware.py` (`estimate_required`, `choose_context`)
- Modify: `llm/config.py` (`Registry` runtime, `set_tuning`, `load_profiles`)
- Test: `tests/test_runtime.py`, `tests/test_hardware.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `runtime.POLICIES` (`conservative`, `balanced`, `extended`, `aggressive`)
  - `runtime.policy_runtime(policy="balanced", overrides=None) -> dict`
  - `runtime.candidates_for(policy, observed=None) -> list`
  - `runtime.choose_runtime(model_size, current_context, machine, metadata=None,
    policy="balanced", observed=None, kv=None, kv_k=None, kv_v=None,
    batch=None, ubatch=None,
    fit_target=None, no_fit=False, no_cache_reuse=False) -> TunedChoice`
  - `@dataclass TunedChoice`: `context`, `runtime`, `reason`, `required`,
    `budget`, `approximate`, `native`
  - `hardware.estimate_required(..., kv_type_k="f16", kv_type_v="f16")`
  - `hardware.choose_context(..., kv_type_k="f16", kv_type_v="f16")`
  - `config.set_tuning(model_id, context, runtime, path=None) -> None`
  - `config.load_profiles` applies `runtime` overrides

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_runtime.py
def test_balanced_policy_fields(self)
    # q8_0/q8_0, 2048/512, cache_reuse 256, fit True/2048, fa on, parallel 1
def test_conservative_and_extended_kv(self)
    # conservative q8_0 64K; extended q4_0 128K
def test_candidates_filter_by_observed_peak(self)
    # peak 40_000 -> [128K, 64K]; peak 70_000 -> [128K]
def test_kv_override_applies_to_both(self)
def test_no_fit_and_no_cache_reuse(self)
def test_choose_runtime_uses_typed_kv(self)
    # q8_0 fits 128K where f16 would not
def test_choose_runtime_keeps_current_when_none_fit(self)

# tests/test_hardware.py
def test_estimate_required_typed_kv(self)

# tests/test_config.py
def test_runtime_override_valid_and_applied(self)
def test_invalid_runtime_override_reported(self)
def test_set_tuning_builtin_and_user(self)
def test_set_tuning_preserves_context(self)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_runtime tests.test_hardware tests.test_config -v`
Expected: missing `llm.runtime` / `set_tuning`.

- [ ] **Step 3: Implement**

`runtime.POLICIES`:
- conservative: f16/f16, [65536], 1024/256, cache-reuse 256, fit 2048.
- balanced: q8_0/q8_0, [131072, 65536], 2048/512.
- extended: q8_0/q8_0, [131072], 2048/1024.
- aggressive: q4_0/q4_0, [131072], 2048/1024.
All: `flash_attn="on"`, `parallel=1`, `cache_prompt=True`, `cache_reuse=256`,
`fit=True`, `fit_target=2048`, `kv_unified=True`, `no_kv_offload=False`.

`policy_runtime` returns the dict with the policy's fields; `overrides` are
applied on top (used by CLI flags). `candidates_for` filters by
`observed * 1.5` when observed is truthy; if the filter is empty, returns all.

`choose_runtime`:
- build runtime from the policy; apply `kv`, `kv_k`, `kv_v`, `batch`, `ubatch`,
  `fit_target`, `no_fit`, `no_cache_reuse`;
- candidates = `candidates_for(policy, observed)`;
- call `hardware.choose_context(size, current, machine, metadata,
  candidates=candidates, kv_type_k=..., kv_type_v=...)`;
- return `TunedChoice` with the choice fields and the runtime dict.

`hardware.estimate_required(model_size, context=8192, metadata=None,
kv_type_k="f16", kv_type_v="f16")`: pass the types to `gguf.kv_cache_bytes`;
otherwise unchanged. `choose_context` forwards the types to
`estimate_required`.

`config`:
- `Registry.overrides[id]` may be `{"context": n, "runtime": {...}}`;
- `_validate_runtime(runtime) -> (clean, errors)` checks the allowed keys/types
  (KV types from `gguf.KV_BYTES_PER_ELEMENT`; `flash_attn` in on/off/auto;
  positive ints; bools);
- `load_profiles` applies both `context` and `runtime` overrides (copy the dict);
- `set_tuning(model_id, context, runtime, path=None)` validates, updates a user
  entry or an override with both fields, saves atomically; callers reload.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_runtime tests.test_hardware tests.test_config -v`
Expected: all pass; existing tests remain green.

- [ ] **Step 5: Commit**

```bash
git add llm/runtime.py llm/hardware.py llm/config.py tests/test_runtime.py tests/test_hardware.py tests/test_config.py
git commit -m "feat: add runtime policies and runtime overrides"
```

---

### Task 3: Server arguments and flag support

**Files:**
- Modify: `llm/service.py`
- Test: `tests/test_service.py`

**Interfaces:**
- Produces:
  - `service.parse_help(text) -> set`
  - `service.supported_flags(llama_server=None) -> set`
  - `service.server_args(profile, llama_server=None, supported=None)`

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_help_collects_short_and_long_flags(self)
def test_balanced_runtime_emits_kv_batch_fit_flags(self)
def test_quantized_kv_forces_flash_attention_on(self)
def test_unsupported_flags_are_omitted(self)
def test_legacy_profile_keeps_old_args(self)
def test_no_kv_offload_and_no_cache_reuse(self)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_service -v`
Expected: missing `parse_help`/`supported_flags` or missing flags.

- [ ] **Step 3: Implement**

`parse_help`: split lines; collect every token that starts with `--` or matches
`-[a-z]+` (strip trailing commas); include `-ctk`, `--cache-type-k`, etc.

`supported_flags`: run `[llama_server, "--help"]` with a short timeout, cache by
path, return `parse_help(stdout+stderr)`; empty set on failure.

`server_args`:
- start from current args (`-m`/`-hf`, `--offline`, `-lm mmap`, `--lazy-mode on`,
  `--no-mmproj`, `-ngl 99`, host/port/alias/jinja/no-webui);
- `rt = profile.get("runtime") or {}`;
- KV types default `f16`; if either != f16 and `-ctk`/`-ctv` are supported:
  add `-ctk/-ctv`; force `-fa on`;
- otherwise add `-fa <rt.get("flash_attn","on")>`;
- add `-c context`, `-b rt.batch(default 512)`, `-ub rt.ubatch(default 256)`,
  `--parallel rt.parallel(default 1)`;
- add `-kvu` if `rt.kv_unified` and supported;
- add `--cache-prompt`/`--no-cache-prompt` if supported (default true when the
  runtime is present; omit for legacy profiles);
- add `--cache-reuse N` if supported and `rt.cache_reuse`;
- add `--fit on --fit-target N` if supported and `rt.fit`;
- add `--no-kv-offload` if `rt.no_kv_offload` and supported;
- sampling args unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_service -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/service.py tests/test_service.py
git commit -m "feat: emit tuned llama.cpp runtime flags"
```

---

### Task 4: `apply`, `context`, and `doctor` integration

**Files:**
- Modify: `llm/cli.py` (`cmd_apply`, `cmd_context`, `cmd_doctor`, parser)
- Modify: `llm/render.py` (`format_apply`, `runtime_summary`)
- Test: `tests/test_cli.py`, `tests/test_render.py`

**Interfaces:**
- Consumes: `runtime.choose_runtime`, `usage.peak_tokens`,
  `config.set_tuning`, `service.supported_flags`.
- Produces: `apply` policy/KV/batch flags; `Runtime:` doctor line; context
  runtime summary; `format_apply` KV/BATCH columns.

- [ ] **Step 1: Write the failing tests**

```python
def test_apply_balanced_writes_runtime_and_context(self)
def test_apply_policy_conservative_and_aggressive(self)
def test_apply_kv_override(self)
def test_apply_uses_observed_peak(self)
def test_apply_dry_run_writes_no_runtime(self)
def test_apply_restarts_on_runtime_change(self)
def test_apply_no_cache_reuse_and_no_fit(self)
def test_context_shows_runtime_summary(self)
def test_doctor_shows_runtime_and_peak(self)
def test_format_apply_includes_kv_and_batch(self)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_cli tests.test_render -v`
Expected: unknown flags / missing output.

- [ ] **Step 3: Implement**

Parser additions:
`--runtime-policy` choices with default `balanced` (alias `--policy`); `--kv`,
`--kv-k`, `--kv-v` choices; `--batch`,
`--ubatch`, `--fit-target` ints; `--no-fit`, `--no-cache-reuse`.

`cmd_apply`:
- for each target, `observed = usage.peak_tokens(service.LOG, profile.get("path"))`;
- `tuned = runtime.choose_runtime(size, profile["context"], machine, metadata,
  policy=args.runtime_policy, observed=observed, kv=args.kv,
  kv_k=args.kv_k, kv_v=args.kv_v, batch=args.batch,
  ubatch=args.ubatch, fit_target=args.fit_target, no_fit=args.no_fit,
  no_cache_reuse=args.no_cache_reuse)`;
- row includes `kv`, `batch`, `ubatch`, `context`, `required`, `action`;
- changed when `tuned.context != profile["context"]` or
  `tuned.runtime != (profile.get("runtime") or {})`;
- write `config.set_tuning(id, tuned.context, tuned.runtime)`;
- reload, sync clients with the chosen contexts, restart when changed or
  `/props` differs (existing logic extended to runtime changes);
- `--dry-run` no writes/restart/sync writes.

`render.runtime_summary(runtime)` → `"q8_0 KV · 2048/512 · fit 2048 · cache-reuse 256"`;
`format_apply` adds `KV` and `BATCH` columns (use `row.get` for compatibility).
`cmd_context` prints the summary after the context line.
`cmd_doctor` prints `Runtime: ... · peak <short>` for the selected model when a
runtime exists; `peak` from `usage.peak_tokens`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_cli tests.test_render -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/cli.py llm/render.py tests/test_cli.py tests/test_render.py
git commit -m "feat: apply runtime policies and show tuning"
```

---

### Task 5: Version 1.3.0, docs, full verification

**Files:**
- Modify: `llm/__init__.py`, `README.md`
- Modify: `docs/superpowers/specs/2026-11-28-llama-runtime-tuning-design.md` (status → approved)
- Test: full suite / build / dry-run

- [ ] **Step 1: Bump to 1.3.0**

Set `__version__` and `LATEST_VERSION` to `1.3.0`.

- [ ] **Step 2: README**

Add a "Runtime tuning" section: policies table, `apply --runtime-policy/--kv/--kv-k/--kv-v/--batch/
--ubatch/--fit-target/--no-fit/--no-cache-reuse`, the `q8_0`/`q4_0` memory
story, `-fa on` requirement, usage-history sizing, and the `--fit` margin.
Include the example command from the spec.

- [ ] **Step 3: Mark the spec approved** if not already.

- [ ] **Step 4: Full suite + build**

Run: `python3 -m unittest discover -s tests -v`; all pass.
Run: `LMM_SKIP_LLAMA_CPP_UPGRADE=1 ./build.sh`; `Built .../dist/llm`.

- [ ] **Step 5: Manual dry-run**

```bash
./dist/llm apply --dry-run
./dist/llm apply --dry-run --runtime-policy extended
./dist/llm apply --dry-run --runtime-policy aggressive
```

Expected: balanced moves qwen to 128K and Inkling/GLM to 64K with q8_0;
extended uses q4_0 and 128K where it fits; the 61.9 GiB model is refused.

- [ ] **Step 6: Commit**

```bash
git add llm/__init__.py README.md docs/superpowers/specs/2026-11-28-llama-runtime-tuning-design.md
git commit -m "docs: release 1.3.0 with runtime tuning"
```
