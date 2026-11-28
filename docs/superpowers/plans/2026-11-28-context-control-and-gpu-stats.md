# Context Control, `apply`, and GPU Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `llm` the source of truth for each model's context window, size it from real GGUF metadata and machine memory, expose it everywhere, and report real NVIDIA/AMD GPU usage.

**Architecture:** A pure-stdlib GGUF metadata reader feeds an exact KV-cache estimate; `apply` chooses 128K/64K per installed model, writes context overrides into `models.json`, optionally syncs OpenCode, and restarts a drifted server. A `gpustat`-first probe enriches `doctor` without becoming a dependency. All context writes go through the existing registry merge, so built-ins stay immutable as code and `models.json` stays backward compatible.

**Tech Stack:** Python 3.9+ standard library only; `unittest`; existing `llm` modules (`config`, `hardware`, `service`, `diagnose`, `render`, `cli`). `gpustat`, `nvidia-smi`, `lspci`, `system_profiler` are optional external commands, never imports.

**Spec:** `docs/superpowers/specs/2026-11-28-context-control-and-gpu-stats-design.md`

## Global Constraints

- Python 3.9 or newer, standard library only. No third-party Python imports. `gpustat` is invoked as a subprocess.
- Target Python 3.9: avoid `X | Y` runtime annotations; use `Optional`/`Union` or no annotations. `dict[str, int]`/`list[int]` are fine.
- macOS and Arch Linux must both keep working. NVIDIA/AMD probes are optional and must degrade to name-only.
- `models.json` must stay backward compatible with the legacy flat shape.
- Version becomes `1.2.0` in `llm/__init__.py`.
- Tests run with `python3 -m unittest discover -s tests`; `build.sh` runs the same suite.
- Commit messages use `feat:`, `fix:`, `test:`, or `docs:` prefixes.

## Review Focus

These are the input classes most likely to bite a user; each has a test in its owning task.

1. Corrupt, truncated, or unusual GGUF metadata during `apply` — must never crash and must fall back to an approximate estimate, marked as such.
2. `apply` while llama-server is loaded — must budget from capacity, never mistake the loaded weights for missing memory, and must restart or tell the user to restart.
3. A legacy flat `models.json` or a hand-edited one — must not lose user models or overrides; conversion to the wrapped shape happens only when needed.
4. Machines without `gpustat`, `nvidia-smi`, or an NVIDIA GPU (macOS, AMD, Intel) — `doctor` still works and never prints invented VRAM/utilisation.
5. Missing, invalid, or third-party OpenCode config — `--sync-opencode` must skip or back up, never corrupt another provider.

---

### Task 1: GGUF metadata reader and KV-cache math

**Files:**
- Create: `llm/gguf.py`
- Create: `tests/test_gguf.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `read_metadata(path) -> Optional[dict]`
  - `kv_cache_bytes(metadata: dict, context: int) -> Optional[int]`
  - `native_context(metadata: dict) -> Optional[int]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gguf.py` with a small GGUF v3 writer helper and these tests. The helper writes the `GGUF` magic, version 3, tensor count 0, the metadata count, then each key/type/value. Metadata used in the happy path:

```python
META = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.embedding_length": 3584,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.context_length": 32768,
}
```

Tests and assertions:

```python
def test_reads_architecture_and_headers(self)
    # read_metadata(path)["general.architecture"] == "qwen2"
    # read_metadata(path)["qwen2.block_count"] == 28

def test_kv_cache_uses_gqa_and_head_dim(self)
    # 28 layers * 4 kv heads * (128 + 128) * 2 bytes * 131072 == 7_516_192_768

def test_kv_cache_sums_per_layer_array(self)
    # head_count_kv = [4] * 28 gives the same 7_516_192_768

def test_kv_cache_honours_explicit_key_and_value_length(self)
    # key_length 64, value_length 64 halves the result

def test_missing_metadata_returns_none(self)
    # {} and {"general.architecture": "qwen2"} -> kv_cache_bytes(...) is None

def test_truncated_file_returns_none(self)
    # write 12 bytes of a header -> read_metadata(...) is None

def test_skips_large_string_arrays(self)
    # append an array of 5,000 strings; read_metadata still returns the arch

def test_native_context_reads_context_length(self)
    # native_context(META) == 32768 and {} -> None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_gguf -v`
Expected: `ModuleNotFoundError: No module named 'llm.gguf'`.

- [ ] **Step 3: Implement `llm/gguf.py`**

Constants: `MAX_METADATA = 100_000`, `MAX_ARRAY = 100_000`, `MAX_STRING = 1 << 20`. Type table:
`0..12` = uint8, int8, uint16, int16, uint32, int32, float32, bool, string, array, uint64, int64, float64.
Format map for fixed-size scalars: `{0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}`.

`read_metadata`:
- open binary; `header = handle.read(4)`; if `header != b"GGUF"` return `None`;
- `version, tensor_count, kv_count = struct.unpack("<IQQ", handle.read(16))`;
- if `kv_count > MAX_METADATA` return `None`;
- loop `kv_count`: read a string key (u64 length + UTF-8, length <= `MAX_STRING`), read u32 type;
- scalar string: read one string and store; scalar fixed-size: `struct.unpack(fmt, read(size))[0]`;
- array (type 9): read u32 element type and u64 count. If element type is a fixed-size scalar and `count <= MAX_ARRAY`, read the whole array into a list. If element type is string, loop `count` times reading length and `seek(length, 1)` without retaining. If nested array, skip recursively with a depth cap of 4. If `count > MAX_ARRAY`, skip the bytes for fixed-size elements.
- Any `struct.error`, `OSError`, `UnicodeDecodeError`, or `ValueError` returns `None`.

`kv_cache_bytes`:
- `arch = metadata.get("general.architecture")`; return `None` if not a non-empty string.
- `n_layer`, `n_head` must be positive ints.
- `n_kv_raw = metadata.get("<arch>.attention.head_count_kv", n_head)`; if list, `n_kv_total = sum(int(x) for x in n_kv_raw)`; else `n_kv_total = n_layer * int(n_kv_raw)`.
- `d_k = metadata.get("<arch>.attention.key_length")`; if absent, `embd // n_head`; same for `d_v`.
- Return `context * n_kv_total * (d_k + d_v) * 2`; `None` if `context <= 0` or any value is not positive.

`native_context`: return `<arch>.context_length` when it is a positive int, else `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_gguf -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/gguf.py tests/test_gguf.py
git commit -m "feat: read GGUF metadata and estimate KV cache size"
```

---

### Task 2: Memory budget and context choice

**Files:**
- Modify: `llm/hardware.py` (after `memory_budget`/`estimate_required`)
- Test: `tests/test_hardware.py` (append to `HardwareTest`)

**Interfaces:**
- Consumes: `llm.gguf.kv_cache_bytes`, `llm.gguf.native_context` (tests may inject a `metadata` dict directly).
- Produces:
  - `@dataclass ContextChoice` with `context: int`, `reason: str`, `required: Optional[int]`, `budget: Optional[int]`, `approximate: bool`, `native: Optional[int]`
  - `context_budget(machine: Hardware) -> Optional[int]`
  - `estimate_required(model_size, context=8192, metadata=None) -> int`
  - `choose_context(model_size: int, current: int, machine: Hardware, metadata=None, candidates=(131072, 65536), ceiling=None) -> ContextChoice`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hardware.py`:

```python
def test_context_budget_unified(self)
    # Hardware(total_ram=36 GiB, unified=True, vram=None)
    # context_budget(...) == int(36 GiB * 0.80)

def test_context_budget_discrete_uses_total_vram(self)
    # Hardware(total_ram=64 GiB, unified=False, vram=24 GiB)
    # context_budget(...) == int(24 GiB * 0.90)

def test_estimate_required_with_metadata_uses_kv(self)
    # model=4 GiB, metadata giving 1 GiB of KV at 8K
    # estimate_required(4 GiB, 8192, meta) == 4 GiB + 1 GiB + 1 GiB + 256 MiB

def test_choose_context_prefers_128k_when_it_fits(self)
    # 128K is returned when required <= budget

def test_choose_context_falls_back_to_64k(self)
    # a metadata/KV shape where only 64K fits returns 65536

def test_choose_context_keeps_current_when_none_fit(self)
    # current=32768 and neither candidate fits -> context == 32768

def test_choose_context_ceiling_caps_candidates(self)
    # ceiling=65536 never returns 131072

def test_choose_context_marks_approximate_without_metadata(self)
    # metadata=None -> approximate is True

def test_choose_context_reports_native_window(self)
    # metadata with context_length=32768 -> choice.native == 32768
```

Use a helper metadata dict built by `gguf`-style keys and a fake `Hardware`; do not start llama-server.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_hardware -v`
Expected: `AttributeError: module 'llm.hardware' has no attribute 'choose_context'`.

- [ ] **Step 3: Implement**

`context_budget(machine)`:
- `machine.unified` → `memory_budget(machine.total_ram, True, machine.available_ram)`
- elif `machine.vram` → `int(machine.vram * 0.90)`
- else → `memory_budget(machine.total_ram, False, machine.available_ram)`

Add `from . import gguf` at the top of `llm/hardware.py`.

`estimate_required(model_size, context=8192, metadata=None)`:
- if `metadata` and `gguf.kv_cache_bytes(metadata, context)` is not `None`:
  `model_size + kv + max(1 GiB, int(model_size * 0.05)) + 256 MiB`
- else keep the existing formula exactly (`2 GiB + (context / 8192) * 1.5 GiB`).

`choose_context`:
- `budget = context_budget(machine)`; if `None`, return current with reason `"memory budget unknown"` and `budget=None`.
- candidates list = `[current] + [c for c in candidates if c != current]`, descending.
- if `ceiling` is not `None`, drop every value in that list that is `> ceiling`.
- for each remaining candidate: `required = estimate_required(...)`; if `required <= budget`, return a `ContextChoice` with that value and reason `"fits"` (or `"fits (approximate)"` when metadata is `None`).
- if none fit, current with reason `"no candidate fits (needs ~X, budget ~Y)"`; when a ceiling dropped the current value, say `"no candidate <= ceiling fits; keeping current"`.
- `approximate = metadata is None`; `native = gguf.native_context(metadata) if metadata else None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_hardware -v`
Expected: all pass, including the existing budget/preflight tests.

- [ ] **Step 5: Commit**

```bash
git add llm/hardware.py tests/test_hardware.py
git commit -m "feat: choose context from exact KV size and memory budget"
```

---

### Task 3: Real GPU stats with gpustat first

**Files:**
- Modify: `llm/hardware.py` (`Hardware`, `detect_hardware`, new probes)
- Test: `tests/test_hardware.py` (append)

**Interfaces:**
- Consumes: existing `_run`, `_linux_gpu`, `_displays`.
- Produces:
  - `@dataclass GpuStats` with `index`, `name`, `total`, `used`, `free`, `utilization`, `processes`, `unified`, `source`
  - `parse_gpustat(payload: str) -> Optional[list[GpuStats]]`
  - `gpustat_gpus(runner=None) -> Optional[list[GpuStats]]`
  - `nvidia_smi_gpus(runner=None) -> Optional[list[GpuStats]]`
  - `amd_sysfs_gpus(sysfs_root=None) -> Optional[list[GpuStats]]`
  - `detect_gpus(runner=None, sysfs_root=None, displays=None) -> list[GpuStats]`
  - `is_unified_gpu_name(name: str) -> bool`
  - `Hardware.gpus: list[GpuStats]`
  - `detect_hardware(..., gpu_stats=None)`

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_gpustat_reads_usage_and_processes(self)
    # payload with two gpus; assert name, total/used/free in bytes,
    # utilization, and processes[0]["pid"]

def test_parse_gpustat_rejects_bad_json(self)
    # parse_gpustat("not json") is None and parse_gpustat("{}") is None

def test_gpustat_falls_back_to_nvidia_smi(self)
    # gpustat runner returns "" (missing); nvidia-smi returns a CSV line;
    # detect_gpus returns a GpuStats with source "nvidia-smi"

def test_nvidia_smi_reads_used_free_util(self)
    # "NVIDIA RTX 4090, 24564, 3200, 21364, 35" -> exact byte values

def test_amd_sysfs_reads_used(self)
    # mem_info_vram_total=8589934592, mem_info_vram_used=1073741824
    # -> total/free/used/ source "sysfs"

def test_unified_gpu_name(self)
    # "NVIDIA GB10" and "Jetson Orin" are True; "NVIDIA RTX 4090" is False

def test_detect_hardware_linux_marks_unified_gb10(self)
    # linux_gpu injected as ("NVIDIA GB10", 128 GiB) plus gpu_stats
    # returning a unified GpuStats -> machine.unified is True
```

All runners are injected; no test runs `gpustat` or `nvidia-smi` on the host.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_hardware -v`
Expected: `AttributeError: module 'llm.hardware' has no attribute 'parse_gpustat'`.

- [ ] **Step 3: Implement**

`parse_gpustat(payload)`:
- `json.loads(payload)`; require `{"gpus": [...]}`.
- Each GPU: `index` (default 0), `name` (required), `memory.total`/`memory.used` (MiB → bytes; may be `None`), `utilization.gpu`, `processes` (keep `pid`, `command`, `gpu_memory_usage`), `memory.unified` (bool).
- `free = max(total - used, 0)` when both exist.
- Return `None` on any parse/type error.

`gpustat_gpus(runner=None)`: run `["gpustat", "--json"]` via the injected runner (default `_run`); `None` if output is empty; else `parse_gpustat`.

`nvidia_smi_gpus(runner=None)`: query `name,memory.total,memory.used,memory.free,utilization.gpu` with `--format=csv,noheader,nounits`; split each line; MiB → bytes; `None` on failure.

`amd_sysfs_gpus(sysfs_root=None)`: for each `card*/device`, read `mem_info_vram_total` and optional `mem_info_vram_used`; return `GpuStats(name="AMD GPU", total=..., used=..., free=..., source="sysfs")`.

`detect_gpus`:
1. `gpustat_gpus(runner)`
2. `nvidia_smi_gpus(runner)`
3. `amd_sysfs_gpus(sysfs_root)`
4. Linux `lspci` via `_linux_gpu(runner=runner)` → name-only `GpuStats`
5. macOS `displays()` → name/VRAM `GpuStats` with `source="system_profiler"`
Return `[]` when nothing is found.

`is_unified_gpu_name(name)`: case-insensitive regex `\b(gb10|jetson|orin|xavier|tegra)\b`.

`Hardware`: add `gpus: list = field(default_factory=list)` (import `field`). Keep `gpu` and `vram`.

`detect_hardware`:
- add `gpu_stats=None`;
- Linux: if `include_gpu`: `gpus = gpu_stats() if gpu_stats else (wrap(_linux_gpu(...)) if linux_gpu else detect_gpus())`; derive `gpu`/`vram` from `gpus`; `unified = any(g.unified for g in gpus)`. The legacy wrap sets `unified=is_unified_gpu_name(name)`; `detect_gpus` sets it from gpustat or the same name test.
- macOS: same precedence (`gpu_stats`, then injected `displays`, then `detect_gpus(displays=displays)`); `unified` stays as today.
- Existing injected-probe tests must keep passing unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_hardware -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/hardware.py tests/test_hardware.py
git commit -m "feat: read real GPU usage via gpustat with fallbacks"
```

---

### Task 4: Registry overrides in `models.json`

**Files:**
- Modify: `llm/config.py`
- Test: `tests/test_config.py` (append to `RegistryTest`)

**Interfaces:**
- Consumes: existing `BUILTIN_PROFILES`, `USER_MODELS`, `validate_profile`, `load_user_profiles`. **After the HF-cache plan, `load_profiles` resolves built-ins from cache coordinates, merges `hfcache.discover(...)` profiles with identity dedupe, and populates `warnings`. Preserve all of that when adding overrides. Merge order is built-ins → discovered → user → overrides; overrides must apply to discovered profiles by id. `set_context` updates a `models.json` entry for user models and writes an override for built-ins or discovered models. `context_source` returns `"cache"` for discovered ids.**
- Produces:
  - `@dataclass Registry` with `models: dict`, `overrides: dict`, `wrapped: bool`, `invalid: list`
  - `load_registry(path=None) -> Registry`
  - `save_registry(registry, path=None) -> None`
  - `load_profiles(path=None, builtin=None, warnings=None, discovered=None) -> dict` (modified)
  - `reload_profiles(path=None, builtin=None, discovered=None) -> dict`
  - `set_context(model_id, context, path=None) -> None`
  - `context_source(model_id, path=None) -> str` (`"override"`, `"user"`, `"built-in"`, `"unknown"`)

- [ ] **Step 1: Write the failing tests**

```python
def test_legacy_flat_registry_still_loads(self)
    # {"mine": profile} -> registry.models["mine"], overrides == {}, wrapped is False

def test_overrides_apply_to_builtin(self)
    # {"overrides": {DEFAULT: {"context": 65536}}} changes merged[DEFAULT]["context"]

def test_overrides_apply_to_user_model(self)
    # user model context 8192 + override 131072 -> merged value 131072

def test_invalid_override_is_ignored_and_reported(self)
    # {"overrides": {"x": {"context": "big"}}} -> ignored, registry.invalid non-empty

def test_save_registry_wraps_when_overrides_exist(self)
    # save -> json has "models" and "overrides" keys

def test_register_model_preserves_overrides(self)
    # pre-write overrides, register a user model, re-read: overrides still there

def test_set_context_writes_override_for_builtin(self)
    # set_context(DEFAULT, 65536) -> overrides[DEFAULT]["context"] == 65536

def test_set_context_updates_user_model_entry(self)
    # user model -> models[id]["context"] updated, no override entry

def test_reload_profiles_refreshes_global(self)
    # mock USER_MODELS to a temp file, write an override, reload_profiles(),
    # assert config.PROFILES[DEFAULT]["context"] changed

def test_context_source(self)
    # built-in -> "built-in", override -> "override", user -> "user"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_config -v`
Expected: `AttributeError: module 'llm.config' has no attribute 'load_registry'`.

- [ ] **Step 3: Implement**

`Registry` + `load_registry`:
- read JSON; on `OSError`/`ValueError` return empty `Registry({}, {}, False, [])`;
- wrapped when the top-level dict has a `models` dict; otherwise legacy flat;
- `models` = dict values only, excluding the literal key `"overrides"`; `overrides` = top-level `overrides` dict;
- each override must be a dict with an `int` `context > 0`; push `(id, reason)` to `invalid` otherwise; store only `{"context": n}`.

`save_registry`:
- doc = `{"models": registry.models}` when `registry.wrapped or registry.overrides`, else `registry.models`;
- include `"overrides": registry.overrides` when non-empty;
- atomic `NamedTemporaryFile` in the parent, `json.dump(indent=2, sort_keys=True)`, `os.replace`, mode `0600`; remove the temp on failure.

`load_profiles`:
- start from built-ins, merge user models exactly as today (no shadowing built-ins/aliases);
- apply overrides to any merged id; unknown override ids append to `warnings` when provided;
- extend `warnings` with `registry.invalid`.

`register_model`: use `load_registry`/`save_registry`; keep the existing built-in/alias checks; set `registry.models[model_id] = profile`; update `profiles` in memory.

`set_context`: validate `type(context) is int and context > 0`; built-in id → `overrides[id] = {"context": context}`; user model id → update `models[id]["context"]` and remove any override; unknown → raise `ModelError`. Save only; no reload.

`reload_profiles`: `global PROFILES; PROFILES = load_profiles(...)`; return it.

`context_source`: check `registry.overrides`, then `registry.models`, then `BUILTIN_PROFILES`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_config -v`
Expected: all pass; existing flat/alias tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add llm/config.py tests/test_config.py
git commit -m "feat: store per-model context overrides in models.json"
```

---

### Task 5: Show context and explain overflow

**Files:**
- Modify: `llm/diagnose.py`
- Modify: `llm/service.py` (add `server_context`)
- Modify: `llm/render.py`
- Modify: `llm/cli.py` (`cmd_list`, `cmd_status`)
- Test: `tests/test_diagnose.py`, `tests/test_service.py`, `tests/test_render.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `config.PROFILES`, `service.healthy`, `service.HTTP`.
- Produces:
  - `diagnose.explain_failure` recognises the overflow line
  - `service.server_context(timeout=2) -> Optional[int]`
  - `render.short_context(n) -> str` (`8192 -> "8K"`, `65536 -> "64K"`, `131072 -> "128K"`, `8191 -> "8,191"`)
  - `render.format_list(..., contexts=None)`
  - `render.format_status(..., context=None, server_context=None)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_diagnose.py
def test_explain_context_overflow(self)
    # "request (13996 tokens) exceeds the available context size (8192 tokens)"
    # message contains "13,996", "8,192", "llm apply"

def test_explain_uses_last_overflow_in_tail(self)
    # two overflow lines -> the later token counts appear

# tests/test_service.py
def test_server_context_reads_props(self)
    # mock HTTP.open to return {"default_generation_settings": {"n_ctx": 8192}}
    # service.server_context() == 8192

def test_server_context_returns_none_when_unreachable(self)
    # HTTP.open raises OSError -> None

# tests/test_render.py
def test_short_context(self)
    # 8192 -> "8K", 65536 -> "64K", 131072 -> "128K", 8191 -> "8,191"

def test_list_shows_context_column(self)
    # format_list(..., contexts={id: 65536}) contains "64K"

def test_status_shows_configured_and_server_context(self)
    # format_status(..., context=8192, server_context=8192) contains "8K"
    # and "server 8K"; a mismatch contains "warning"

# tests/test_cli.py
def test_status_warns_when_server_context_differs(self)
    # mock selected context 65536 and server_context 8192 -> output has "warning"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_diagnose tests.test_service tests.test_render tests.test_cli -v`
Expected: missing attributes/columns.

- [ ] **Step 3: Implement**

`diagnose.py`: add
`CONTEXT_OVERFLOW_RE = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")`.
Scan lines in order, keep the last match; when found return:
`"request of %s tokens exceeds the %s-token context \u2192 run `llm apply` (or `llm context <model> N`) and restart"` with comma-formatted counts. Check this before the generic "failed to load" branch, after the architecture branch.

`service.py`:
```python
def server_context(timeout=2):
    try:
        with HTTP.open(URL + "/props", timeout=timeout) as response:
            data = json.load(response)
    except (OSError, ValueError):
        return None
    settings = data.get("default_generation_settings")
    value = settings.get("n_ctx") if isinstance(settings, dict) else data.get("n_ctx")
    return value if type(value) is int and value > 0 else None
```

`render.py`: add `short_context`; extend `format_list` with a `CONTEXT` column (width 7, right-aligned) and `format_status` with `Context: <short>`, plus `server <short>` and a warning line when both are present and differ. Keep all existing parameters and defaults.

`cli.py`:
- `_running_info` is unchanged; `cmd_list` passes `{id: profile["context"] for ...}`.
- `cmd_status` passes `selected["context"]` and, when running and healthy, `service.server_context()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_diagnose tests.test_service tests.test_render tests.test_cli -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/diagnose.py llm/service.py llm/render.py llm/cli.py tests/test_diagnose.py tests/test_service.py tests/test_render.py tests/test_cli.py
git commit -m "feat: show context and explain context-overflow failures"
```

---

### Task 6: Client config sync — OpenCode and Pi

**Files:**
- Create: `llm/opencode.py`
- Create: `llm/pi.py`
- Create: `tests/test_opencode.py`
- Create: `tests/test_pi.py`

**Interfaces:**
- Consumes: nothing from earlier tasks; each adapter receives a profiles dict.
- Produces (both adapters):
  - `config_path() -> Path`
  - `read_config(path=None) -> Optional[dict]`
  - `@dataclass SyncResult` with `path`, `updated`, `added`, `unchanged`, `skipped`, `backup`
  - `sync(profiles, path=None, dry_run=False) -> SyncResult`
  - `mismatches(profiles, path=None) -> list`
  - `unknown_entries(profiles, path=None) -> list`

- [ ] **Step 1: Write the failing tests**

OpenCode: use a temp `opencode.json` shaped like the real one (a `llamacpp` provider with `qwen2.5-7b-instruct-graphify` at 65536, plus a `deepseek` provider).

```python
def test_opencode_sync_updates_existing_entry(self)
    # limit.context == 8192, other keys preserved
def test_opencode_sync_adds_missing_entry(self)
def test_opencode_sync_preserves_other_providers(self)
def test_opencode_sync_leaves_unknown_entries(self)
def test_opencode_sync_writes_backup(self)
def test_opencode_sync_skips_invalid_config(self)
def test_opencode_sync_dry_run_does_not_write(self)
def test_opencode_mismatches_reports_differences(self)
```

Pi: use a temp `models.json` shaped like the real `~/.pi/agent/models.json` — a `providers.llamacpp-local` with `baseUrl` `http://127.0.0.1:18081/v1`, `api` `openai-completions`, `apiKey` `local`, and a `models` list whose entries have `id`, `name`, `contextWindow`, `maxTokens`; plus an unrelated provider.

```python
def test_pi_config_path_uses_env(self)
    # PI_CODING_AGENT_DIR -> <dir>/models.json; otherwise ~/.pi/agent/models.json
def test_pi_sync_updates_context_window_by_id(self)
    # existing {"id": alias, "contextWindow": 8192, ...} -> contextWindow updated, siblings preserved
def test_pi_sync_adds_missing_model(self)
    # new {"id": alias, "name": alias, "contextWindow": N, "maxTokens": 2048}
def test_pi_sync_preserves_other_providers(self)
def test_pi_sync_requires_the_local_provider(self)
    # no provider with the local baseUrl and no llamacpp-local -> skipped, file unchanged
def test_pi_sync_writes_backup(self)
def test_pi_sync_skips_invalid_config(self)
def test_pi_sync_dry_run_does_not_write(self)
def test_pi_mismatches_reports_differences(self)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_opencode tests.test_pi -v`
Expected: `ModuleNotFoundError` for `llm.opencode` / `llm.pi`.

- [ ] **Step 3: Implement**

OpenCode (`llm/opencode.py`):
- `config_path`: `LMM_OPENCODE_CONFIG`, else `$XDG_CONFIG_HOME/opencode/opencode.json`, else `~/.config/opencode/opencode.json`.
- `read_config`: `None` on `OSError`/`ValueError`/non-dict.
- `sync`: use `provider.llamacpp.models`; update `entry["limit"]["context"]` by alias, add missing `{"name": alias, "limit": {"context": N}}`; preserve siblings; skip when the provider/models are missing; no write/backup if nothing changed; atomic write with a timestamped backup.
- `mismatches`/`unknown_entries` as before.

Pi (`llm/pi.py`):
- `config_path`: `PI_CODING_AGENT_DIR` + `/models.json`, else `~/.pi/agent/models.json`.
- `read_config`: `None` on `OSError`/`ValueError`/non-dict.
- `sync`:
  - find a provider whose `baseUrl` (trailing slash trimmed) equals `config.URL + "/v1"`; if none, use `providers["llamacpp-local"]` when present; otherwise skip with `"no local llamacpp provider"`.
  - `models` must be a list; map entries by `id`.
  - existing `id` == alias → set `contextWindow`; preserve every other key.
  - missing → append `{"id": alias, "name": alias, "contextWindow": N, "maxTokens": 2048}`.
  - no write/backup when nothing changed; `dry_run` returns without writing.
  - atomic write with a timestamped backup (`models.json.bak-YYYYmmdd-HHMMSS`), 2-space indent, trailing newline.
- `mismatches`: `(alias, contextWindow_or_None, llm_context)`.
- `unknown_entries`: provider model ids not in the supplied profiles.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_opencode tests.test_pi -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/opencode.py llm/pi.py tests/test_opencode.py tests/test_pi.py
git commit -m "feat: sync context limits into OpenCode and Pi configs"
```

### Task 7: Doctor integration

**Files:**
- Modify: `llm/cli.py` (`cmd_doctor`)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `config.load_registry`, `hardware.detect_hardware` (with `gpus`), `opencode.mismatches`, `opencode.unknown_entries`, `pi.mismatches`, `pi.unknown_entries`, `service.server_context`.
- Produces: no new public functions; `doctor` prints context, GPU, install hint, drift and override warnings.

- [ ] **Step 1: Write the failing tests**

```python
def test_doctor_shows_context_and_gpu(self)
    # patch detect_hardware to a Hardware with one GpuStats(total=24 GiB,
    # used=3 GiB, free=21 GiB, utilization=35, source="gpustat")
    # output contains "65K"/"64K" for a model and "35%" and "used"

def test_doctor_warns_when_gpustat_missing(self)
    # Linux + NVIDIA GpuStats(source="lspci") + shutil.which("gpustat") None
    # output contains "pipx install gpustat"

def test_doctor_warns_server_context_mismatch(self)
    # running, healthy, server_context 8192, selected context 65536
    # output contains "configured 65K" and "server 8K"

def test_doctor_warns_opencode_mismatch(self)
    # opencode.mismatches returns [("qwen...", 65536, 8192)]
    # output contains "opencode" and "--sync-opencode"

def test_doctor_warns_pi_mismatch(self)
    # pi.mismatches returns [("qwen...", 65536, 8192)]
    # output contains "pi" and "--sync-pi"

def test_doctor_shows_invalid_override_warning(self)
    # config.load_registry().invalid = [("x", "invalid context")]
    # output contains "warning" and "x"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_cli -v`
Expected: assertions fail on the missing output.

- [ ] **Step 3: Implement `cmd_doctor` additions**

- Per-model line: `name`, size, `short_context(profile["context"])`, `min_llama_cpp` note.
- GPU: for each `machine.gpus`, print
  `GPU: <name> · <total> · <used> used · <free> free · <util>% util`
  (omit missing fields); unified GPUs append `· unified`. If `gpus` is empty, print nothing new.
- Hint: if `host.IS_LINUX` and any GPU name contains `NVIDIA` and `shutil.which("gpustat")` is `None`, print the install hint. Use the pacman variant when `shutil.which("pacman")` exists; otherwise `pipx install gpustat`.
- Server drift: when `pid` and `service.healthy()`, `actual = service.server_context()`; if `actual` and `actual != selected["context"]`, print `warning: configured <short>, server <short> → run llm restart`.
- OpenCode: `installed = {id: p for id, p in config.PROFILES.items() if _local_size(p)}`; print each mismatch from `opencode.mismatches(installed)` as `warning: opencode.json says <x>, llm says <y> for <alias> → run llm apply --sync-opencode`; print unknown entries as a note.
- Pi: the same for `pi.mismatches(installed)` / `pi.unknown_entries(installed)`, using `warning: pi models.json says <x>, llm says <y> for <alias> → run llm apply --sync-pi`.
- Invalid overrides: `for model_id, reason in config.load_registry().invalid: print("warning: override for %s ignored (%s)" % ...)`.
- Keep the existing output and `Last failure` block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_cli -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/cli.py tests/test_cli.py
git commit -m "feat: report context, GPU and config drift in doctor"
```

---

### Task 8: `llm context`

**Files:**
- Modify: `llm/cli.py` (`cmd_context`, parser)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `config.resolve_model`, `config.set_context`, `config.reload_profiles`, `config.context_source`, `hardware.detect_hardware`, `hardware.choose_context`, `llm.gguf.read_metadata`.
- Produces: `cmd_context(args)` and the `context` subcommand.

- [ ] **Step 1: Write the failing tests**

```python
def test_context_lists_models(self)
    # ["context"] -> each alias and its short context

def test_context_sets_builtin_override(self)
    # patch USER_MODELS to temp; ["context", "qwen2.5-7b-instruct", "65536"]
    # -> models.json overrides[DEFAULT]["context"] == 65536

def test_context_auto_writes_choice(self)
    # patch detect_hardware and gguf.read_metadata and _local_size
    # ["context", "qwen2.5-7b-instruct", "auto"] -> override written

def test_context_auto_dry_run_does_not_write(self)
    # same with --dry-run -> models.json unchanged

def test_context_warns_when_explicit_does_not_fit(self)
    # explicit 131072 on a machine whose budget cannot host it
    # -> output contains "warning" and the value still written
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_cli -v`
Expected: unknown command / `cmd_context` missing.

- [ ] **Step 3: Implement**

Parser:
```python
command = add("context", cmd_context, "show or set a model's context window")
command.add_argument("model", nargs="?")
command.add_argument("tokens", nargs="?")
command.add_argument("--dry-run", action="store_true")
```

Add `gguf` to the `from . import ...` line in `llm/cli.py`.

`cmd_context`:
- no model: table `MODEL CONTEXT SOURCE`, using `config.PROFILES` and `config.context_source`.
- model + no tokens: print `alias: <short> (<source>)`.
- tokens `auto`: size = `_local_size(profile)` (error if missing); metadata = `gguf.read_metadata(profile["path"])` if a path exists; machine = `hardware.detect_hardware(models_dir=MODELS_DIR)`; `choice = hardware.choose_context(size, profile["context"], machine, metadata)`; print the choice and reason; if not `--dry-run`, `config.set_context(id, choice.context)` then `config.reload_profiles()`.
- numeric tokens: `int(tokens)`; require `> 0`; print a warning when `estimate_required(size, n, metadata) > context_budget(machine)`; if not `--dry-run`, set + reload.
- Never restart.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_cli -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/cli.py tests/test_cli.py
git commit -m "feat: add llm context to show and set the context window"
```

---

### Task 9: `llm apply`

**Files:**
- Modify: `llm/cli.py` (`cmd_apply`, parser)
- Modify: `llm/render.py` (`format_apply`)
- Test: `tests/test_cli.py`, `tests/test_render.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `cmd_apply(args)`, `render.format_apply(rows)`, `apply` subcommand.

Row dict keys: `model`, `size`, `old`, `new`, `required`, `action`, `approximate`, `native`. `format_apply` prints the table from the spec with `short_context` and `human_size`.

- [ ] **Step 1: Write the failing tests**

```python
def test_apply_dry_run_writes_nothing(self)
    # patch detect_hardware/read_metadata/_local_size; snapshot models.json
    # ["apply", "--dry-run"] -> file unchanged, output has "→"

def test_apply_updates_and_reloads(self)
    # ["apply"] writes overrides and config.PROFILES reflects the new context

def test_apply_context_is_ceiling(self)
    # ["apply", "--context", "65536"] never writes 131072

def test_apply_skips_missing_models(self)
    # a profile whose file is absent is not written; output says "missing"

def test_apply_restarts_when_running_context_changed(self)
    # service.running returns a pid, server_context 8192, chosen 65536
    # -> service.stop and service.start called once

def test_apply_no_restart_prints_hint(self)
    # --no-restart -> no stop/start, output contains "llm restart"

def test_apply_syncs_opencode_with_flag(self)
    # --sync-opencode -> opencode.sync called with installed profiles

def test_apply_syncs_pi_with_flag(self)
    # --sync-pi -> pi.sync called with installed profiles

def test_apply_syncs_both_with_sync_clients(self)
    # --sync-clients -> both opencode.sync and pi.sync called

def test_apply_default_does_not_touch_clients(self)
    # without a flag -> neither opencode.sync nor pi.sync is called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_cli tests.test_render -v`
Expected: unknown command / missing `format_apply`.

- [ ] **Step 3: Implement**

Parser:
```python
command = add("apply", cmd_apply, "size context for installed models and write models.json")
command.add_argument("models", nargs="*")
command.add_argument("--context", type=int, default=None)
command.add_argument("--dry-run", action="store_true")
command.add_argument("--no-restart", action="store_true")
command.add_argument("--sync-opencode", action="store_true")
command.add_argument("--sync-pi", action="store_true")
command.add_argument("--sync-clients", action="store_true")
```

Add `opencode` and `pi` to the `from . import ...` line in `llm/cli.py`.

`cmd_apply`:
1. `targets`: named models via `config.resolve_model`, or every profile with `_local_size` not `None`. Error on a named model that is not installed.
2. `machine = hardware.detect_hardware(models_dir=hfcache.cache_root())`.
3. For each target: `size = _local_size(profile)`; `metadata = gguf.read_metadata(profile["path"])` when a path exists; `choice = hardware.choose_context(size, profile["context"], machine, metadata, ceiling=args.context)`; build a row. Missing files are rows with `action="missing"` and are not written.
4. Print `render.format_apply(rows)`.
5. `installed = {id: config.PROFILES[id] for id in target_ids if _local_size(profile) is not None}`.
6. If `--dry-run`:
   - `--sync-opencode` or `--sync-clients` → `opencode.sync(installed, dry_run=True)` and print the summary;
   - `--sync-pi` or `--sync-clients` → `pi.sync(installed, dry_run=True)` and print the summary;
   - return 0.
7. `with service.locked():`
   - for each changed row: `config.set_context(id, choice.context)`;
   - `config.reload_profiles()`;
   - `--sync-opencode` or `--sync-clients` → `opencode.sync(installed)`, print added/updated/backup;
   - `--sync-pi` or `--sync-clients` → `pi.sync(installed)`, print added/updated/backup;
   - restart decision: `pid = service.running()`; if `pid`:
     - `actual = service.server_context()`;
     - `running_id` = the profile whose alias appears in `service.identity(pid)`;
     - `expected` = that running profile's effective `context`, falling back to
       `config.selected_profile()["context"]` when no profile matches;
     - restart if `actual is not None and actual != expected`, or `running_id in changed_ids`;
     - if restart and not `--no-restart`: `service.stop()`; `pid, created, profile = service.start()`; `service.wait_ready(pid, created, profile)`; print `restarted PID`.
     - if restart and `--no-restart`: print `context changed; run llm restart`.
8. Return 0.

`render.format_apply(rows)`:
- header `MODEL`, `FILE`, `CONTEXT`, `EST. NEEDED`, `ACTION`;
- context cell `old → new` when changed, else `new`;
- append ` approx` when `approximate`; append ` · beyond native N` when `native` is set and `new > native`;
- action text from the row.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_cli tests.test_render -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llm/cli.py llm/render.py tests/test_cli.py tests/test_render.py
git commit -m "feat: add llm apply to size and persist context"
```

---

### Task 10: Version, docs, full verification

**Files:**
- Modify: `llm/__init__.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-11-28-context-control-and-gpu-stats-design.md` (status → approved)
- Test: full suite / build

**Interfaces:**
- Consumes: all previous tasks.
- Produces: release 1.2.0 with user documentation.

- [ ] **Step 1: Bump the version**

Set `__version__ = "1.2.0"` and `LATEST_VERSION = "1.2.0"`.

- [ ] **Step 2: Update the README**

Add:
- `llm context [MODEL] [N|auto]` and `llm apply [MODEL...]` to the command list;
- an `apply` section with the table example, `--dry-run`, `--context` as a ceiling, `--no-restart`, `--sync-opencode`, `--sync-pi`, `--sync-clients`, and the 64K/128K rule;
- a GPU section: `gpustat` install on Arch
  (`sudo pacman -S python-pipx`, `pipx ensurepath`, `pipx install gpustat`,
  reopen the shell), macOS fallback, and that it is optional;
- a "Context overflow" note: `doctor` now explains
  `request (N tokens) exceeds the available context size (M tokens)` and
  points at `llm apply`.
- OpenCode sync: `llm apply --sync-opencode` rewrites only the
  `llamacpp` provider's `limit.context`, with a timestamped backup.
- Pi sync: `llm apply --sync-pi` rewrites the local provider's
  `contextWindow` in `~/.pi/agent/models.json` (or `PI_CODING_AGENT_DIR`),
  with a timestamped backup. `--sync-clients` does both.

- [ ] **Step 3: Mark the spec approved**

Change `Status: draft (awaiting approval)` to `Status: approved`.

- [ ] **Step 4: Run the full suite and build**

Run: `python3 -m unittest discover -s tests -v`
Expected: all tests pass.

Run: `LMM_SKIP_LLAMA_CPP_UPGRADE=1 ./build.sh`
Expected: `Built /Users/davide/projects/llm/dist/llm` and the zipapp `help` smoke test passes.

- [ ] **Step 5: Manual verification on this Mac**

```bash
./dist/llm doctor
./dist/llm apply --dry-run
./dist/llm apply --dry-run --sync-clients
```

Expected: doctor shows per-model contexts and the storage/GPU lines; `apply --dry-run` shows qwen moving toward 128K when exact KV fits and inkling/glm kept or moved to 64K with a reason; the client dry-run lists the `limit.context`/`contextWindow` changes without writing.

- [ ] **Step 6: Commit**

```bash
git add llm/__init__.py README.md docs/superpowers/specs/2026-11-28-context-control-and-gpu-stats-design.md
git commit -m "docs: release 1.2.0 with context control and GPU stats"
```

---

## Self-review notes

- Every spec section maps to a task: registry overrides → 4; GGUF sizing → 1-2; budget/choice → 2; visibility/diagnosis → 5 and 7; GPU probing → 3; OpenCode sync → 6 and 7; commands → 5, 8, 9; docs/version → 10.
- Review Focus items map to tests: corrupt GGUF → Task 1 `test_truncated_file_returns_none` and Task 9 `test_apply_skips_missing_models`; loaded server → Task 2 `test_context_budget_discrete_uses_total_vram` and Task 9 restart tests; legacy `models.json` → Task 4 tests; no-GPU machines → Task 3 fallbacks and Task 7 missing-gpustat test; bad OpenCode config → Task 6 tests.
- Interfaces are consistent: `choose_context(model_size, current, machine, metadata, candidates, ceiling)` is used the same way in Tasks 2, 8, and 9; `Registry` is consumed by `load_profiles`, `set_context`, and `context_source`; `GpuStats` is consumed by `detect_hardware`, `doctor`, and `context_budget`.
