# LLM — design

Date: 2026-11-27
Status: approved

## Intent

The local llama.cpp model manager currently hides behind the graph-analysis CLI:
you start, stop, switch and inspect models with flags like
`graphify -select-llm-model=…` and `graphify -start-llm`. Its `list` command
prints configured profiles but never says which model is actually running, and a
failed load ends in a long timeout with the real cause buried in a 130 MB log.

Goal: give the model manager its own friendly command, `llm`
(alias `lmm`), that answers "what do I have, what is running, and why did the
last one fail", while leaving the graph tool untouched.

Concrete trigger: the downloaded `Inkling-2-Mini-UD-Q4_K_XL.gguf` fails to load
with `unknown model architecture: 'inkling2_loop'`. The installed llama.cpp is
Homebrew 1.0.0 (build 12359); Inkling 2 support landed in llama.cpp v1.1.0,
which Homebrew already offers. The download itself is complete and
manifest-verified.

## Goals

- `llm` / `lmm` with: `list`, `status`, `use`, `start`, `stop`,
  `restart`, `logs`, `doctor`, `help`.
- `list` shows per-model availability **and live state**: running (ready or
  loading), PID, uptime, or selected-but-failed, plus the last load-error
  explanation.
- `doctor` checks llama.cpp binary/version, model files, port, state dir, and
  turns known load failures into an actionable hint.
- Failed loads surface the log's cause and a hint in seconds, not after a
  60-minute wait.
- `graphify` and every existing flag keep working; old model flags delegate to
  the new CLI and print a short migration hint.
- Upgrade Homebrew llama.cpp to 1.1.0 and verify Inkling 2 starts and answers
  `/health`.

## Non-goals

- Renaming or changing the graph-analysis CLI (`graphify` stays as-is).
- Moving the 34 GB of GGUF files; profiles keep their existing paths.
- Adding model downloads; models are still fetched out-of-band.
- Multi-model concurrency; the service still runs one model on port 18081.

## Approaches considered

1. **Rename `graphify` itself.** Rejected: `graphify` is a real graph CLI used
   by skills; renaming breaks it and every existing invocation.
2. **Separate manager CLI with compatibility shims.** Chosen: clean UX without
   breaking the graph workflow.
3. **Rename only the service binary.** Rejected: the user-facing flags stay
   unfriendly, which is half the request.

Source lives in a small repo `~/projects/llm`, with entry points
symlinked into `~/.local/bin`. Tests and README have a home; the working system
is not restructured beyond the symlinks.

## Behaviour

### Command surface

```
lmm list                 # models + live status
lmm status               # selected model, service health, endpoint
lmm use <model>          # select; restart service if it is running
lmm start [--no-wait]    # start selected model (default: wait until ready)
lmm stop
lmm restart
lmm logs [-f] [-n N]     # tail server log
lmm doctor               # environment + last-failure diagnosis
lmm help [command]
```

Model arguments accept a unique prefix (`lmm use inkling`); ambiguous or
unknown prefixes list the valid choices. `use` selects only when the service is
stopped, and stops → selects → starts (waiting for health) when it is running.
`select` remains available as the internal select-only operation.

### `list` output

```
  MODEL                          SIZE      STATUS
* inkling-2-mini                16.4 GiB  selected · last start failed
  glm-4.7-flash                 16.3 GiB  downloaded · not running
  qwen2.5-7b-instruct            8.5 GiB  downloaded · not running

Service stopped.   Endpoint http://127.0.0.1:18081/v1
Last failure: unknown model architecture 'inkling2_loop'
→ llama.cpp 1.0.0 can't load it; 1.1.0 adds Inkling 2. Run: brew upgrade llama.cpp
```

- `*` marks the selected model; `▶` marks the running model.
- A running model shows `RUNNING · ready|loading · PID n · up 2m`.
- If the running model differs from the selected one, print a warning.
- Missing local file → `missing (path)` and no size.
- Log tail is read from the end of the file, never by loading it whole.

### Failure handling

`wait_ready` keeps polling health and liveness but also scans the tail of the
log for fatal lines. On `unknown model architecture: '<arch>'`, exit with a hint;
on "failed to load model" without a known architecture, show the last error line
and `lmm logs`. The known mapping is:

- `inkling2_loop` → requires llama.cpp >= 1.1.0 ("brew upgrade llama.cpp").

Progress messages print at most once per 15-second bucket (current code prints
every loop iteration whose integer second is divisible by 15).

### `doctor`

- `llama-server` path and parsed version/build.
- Compare installed version against each profile's optional
  `min_llama_cpp`; `inkling-*` requires 1.1.0.
- Homebrew availability (`brew outdated llama.cpp` / `brew info`) as a hint.
- Per-model: file(s) present, total size, architecture requirement status.
- Service state, listener on 18081, state dir writability, log size/rotation.
- Last failure explanation, same as `list`.

### Compatibility

- `graphify -start-llm|-stop-llm|-restart-llm|-status-llm` → `lmm` equivalents.
- `graphify -list-llm-model` → `lmm list`.
- `graphify -select-llm-model=X` → select only (preserves old semantics).
- Each prints one stderr hint: `hint: the model commands now live in 'lmm'`.
- `graphify-model-service` becomes a shell shim to the new CLI, accepting the
  legacy positional forms (`start <profile-id>`, `select <id>`, etc.) because
  the labeling adapter still calls it.
- `graphify` shim itself is unchanged.

### State and migration

- State moves to `~/.local/state/llm`.
  On first run, if the new dir does not exist and the legacy
  `graphify-model-service` dir does, move its contents (same filesystem,
  including `label-cache-v3`) and leave `server.log` in place for inspection.
- `server.log` rotates to `server.log.1` at 20 MB before the next append.
- Config module renamed to `llm_config.py`; a compatibility
  `graphify_llm_config.py` re-exports it for any older importer.

## Decided details

- Alias is `lmm` (checked free).
- Port and endpoint stay `127.0.0.1:18081` / `http://127.0.0.1:18081/v1`.
- Sampling/context per existing profiles; no behaviour change.
- Tests use the standard library `unittest` (pytest is not installed in the
  graphify virtualenv).
- Existing files receive timestamped backups before being replaced, matching the
  current convention.
- `min_llama_cpp` is metadata in the profile, not GGUF parsing.

## Left to the builder

- Exact table widths/wrapping and colour (plain text is fine).
- How to detect a foreign listener robustly (reuse `lsof` logic).
- Version comparison mechanics and brew command output parsing.
- Log rotation details and how `logs -f` is implemented.
- Test seams: allow an injectable state dir/runner so unit tests never touch the
  real service or port 18081.
