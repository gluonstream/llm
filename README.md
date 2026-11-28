# llm

**Author:** Davide Listello <davide.listello@gmail.com>

A small command-line tool for managing a persistent local
[llama.cpp](https://github.com/ggerganov/llama.cpp) server and the GGUF models
behind it. It keeps one model resident behind an OpenAI-compatible endpoint,
and gives you simple commands to search, download, select, start, stop, and
diagnose it.

The command is installed as **`llm`**, with **`lmm`** as an alias.
All examples use `llm`; either name behaves identically.

```
Endpoint: http://127.0.0.1:18081/v1
```

## Why

The tool was built to manage the persistent local model used by `graphify`,
but it is a standalone CLI. It handles the tedious parts of running a local
model:

- one selected model, started and stopped as a background service;
- model discovery on Hugging Face, resumable downloads, and checksum checks;
- a preflight estimate of whether a model will fit in RAM and on disk;
- safe process handling, so it never kills an unrelated process on the port;
- actionable diagnostics when llama.cpp rejects a model.

## Requirements

- macOS on Apple Silicon, or Arch Linux.
- Python 3.9 or newer on `PATH` (the CLI uses only the standard library).
- Bash and `git` for `llm update`.
- `llama-server` from Homebrew on macOS, at `/opt/homebrew/bin/llama-server`:

  ```sh
  brew install llama.cpp
  ```

  On Arch, install `llama-server` from an AUR package or build llama.cpp from
  source. For example:

  ```sh
  # NVIDIA/CUDA:
  yay -S llama.cpp-cuda-git
  # or CPU/other build:
  # yay -S llama.cpp-git
  sudo pacman -S lsof pciutils
  ```

- On Arch, `lsof` is required for safe process and port checks.
- Optional, for richer hardware reporting: `pciutils` (`lspci`), `nvidia-smi`,
  and `btrfs-progs` (accurate free space on Btrfs).
- If `llama-server` is not on `PATH`, set `LMM_LLAMA_SERVER` to its path.

No third-party Python packages are required.

## Installation

Build the single-file executable and install it into `~/.local/bin`:

```sh
./build.sh
./install.sh
```

- `build.sh` runs the unit test suite, then produces `dist/llm`, a
  self-contained Python zipapp containing compiled bytecode. It still needs
  `python3` on `PATH`; set `LMM_SKIP_TESTS=1` to skip the tests. On macOS,
  `build.sh` may also run `brew upgrade llama.cpp`; set
  `LMM_SKIP_LLAMA_CPP_UPGRADE=1` to skip that upgrade.
- `install.sh` installs `dist/llm` as `~/.local/bin/llm` and creates the
  `lmm` alias.
- Set `LMM_BIN_DIR` to install somewhere else:

  ```sh
  LMM_BIN_DIR="$HOME/bin" ./install.sh
  ```

Make sure the install directory is on your `PATH`:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

Both scripts are idempotent and can be re-run at any time. `build.sh` writes
only to `build/` and `dist/`; `install.sh` replaces only the two command
names above.

## Updating

```sh
llm update
```

`llm update` clones the repository into `/private/tmp/llm-update` on macOS
(or `/tmp/llm-update` on Arch), pulls the latest `main` if the checkout
already exists, runs `build.sh`, and installs the new binary with
`install.sh` into `~/.local/bin` (or `LMM_BIN_DIR`). Set `LMM_UPDATE_DIR` or
pass `--dir` to use another checkout directory.

From a source checkout, `./update.sh` runs the same flow.

## Quick start

```sh
llm doctor                                    # check llama.cpp and hardware
llm list                                      # show models and what is running
llm search inkling                            # find GGUF repositories
llm get unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_XL
llm use inkling                               # select and start it
llm status                                    # confirm the endpoint
```

Then point any OpenAI-compatible client at `http://127.0.0.1:18081/v1`.

## Commands

| Command | Description |
| --- | --- |
| `llm list` | List models, their size, and which one is selected/running. |
| `llm status` | Show the selected model, service health, and endpoint. |
| `llm use <model> [--context-size N]` | Select a model; restart the service if it is running. |
| `llm start [--no-wait]` | Start the selected model and wait until healthy. |
| `llm stop` | Stop the model service. |
| `llm restart` | Restart the model service. |
| `llm logs [-n N] [-f]` | Show the last `N` lines of the server log, optionally follow. |
| `llm search <query>` | Search Hugging Face for GGUF models. |
| `llm get <repo>[:<quant>]` | Download, verify, and register a GGUF model into the cache. |
| `llm migrate-cache [--dry-run] [--verify] [--keep-old]` | Move legacy model directories into the cache. |
| `llm remove <model> [--apply]` | Delete one model's files from the cache (dry run by default). |
| `llm context [<model>] [<tokens>|auto]` | Show all contexts, or show/set one model's context window. |
| `llm analyze [--context N]` | Show the KV-cache memory for each model at f16/q8_0/q4_0. |
| `llm apply [<model>...]` | Size context and runtime flags for installed models and write them back. |
| `llm doctor` | Check llama.cpp, hardware, models, port, and the last failure. |
| `llm update [--dir PATH]` | Fetch, build, and install the latest source. |
| `llm version` | Show the installed version and whether it is the latest release. |
| `llm help [command]` | Show help for the CLI or a single command. |

Every command resolves model names by exact id or alias, then by a unique
prefix, then by a unique substring, so `llm use inkling` is enough when only
one model matches. `llm remove` is the exception: it matches the exact id or
alias only, never a prefix or substring, so you never delete the wrong model.

### Inspecting models

```sh
llm list
llm status
```

`llm list` marks the selected model with `*` and the running model with `▶`,
and reports the status of each entry. `llm status` prints one service line:

```
Selected: inkling-2-mini-gguf-ud-q4-k-xl
Service:  RUNNING · ready · PID 4242 · up 00:05:12
Endpoint: http://127.0.0.1:18081/v1
```

### Selecting and running a model

```sh
llm use inkling          # select by unique prefix and start
llm start                # start the selected model and wait for health
llm start --no-wait      # start in the background and return immediately
llm restart
llm stop
llm logs -n 50
llm logs -f
```

`llm use` selects the model and, if the service was already running, restarts
it on the new model. `llm start` waits for the health endpoint before
returning; use `--no-wait` for scripts that prefer to poll themselves.

By default the context size adapts to the memory that constrains the model:
VRAM on a discrete GPU, unified RAM on Apple Silicon, and system RAM on a
CPU-only host. Up to 12 GiB gets 32k, larger machines get 64k, and more than
128 GiB gets 128k. Override it for one selection with:

```sh
llm use qwen2.5-7b-instruct --context-size 32768
llm use qwen2.5-7b-instruct   # back to the hardware default
```

### Searching and downloading models

```sh
llm search inkling
llm search qwen --limit 5
llm search --all qwen --limit 10
llm get unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_XL
llm get unsloth/Inkling-2-Mini-GGUF --file Inkling-2-Mini-UD-Q4_K_XL.gguf
llm get unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_XL --name inkling
```

- `search` shows GGUF repositories by default; `--all` includes everything.
- `get` accepts `repo:QUANT` or an exact `--file`. If the repository has more
  than one GGUF file and you specify neither, it lists the candidates and
  exits.
- Downloads resume from a `.part` file, verify the Hugging Face SHA-256, and
  register the model automatically.
- `get` writes the same layout Hugging Face uses: the file is stored once as
  `blobs/<sha256>`, a relative symlink is created at
  `snapshots/<commit>/<filename>`, and the requested revision is recorded in
  `refs/<revision>`. Re-running the same command reuses the existing blob.
- A provenance manifest is written to
  `~/.local/state/llm/manifests/<model-id>.json`; nothing is added to the
  snapshot directory.
- Set `HF_TOKEN` for gated repositories.
- Split GGUF sets (`-00001-of-00002`) are not supported by `get` yet.
  `migrate-cache` and discovery still handle existing complete split sets.

Before downloading, `get` estimates whether the model fits in RAM (including
context overhead) and on disk. If it does not, the command prints a warning
and exits without downloading:

```
<!> Model M-Q8.gguf may not fit on this machine <!>
  model does not fit: needs ~33.5 GiB (30.0 GiB file + 3.5 GiB overhead ...)
Use --force to download anyway; the model may fail to load.
```

`--force` bypasses the refusal and downloads anyway. By default `get` stores
the context size recommended for the detected hardware; `--context` overrides
it and stores a fixed value. `--requires-llama-cpp VERSION` records a minimum
llama.cpp version that `llm doctor` reports against.

### Model storage and discovery

Models live in the standard Hugging Face hub cache. `llm` checks these
locations, in order:

1. `HF_HUB_CACHE`
2. `HUGGINGFACE_HUB_CACHE`
3. `$HF_HOME/hub`
4. `~/.cache/huggingface/hub`

Each repository gets a `models--org--repo` directory with `blobs/`, `refs/`,
and `snapshots/`. `llm get` and `llm migrate-cache` both write this layout, so
files downloaded by `llm` can be reused by other Hugging Face tools (and vice
versa). There is no separate `~/.local/share/llm/models` store.

On startup, `llm` discovers complete GGUF models already present in the cache
and lists them alongside the built-in profiles. A model is only picked up when
all shards of a split set are present. Set `LMM_NO_CACHE=1` to skip discovery
(for tests or a deliberately restricted view).

#### Cache locations

| Tool | macOS | Arch Linux | Windows |
| --- | --- | --- | --- |
| Hugging Face (`llm`, Unsloth, HF tools) | `~/.cache/huggingface/hub/` | `~/.cache/huggingface/hub/` | `%USERPROFILE%\.cache\huggingface\hub\` (e.g. `C:\Users\<username>\.cache\huggingface\hub\`) |
| llama.cpp `-hf` downloader | `~/Library/Caches/llama.cpp/` | `~/.cache/llama.cpp/` | `%LOCALAPPDATA%\llama.cpp\` |

`llm` follows the standard Hugging Face location,
`~/.cache/huggingface/hub/`, and reads/writes that layout. To share the same
model files with Unsloth Desktop/Studio or other Hugging Face tools, set
`HF_HOME` or `HF_HUB_CACHE` **before** launching them and `llm`. `llm` also
honours `HUGGINGFACE_HUB_CACHE`; the full precedence is listed above.

llama.cpp's own `-hf` downloader uses a separate cache controlled by
`LLAMA_CACHE` (macOS `~/Library/Caches/llama.cpp/`, Linux
`~/.cache/llama.cpp/`, Windows `%LOCALAPPDATA%\llama.cpp\`). `llm` passes
explicit model paths and does not use that cache; set `LLAMA_CACHE` only when
you use `llama.cpp -hf` directly.

#### Supported platforms

`llm` supports macOS on Apple Silicon and Arch Linux. Other Linux distributions
are not currently planned, and Microsoft Windows is not supported at this time.
The Windows cache paths above are shown for reference only.

### Migrating legacy downloads

Older versions of `llm` (and `graphify`) stored models under
`~/.local/share/graphify-*`. Move them into the cache without downloading
again:

```sh
llm migrate-cache --dry-run     # show the plan
llm migrate-cache               # hardlink/copy and remove the old directory
llm migrate-cache --verify      # hash files before and after (slow)
llm migrate-cache --keep-old    # leave the old directories in place
```

Migration reuses an existing blob when its size matches, hardlinks when
possible, falls back to copying across filesystems, and only removes a legacy
directory after every file in it migrated.

### Removing a model

`llm remove` deletes a single model's files from the Hugging Face cache. It
matches the name exactly (case-insensitive id or alias); a prefix or substring
is rejected with a list of exact choices so you never remove the wrong quant.

```sh
llm remove inkling-2-mini          # dry run: print the plan, delete nothing
llm remove inkling-2-mini --dry-run
llm remove inkling-2-mini --apply  # actually delete the files
```

- Only the named model's files are removed. For a split set, every shard in the
  snapshot is collected; other quants in the same snapshot are left alone.
- Blobs are shared between snapshots by design, so a blob that is still
  referenced by another snapshot is kept and reported; only the symlink is
  removed. Empty snapshot directories, `refs/<revision>`, and the repository
  directory are cleaned up once nothing references them.
- The running model is stopped first if its alias matches the model being
  removed. Built-in profiles keep their curated entry (the file simply shows as
  missing and any context/runtime override is dropped), user models are removed
  from `models.json`, and cache-discovered models disappear on the next
  discovery.
- `--apply` asks you to retype the exact model name (the alias or id,
  case-sensitive) before it deletes anything; a mismatch cancels and deletes
  nothing. The default and `--dry-run` never prompt.

### Sizing the context window

`llm context` shows and sets the context window per model, and `llm apply`
sizes every installed model in one step.

```sh
llm context                      # table of every model and its context
llm context qwen                 # one model's value and source (built-in/user/override)
llm context qwen 65536           # set an explicit value
llm context qwen auto            # size from the GGUF and machine, then write it
llm context qwen auto --dry-run
llm apply                        # size every installed model
llm apply --dry-run
```

`llm apply` reads each model's real GGUF metadata (layers, KV heads, head
dims) to compute the exact KV-cache size, adds the weights and a conservative
compute margin, and compares it to the machine's memory budget. It picks 128K
when that fits with headroom, otherwise 64K when that fits, otherwise leaves
the current value and says why:

```
MODEL                 FILE       CONTEXT          EST. NEEDED   ACTION
qwen2.5-7b-instruct   4.4 GiB    64K → 128K       9.3 GiB       updated
inkling-2-mini       16.4 GiB    8K → 128K        24.0 GiB      updated
glm-4.7-flash        16.3 GiB    32K → 128K       24.2 GiB      updated
```

- `--context N` sets an upper bound (a ceiling) for the chooser; it never
  forces a value that does not fit. For a hard per-model value, use
  `llm context MODEL N`.
- `--dry-run` prints the plan and performs no writes, no client sync, and no
  restart.
- `--no-restart` writes the new contexts but skips the service restart and
  prints `llm restart` for you to run.
- When the running model's context changed (or the server's real `n_ctx`
  drifted from the profile), `apply` restarts the service to pick it up.

### Runtime tuning

`llm apply` now chooses context and llama.cpp runtime flags together: KV-cache
quantization, flash attention, batch/ubatch sizing, prompt-cache reuse, and a
`--fit` headroom guard. A policy picks sensible defaults, and each flag is
written only when the installed `llama-server --help` supports it (old builds
silently fall back to f16 KV, 512/256 batches, and no cache-reuse).

| Policy | KV K/V | Context candidates | batch/ubatch | cache-reuse | fit |
| --- | --- | --- | --- | --- | --- |
| `conservative` | `f16` / `f16` | 64K | 1024 / 256 | 256 | on, 2048 |
| `balanced` (default) | `q8_0` / `q8_0` | 64K | 1024 / 256 | 256 | on, 2048 |
| `extended` | `q8_0` / `q8_0` | 128K | 1024 / 256 | 256 | on, 2048 |
| `aggressive` | `q4_0` / `q4_0` | 128K | 1024 / 256 | 256 | on, 2048 |

- `conservative` is maximum accuracy at the cost of more KV RAM: f16 K and V, 64K.
- `balanced` cuts KV RAM roughly in half with near-identical perplexity: q8_0 K
  and V, 128K or 64K depending on what fits.
- `extended` keeps q8_0 but prefers 128K and larger batches when memory allows.
- `aggressive` shrinks KV RAM to about a quarter (q4_0) for the largest context,
  accepting some precision/recall loss.

The memory story: a quantized KV cache stores fewer bytes per value than f16 —
`q8_0` is 34 bytes per 32 values (≈ 0.53×) and `q4_0` is 18 bytes per 32 values
(≈ 0.28×), so `q8_0` roughly halves the KV cache and `q4_0` roughly quarters it.
Because quantized KV needs flash attention, any non-f16 KV forces `-fa on`; if
the server does not support that, `apply` keeps f16 and reports it.

Select a policy or override individual fields:

```sh
llm apply --runtime-policy extended
llm apply --runtime-policy aggressive
llm apply --kv q8_0                 # both K and V
llm apply --kv-k q5_0 --kv-v q4_0   # asymmetric K/V (slightly safer than q4_0/q4_0)
llm apply --batch 2048 --ubatch 512
llm apply --fit-target 4096         # --fit margin in MiB (default 2048)
llm apply --no-fit                  # disable the fit headroom guard
llm apply --no-cache-reuse          # disable prompt-cache reuse
```

`--fit` preserves a headroom margin (2 GiB by default) so a context that only
just fits is not chosen; `--fit-target` tunes that margin. Usage history from
`server.log` right-sizes the context: `apply` reads the observed peak token
count and keeps only candidates that hold peak + 50%, rounded up to the nearest
candidate, instead of always reaching for the model's native maximum.

`apply` reports the plan one row per model, with the chosen KV type and
batch/ubatch:

```
MODEL                  FILE      CONTEXT         KV     BATCH     EST. NEEDED   ACTION
qwen2.5-7b-instruct    4.4 GiB   64K            q8_0   1024/256   6.5 GiB      updated
inkling-2-mini        16.4 GiB   8K → 64K       q8_0   1024/256  21.1 GiB      updated
glm-4.7-flash         16.3 GiB   32K → 64K      q8_0   1024/256  21.1 GiB      updated
```

`llm context MODEL` prints the effective runtime summary
(`q8_0 KV · 1024/256 · fit 2048 · cache-reuse 256`), and `llm doctor` adds the
same runtime line plus the observed peak from the log.

### Analyzing KV memory

```sh
llm analyze                # every cached model at 64K context
llm analyze --context 128K
```

`llm analyze` prints one row per model (built-ins plus cache-discovered models)
with the KV-cache size at f16, q8_0, and q4_0 precision, the saving from q8_0,
and whether the q8_0 estimate fits the machine's memory budget:

```
MODEL                  SIZE    CTX     f16     q8_0    q4_0    SAVE    FIT
qwen2.5-7b-instruct    4.4G    64K     3.50G   1.86G   0.98G   46.9%   yes
```

### Generation and speculative decoding

`apply` also stores sampling and speculative-decoding defaults in the runtime
block, and emits them only when the server supports them:

```sh
llm apply --top-k 40 --predict -1          # sampling defaults (--top-k, -n)
llm apply --spec-type draft-mtp --draft MODEL
llm apply --draft MODEL                    # implies --spec-type draft-simple
llm apply --cmoe                           # keep MoE expert weights on CPU
llm apply --reasoning on --reasoning-budget 4096
llm apply --device 0,1                     # -dev list
```

`--spec-type` is one of `none`, `draft-simple`, `draft-mtp`, or `draft-eagle3`;
`--draft MODEL` resolves another profile and stores its cache path. `--cmoe`
keeps MoE expert weights on CPU (`-cmoe`), and `--reasoning`/`--reasoning-budget`
map to `-rea` and `--reasoning-budget`.

### Keeping OpenCode, Pi, and Codex in sync

`llm apply --sync-opencode` rewrites only the local `llamacpp` provider's
`limit.context` for each installed model in `~/.config/opencode/opencode.json`
(or `LMM_OPENCODE_CONFIG`, or `$XDG_CONFIG_HOME/opencode/opencode.json`),
leaving every other provider and setting untouched. A timestamped backup is
written first.

`llm apply --sync-pi` rewrites the local provider's `contextWindow` for each
installed model in `~/.pi/agent/models.json` (or `PI_CODING_AGENT_DIR`), again
with a timestamped backup.

`llm apply --sync-codex` edits `~/.codex/config.toml` (or
`CODEX_HOME/config.toml`) so the selected model points at the local llama.cpp
endpoint: it sets the top-level `model` to the selected alias and
`model_provider = "llamacpp"`, and writes a `[model_providers.llamacpp]` section
with `name = "llama.cpp"`, `base_url = "<llm URL>/v1"`, `wire_api = "responses"`,
`requires_openai_auth = false`, and `supports_websockets = false`. The edit is a
targeted line rewrite, so comments, plugins, and marketplaces survive; a
timestamped backup is written and nothing changes when the file is already in
sync. `--sync-clients` covers OpenCode, Pi, and Codex.

No client is ever written unless you ask; `llm doctor` reports mismatches and
points at the right flag.

### GPU statistics

On Linux with an NVIDIA GPU, `llm doctor` reports precise per-GPU VRAM,
utilisation, and per-process usage when `gpustat` is available; otherwise it
falls back to `nvidia-smi`, AMD sysfs, or `lspci` name-only. On macOS, it uses
`system_profiler` as before.

`gpustat` is optional. To install it on Arch:

```sh
sudo pacman -S python-pipx
pipx ensurepath
pipx install gpustat
```

Then reopen the shell (so `pipx`'s bin directory is on `PATH`) and re-run
`llm doctor`. The fallbacks keep working without it.

### Diagnosing problems

```sh
llm doctor
```

`llm doctor` reports the detected chip, memory, free disk, installed models,
whether `llama-server` is present and current, the service state, and a
one-line explanation of the last load failure. It also shows each model's
context, per-GPU statistics, and any mismatch between `llm`, the running
server, OpenCode, or Pi.

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LMM_STATE_DIR` | `~/.local/state/llm` | State, registry, pid file, and logs. |
| `LMM_LEGACY_STATE_DIR` | `~/.local/state/graphify-model-service` | Older state moved into place once. |
| `LMM_PORT` | `18081` | Port the server binds and the CLI checks. |
| `LMM_URL` | `http://127.0.0.1:18081` | Base URL used for health checks. |
| `LMM_BIN_DIR` | `~/.local/bin` | Install destination for `install.sh`. |
| `LMM_BUILD_DIR` | `<repo>/build` | Staging directory for `build.sh`. |
| `LMM_DIST_DIR` | `<repo>/dist` | Output directory for `dist/llm`. |
| `LMM_SKIP_TESTS` | unset | Set to `1` to skip the test suite during `build.sh`. |
| `LMM_UPDATE_DIR` | unset | Override the self-update checkout directory. |
| `LMM_SKIP_LLAMA_CPP_UPGRADE` | unset | Set to `1` to skip the build-time llama.cpp package check. |
| `LMM_LLAMA_SERVER` | detected | Path to `llama-server`; overrides `PATH` detection. |
| `LMM_NO_CACHE` | unset | Set to `1` to skip HF cache discovery. |
| `HF_HUB_CACHE` | unset | Hugging Face cache root; highest precedence. |
| `HUGGINGFACE_HUB_CACHE` | unset | Hugging Face cache root; used when `HF_HUB_CACHE` is unset. |
| `HF_HOME` | unset | Hugging Face home; the cache is `$HF_HOME/hub`. |
| `HF_TOKEN` | unset | Token for gated Hugging Face repositories. |

## Where things live

| Path | Contents |
| --- | --- |
| `~/.local/state/llm/` | State directory. |
| `.../selected-model.json` | The currently selected model id. |
| `.../models.json` | Models registered by `llm get`. |
| `.../manifests/<id>.json` | Provenance for each `llm get` download (URL, commit, size, SHA-256). |
| `.../process.json` | PID and process identity of the managed server. |
| `.../server.log` | Server output; rotated to `server.log.1` at 20 MiB. |
| `.../lock` | Lock file serializing start/stop/select operations. |
| `~/.cache/huggingface/hub/` | Default HF cache root; see the precedence list above. |
| `.../models--org--repo/blobs/<sha256>` | Each model file, stored once under its SHA-256. |
| `.../models--org--repo/snapshots/<commit>/` | Relative symlinks to blobs, one tree per commit. |
| `.../models--org--repo/refs/<revision>` | Revision name to commit mapping (`main` by default). |
| `~/.local/share/graphify-*` | Legacy model files; move them with `llm migrate-cache`. |

Built-in model profiles are defined in `llm/config.py` and resolve to their
snapshot in the cache (or the path where the snapshot would be, so `doctor`
can report it missing). Models downloaded with `llm get` live in the registry
above and can be added or removed without touching the built-ins.

## Architecture

The CLI is a thin layer over a handful of focused modules:

```
llm/
├── build.sh                        # build dist/llm (zipapp)
├── install.sh                      # install llm + aliases
├── update.sh                       # fetch, build, and install latest source
├── bin/
│   └── llm                         # source-tree launcher
├── llm/
│   ├── cli.py                      # argparse commands and handlers
│   ├── config.py                   # paths, profiles, registry, selection
│   ├── hfcache.py                  # HF cache layout, discovery, snapshots
│   ├── host.py                     # OS and external tool discovery
│   ├── service.py                  # llama-server lifecycle and health
│   ├── hub.py                      # Hugging Face search and downloads
│   ├── hardware.py                 # hardware/GPU detection and fit preflight
│   ├── gguf.py                     # GGUF metadata and KV-cache sizing
│   ├── runtime.py                  # runtime tuning policies and overrides
│   ├── analysis.py                 # per-model KV-memory analysis rows
│   ├── usage.py                    # observed context peak from server.log
│   ├── opencode.py                 # OpenCode provider sync (limit.context)
│   ├── pi.py                       # Pi provider sync (contextWindow)
│   ├── codex.py                    # Codex provider sync (config.toml)
│   ├── update.py                   # self-update checkout, build, install
│   ├── diagnose.py                 # llama.cpp version and failure hints
│   └── render.py                   # plain-text tables and status output
├── tests/                          # unittest suite
└── docs/superpowers/               # design specs and implementation plans
```

- **`cli.py`** defines the argument parser, dispatches each command, and maps
  exceptions to a non-zero exit code with a one-line error.
- **`config.py`** owns every path and the model registry. It merges built-in
  profiles with the user registry, validates profiles, resolves names, and
  stores the selection atomically.
- **`service.py`** supervises `llama-server`. It serializes mutations with a
  file lock, records the process identity (start time plus command line) so a
  recycled PID is never mistaken for the server, refuses to adopt an
  unrelated listener on the port, rotates the log, and waits for `/health`.
- **`hub.py`** talks to the Hugging Face API, selects GGUF files, resumes
  partial downloads, verifies SHA-256, and writes blobs and provenance
  manifests.
- **`hfcache.py`** owns the cache layout: root precedence, snapshot and ref
  resolution, snapshot symlinks, discovery of complete GGUF models, and
  profile synthesis. `llm migrate-cache` uses the same helpers to import
  legacy directories.
- **`host.py`** detects the OS and locates external tools such as `llama-server`
  and `lsof`, honoring `LMM_LLAMA_SERVER`.
- **`hardware.py`** detects RAM, unified memory, GPU, and free disk, then
  estimates model-fit before a download. It also probes per-GPU statistics via
  `gpustat`/`nvidia-smi`/sysfs and owns the context budget and chooser.
- **`gguf.py`** parses GGUF metadata (layers, KV heads, head dims) and computes
  the exact KV-cache size at any KV precision, used by `llm apply`, `llm
  context auto`, and `llm analyze`.
- **`runtime.py`** holds the four runtime policies, applies per-flag overrides,
  and filters context candidates by the observed usage peak.
- **`analysis.py`** builds the per-model KV-memory rows for `llm analyze`.
- **`usage.py`** reads `server.log` to find the observed peak token count.
- **`opencode.py`** rewrites only the OpenCode `llamacpp` provider's
  `limit.context` when `llm apply --sync-opencode` is requested, with a backup.
- **`pi.py`** rewrites only the Pi local provider's `contextWindow` when
  `llm apply --sync-pi` is requested, with a backup.
- **`codex.py`** rewrites only the `model`/`model_provider` and
  `[model_providers.llamacpp]` lines in `~/.codex/config.toml` when
  `llm apply --sync-codex` is requested, with a backup.
- **`update.py`** implements `llm update`: clone or pull a checkout, then run
  `build.sh` and `install.sh`.
- **`diagnose.py`** parses `llama-server --version`, Homebrew status, and known
  log failures into actionable messages.
- **`render.py`** contains the plain-text formatting for tables and status.

On Arch, hardware reporting is best-effort: CPU and RAM are reliable; AMD and
NVIDIA VRAM are read from sysfs or `nvidia-smi`; Intel integrated GPUs report
shared system RAM.

The lifecycle is straightforward: `llm use` (or `select`) writes
`selected-model.json`; `llm start` builds the `llama-server` argument list from
that profile, launches it detached with output in `server.log`, records its
identity in `process.json`, and polls `http://127.0.0.1:18081/health` until
ready. `llm stop` terminates only the process whose recorded identity still
matches.

## Build and install internals

`build.sh` runs the unit tests, then copies the package into `build/app`,
compiles it to legacy-layout `.pyc` files with `compileall -b`, adds a
`__main__.py`, and packages
everything as `dist/llm` with `python -m zipapp` and a
`#!/usr/bin/env python3` shebang. The zip keeps the `.py` sources next to the
bytecode, so the command still works if the interpreter is upgraded.

`install.sh` copies the artifact to a temporary file in the destination and
renames it into place, so a running command never sees a partial binary. It
then creates a relative symlink for the `lmm` alias. Point `LMM_BIN_DIR` at
another directory to install elsewhere.

## Compatibility with graphify

`graphify -start-llm`, `-stop-llm`, `-restart-llm`, `-status-llm`,
`-list-llm-model`, and `-select-llm-model=MODEL` keep working as compatibility
shims and print a hint pointing at `llm`. The shims
`graphify-model-service` and `graphify_llm_config.py` in `~/.local/bin` are
thin wrappers into this repository, and `graphify-local-model` delegates the
old `-start-llm` family of flags to the CLI.

## Development

Run the test suite:

```sh
python3 -m unittest discover -s tests -v
```

Rebuild and reinstall after changing the source:

```sh
./build.sh
./install.sh
```

For a live development loop you can also run the source-tree launcher
directly, without building:

```sh
./bin/llm help
```

## Troubleshooting

Start with:

```sh
llm doctor
llm logs -n 100
```

- **`unknown model architecture`** — the installed llama.cpp is too old. For
  example, Inkling 2 (`inkling2_loop`) requires llama.cpp >= 1.1.0. Upgrade it
  for your platform:

  ```sh
  # macOS (Homebrew)
  brew upgrade llama.cpp

  # Arch: update the AUR package, or rebuild llama.cpp from source
  yay -S llama.cpp-cuda-git
  ```

- **Inkling 2 returns empty content with a very small `max_tokens`** — it
  emits a short reasoning preamble first. Give it enough output tokens, or
  pass `chat_template_kwargs: {"reasoning_effort": "none"}`.

- **Context overflow: `request (N tokens) exceeds the available context size
  (M tokens)`** — the server's context window is smaller than the request (a
  common cause is a client such as OpenCode believing the window is 64K while
  the server runs at 8,192). `llm doctor` explains this and points at
  `llm apply` (or `llm context <model> N`) to raise it, followed by a
  restart.

- **`port 18081 is held by an unmanaged PID`** — another process owns the
  port. Stop that process, or set `LMM_PORT` (and `LMM_URL`) to a free port.

- **A download is refused for size** — the model is estimated not to fit.
  Use `--force` to proceed anyway.

- **A gated repository returns HTTP 401/403** — set `HF_TOKEN`.

## Design documents

The design specs and implementation plans live in `docs/superpowers/`:

- `docs/superpowers/specs/` — feature designs.
- `docs/superpowers/plans/` — implementation plans.

## License

Copyright (C) 2026 Davide Listello <davide.listello@gmail.com>.

This repository contains the free, open-source Python implementation of `llm`.
It is licensed under the [GNU Affero General Public License, version 3 only
(AGPL-3.0-only)](LICENSE). You may use, modify, and distribute it under those
terms.

### Commercial licensing

If you want to use this code in a proprietary product or a hosted/cloud service
without complying with AGPL-3.0's source-sharing obligations, you need a
separate written commercial license from the copyright holder. Contact Davide
Listello at <davide.listello@gmail.com>.

The future Rust implementation is a separate product with its own commercial
license; it is not part of this repository.

### Contributions

Contributions are welcome. Every contributor must agree to the
[Contributor License Agreement](CLA.md); this is what lets the project offer
the same code under both AGPL-3.0-only and a commercial license. See
[CONTRIBUTING.md](CONTRIBUTING.md).

This project is licensed under the [GNU Affero General Public License v3.0
only (AGPL-3.0-only)](LICENSE).
