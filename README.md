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
| `llm use <model>` | Select a model; restart the service if it is running. |
| `llm start [--no-wait]` | Start the selected model and wait until healthy. |
| `llm stop` | Stop the model service. |
| `llm restart` | Restart the model service. |
| `llm logs [-n N] [-f]` | Show the last `N` lines of the server log, optionally follow. |
| `llm search <query>` | Search Hugging Face for GGUF models. |
| `llm get <repo>[:<quant>]` | Download, verify, and register a GGUF model into the cache. |
| `llm migrate-cache [--dry-run] [--verify] [--keep-old]` | Move legacy model directories into the cache. |
| `llm doctor` | Check llama.cpp, hardware, models, port, and the last failure. |
| `llm update [--dir PATH]` | Fetch, build, and install the latest source. |
| `llm version` | Show the installed version and whether it is the latest release. |
| `llm help [command]` | Show help for the CLI or a single command. |

Every command resolves model names by exact id or alias, then by a unique
prefix, then by a unique substring, so `llm use inkling` is enough when only
one model matches.

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

`--force` bypasses the refusal and downloads anyway. `--context` changes the
context size used for the estimate (and stored with the model), and
`--requires-llama-cpp VERSION` records a minimum llama.cpp version that
`llm doctor` reports against.

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

### Diagnosing problems

```sh
llm doctor
```

`llm doctor` reports the detected chip, memory, free disk, installed models,
whether `llama-server` is present and current, the service state, and a
one-line explanation of the last load failure.

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
│   ├── hardware.py                 # hardware detection and fit preflight
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
  estimates model-fit before a download.
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
