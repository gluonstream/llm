# Self-update and full macOS/Arch Linux support — design

Date: 2026-11-27
Status: approved

## Intent

Add self-update to `llm` and make the project genuinely usable on Arch Linux as
well as macOS. The three shell scripts, the Python CLI, hardware/diagnostics,
and the update flow must all handle both operating systems. Linux is treated
as Arch Linux; other Linux distributions are not a target.

## Goals

- Add `llm update`.
- Add `./update.sh` as a thin wrapper over the same implementation.
- Detect the OS in `build.sh`, `install.sh`, and `update.sh`.
- Exit with `unsupported OS` on anything other than macOS or `linux-gnu*`.
- Use `/private/tmp/llm-update` on macOS and `/tmp/llm-update` on Arch.
- Allow `LMM_UPDATE_DIR` or `llm update --dir PATH` to override the checkout.
- Clone when the checkout is missing; otherwise run `git pull --ff-only`.
- Run `./build.sh`, then `./install.sh`.
- Make the full CLI work on Arch:
  - find `llama-server` through `LMM_LLAMA_SERVER` or `PATH`;
  - find `ps` and `lsof` through `PATH`;
  - detect Arch CPU, RAM, GPU, and disk for `llm doctor` and preflight;
  - report Arch package status when `llama-server` belongs to a pacman package;
  - use Arch-appropriate llama.cpp upgrade hints.
- Best-effort keep `llama.cpp` current during `build.sh`:
  - macOS/Homebrew: `brew upgrade llama.cpp`.
  - Arch: detect `llama-server` and print AUR/source guidance. Do not run
    pacman automatically because the official repos have no `llama.cpp`
    package and AUR package names vary.
- Install to `LMM_BIN_DIR`, or `~/.local/bin` by default.
- Fail with an actionable message; never install after a failed build.

## Non-goals

- No Windows support.
- No Linux distributions other than Arch Linux.
- No automatic pacman or AUR install/upgrade.
- No release artifacts, checksums, or signature verification.
- No branch, tag, or commit selection.
- No forced llama.cpp upgrade: failure is a warning, not a build failure.
- No pushing to GitHub.
- No GPU-specific tuning beyond detection and reporting.
- No root-required probes (`lshw`, privileged Btrfs queries); detection must
  work as an ordinary user and fall back cleanly.

## Approaches

1. **Shell-only cross-platform scripts.** Build/install/update work on both
   OSes, but the CLI still hardcodes macOS paths. Rejected because Arch would
   build an unusable tool.
2. **Shared Python host layer + cross-platform scripts + self-update.** One
   OS layer used by scripts and runtime. Chosen.
3. **Separate Arch port.** Duplicates the project and drifts. Rejected.

## Design

### OS detection in shell scripts

Each of `build.sh`, `install.sh`, and `update.sh` gets the same block:

```bash
detect_os() {
    local value="${OSTYPE:-}"
    if [[ -z "$value" ]]; then
        case "$(uname -s)" in
            Darwin) value=darwin ;;
            Linux) value=linux-gnu ;;
        esac
    fi
    case "$value" in
        darwin*) OS=macos ;;
        linux-gnu*) OS=arch ;;
        *)
            echo "unsupported OS: ${value:-$(uname -s)} (supported: macOS, Arch Linux)" >&2
            exit 1
            ;;
    esac
}
detect_os
```

`darwin26.0` matches `darwin*`. `linux-gnu` matches `linux-gnu*`. The
`uname -s` fallback covers non-interactive shells where `OSTYPE` is unset.

### `build.sh`

- Detect the OS before doing anything else.
- On macOS, when `brew` exists and `llama.cpp` is installed, attempt
  `brew upgrade llama.cpp`. Missing Homebrew, a missing formula, or a failed
  upgrade only produces a warning.
- On Arch, if `llama-server` is on `PATH`, print its version. If not, print:
  `llama-server not found; install llama.cpp (AUR, e.g. yay -S llama.cpp-git)
  or set LMM_LLAMA_SERVER`.
- `LMM_SKIP_LLAMA_CPP_UPGRADE=1` skips the package-check step on both OSes.
- Continue with the existing test/build/zipapp flow unchanged.

### `install.sh`

- Detect the OS and keep the same install destination and replacement logic on
  both systems.
- On Arch, warn if `llama-server` is not on `PATH`.
- No pacman/brew calls from `install.sh`.

### `update.sh`

- Detect the OS.
- Set `LMM_UPDATE_DIR` to `/private/tmp/llm-update` on macOS or
  `/tmp/llm-update` on Arch when the user has not set it.
- Run `python3 -m llm.update "$@"` from the checkout root.

### `llm/host.py` (new)

One place for OS and tool discovery:

- `IS_MACOS`, `IS_LINUX` from `sys.platform`.
- `find_tool(name, fallback)` uses `shutil.which(name)` then the fallback.
- `find_llama_server(env=None, which=None)` precedence:
  1. `LMM_LLAMA_SERVER`;
  2. `shutil.which("llama-server")`;
  3. `/opt/homebrew/bin/llama-server` on macOS or `/usr/bin/llama-server`
     on Arch.
- `llama_cpp_upgrade_hint()` returns the macOS brew command or an Arch/AUR or
  source-build hint.

### Runtime portability

- `llm/config.py` exposes:
  - `LLAMA_SERVER = host.find_llama_server()`;
  - `PS = host.find_tool("ps", "/bin/ps")`;
  - `LSOF = host.find_tool("lsof", "/usr/sbin/lsof")`.
- `llm/service.py` uses `config.LLAMA_SERVER`, `config.PS`, and
  `config.LSOF` instead of hardcoded paths.
- `llm/diagnose.py` uses `config.LLAMA_SERVER` and the OS-aware upgrade hint.
  On Arch, `llama_cpp_package_status()` resolves the owning pacman package with
  `pacman -Qo` when possible and reports its version. On macOS it keeps the
  Homebrew status.
- `llm/cli.py` prints `Homebrew:` on macOS and `Package:` on Arch. `doctor`
  also warns when `lsof` is missing, because port-ownership checks are weaker
  without it.
- `llm/hardware.py` becomes OS-aware:
  - macOS keeps `sysctl` / `system_profiler`.
  - Arch reads the kernel interfaces directly instead of parsing
    human-readable CLI output:
    - CPU: `/proc/cpuinfo` `model name`.
    - RAM: `/proc/meminfo` `MemTotal` and `MemAvailable`, converted from KiB
      to bytes. Do not parse `free`.
    - Disk: `shutil.disk_usage` / `os.statvfs` (POSIX), not `df -h`. If the
      root filesystem is Btrfs and `btrfs` is available without root, try
      `btrfs filesystem usage -b /` for the estimated free figure, then fall
      back to statvfs.
    - GPU:
      - NVIDIA: `nvidia-smi --query-gpu=name,memory.total,memory.free
        --format=csv,noheader,nounits`.
      - AMD: `/sys/class/drm/card*/device/mem_info_vram_total` (and
        `mem_info_vram_used`), choosing the card with the largest VRAM.
      - Intel/integrated: `lspci` for the model name; no fixed VRAM, shared
        system RAM.
      - `lspci` is optional and only used for names. Never require root.
  - `Hardware` gains `available_ram`; preflight uses `MemAvailable` when it is
    known and falls back to total RAM otherwise.
  - `unified=False` on Arch.
  - Missing detection tools fall back to `unknown` rather than failing.
- The built-in `graphify` model profiles remain macOS-oriented. On Arch they
  may show as missing; users select or download their own models with
  `llm get` / `llm use`. No profile rewrite is part of this change.

### `llm/update.py` (new)

- `REPO_URL = "https://github.com/gluonstream/llm.git"`.
- `temp_root()` returns `/private/tmp` on `darwin`, otherwise `/tmp`.
- `default_checkout()` honors `LMM_UPDATE_DIR`, else
  `temp_root() / "llm-update"`.
- `update(checkout=None, runner=subprocess.run)`:
  - if the checkout exists, require `<checkout>/.git` and run
    `git -C <checkout> pull --ff-only`;
  - otherwise create the parent and run
    `git clone https://github.com/gluonstream/llm.git <checkout>`;
  - run `./build.sh` with `cwd=<checkout>`;
  - run `./install.sh` with `cwd=<checkout>`;
  - wrap command failures in `UpdateError` with the failing command.
- Refuse a non-git path with a message telling the user to remove it.
- Check for `git` first and report a clear error if it is missing.

### CLI and docs

- `llm/cli.py`: add `update` with optional `--dir PATH`; print the checkout
  path and build/install progress.
- README: add Arch requirements and install commands, the Updating section,
  the command table row, and environment variables `LMM_UPDATE_DIR`,
  `LMM_SKIP_LLAMA_CPP_UPGRADE`, and `LMM_LLAMA_SERVER`.
- README notes that Arch GPU reporting is best-effort: `lspci` is used for the
  model and `nvidia-smi` for NVIDIA VRAM.

### Tests

- New `tests/test_update.py`: clone, pull, non-git checkout, missing git,
  build failure, install ordering, and `LMM_UPDATE_DIR`/`--dir`.
- New `tests/test_host.py`: env/path/fallback for `llama-server`, tool
  discovery, upgrade hint.
- Extend `tests/test_hardware.py`: explicit macOS path plus Linux CPU,
  `/proc/meminfo` total/available, AMD sysfs VRAM, NVIDIA `nvidia-smi` parsing,
  Intel/shared fallback, and disk fallback. Tests use temporary sysfs-like
  paths and injected runners, never the host's real hardware.
- Extend `tests/test_diagnose.py`: macOS and Arch hints/package status.
- Extend build/install tests: set `LMM_SKIP_LLAMA_CPP_UPGRADE=1`; reject
  unsupported `OSTYPE` in `build.sh`, `install.sh`, and `update.sh`.
- Existing tests must pass on macOS and remain platform-injectable for Linux.

## Decided details

- Repository: `https://github.com/gluonstream/llm.git`.
- Checkout name: `llm-update`.
- Pull strategy: `git pull --ff-only`; if it fails, the user removes the temp
  checkout or sets `LMM_UPDATE_DIR`.
- Homebrew command: `brew upgrade llama.cpp`. `brew update` does not accept a
  formula, so the valid upgrade command is used.
- Arch package policy: no automatic pacman/AUR call. The official repos do not
  contain `llama.cpp`; AUR helpers can install it quickly, for example
  `yay -S llama.cpp-cuda-git` for CUDA or `yay -S llama.cpp-git` for a CPU
  build. The build script prints the guidance instead of running a system
  package manager.
- Arch dependency assumptions: `python3`, `bash`, `git`, `lsof`, `pciutils`
  (`lspci`, optional for GPU names), and a `llama-server` installation. `ps`
  comes from the base system. `nvidia-smi` is optional. `btrfs-progs` is
  optional for the Btrfs free-space estimate.
- Install destination: existing `install.sh` behavior (`LMM_BIN_DIR` or
  `~/.local/bin`).
- The current `llm` process keeps running the old code; the next invocation
  uses the installed update.
- `update.sh` works from a source checkout, not from an arbitrary download of
  the script alone.

## Left to the builder

- Exact output and error wording.
- How to present hybrid GPU setups in `doctor`; detection picks the largest
  VRAM device, but the display can mention the secondary device later.
- Test fake-runner shape and temporary-directory cleanup.
