# Self-update and Arch Linux Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `llm update` / `./update.sh` and make the CLI, diagnostics, hardware detection, and shell scripts work on macOS and Arch Linux.

**Architecture:** Add a small `llm/host.py` OS/tool-discovery layer, make existing runtime paths use it, add Linux kernel-interface probes to `hardware.py`, add `llm/update.py` for clone/pull/build/install, and give all three shell scripts the same `$OSTYPE` detector.

**Tech Stack:** Python 3.9+ standard library, Bash, `unittest`, `git`, `pacman`/AUR metadata, Linux `/proc` and sysfs.

**Spec:** `docs/superpowers/specs/2026-11-27-self-update-design.md`

## Global Constraints

- Supported OSes: macOS (`darwin*`) and Arch Linux (`linux-gnu*`). Any other `OSTYPE` exits 1 with `unsupported OS: ... (supported: macOS, Arch Linux)`.
- Linux is treated as Arch Linux. Do not add other-distro branches.
- Repository URL is exactly `https://github.com/gluonstream/llm.git`.
- Temp checkout is `/private/tmp/llm-update` on macOS and `/tmp/llm-update` on Arch, overridable with `LMM_UPDATE_DIR` or `llm update --dir`.
- Pull only with `git pull --ff-only`. Never install after a failed build.
- Do not run pacman or AUR helpers automatically. Arch build guidance is printed, not executed.
- Use `brew upgrade llama.cpp` on macOS; `brew update llama.cpp` is not valid.
- `LMM_SKIP_LLAMA_CPP_UPGRADE=1` skips the llama.cpp package-check step.
- No root-required probes. Missing `/proc`, sysfs, `nvidia-smi`, `lspci`, `lsof`, or `btrfs` must fall back cleanly.
- Keep Python 3.9 compatibility and standard-library-only dependencies.
- Preserve existing macOS behavior and all existing commands.
- `.DS_Store` stays ignored and untracked.

## Review Focus

- An unsupported `OSTYPE` must exit before clone, build, install, or package checks.
- A missing `git`, failed clone, failed pull, or failed build must leave the installed `llm` untouched.
- `LMM_LLAMA_SERVER` must win over `PATH`; `PATH` must win over the OS fallback.
- Linux probes must not crash when `/proc`, sysfs, `nvidia-smi`, `lspci`, or `btrfs` are absent.
- Build/install tests must never run `brew`, `pacman`, `git clone`, or network access.

---

### Task 1: Add the host/tool discovery layer

**Files:**
- Create: `llm/host.py`
- Create: `tests/test_host.py`

**Interfaces:**
- Produces:
  - `IS_MACOS: bool`
  - `IS_LINUX: bool`
  - `find_tool(name: str, fallback: str, which=None) -> str`
  - `find_llama_server(env=None, which=None) -> str`
  - `llama_cpp_upgrade_hint() -> str`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
from unittest import mock

from llm import host


class HostTest(unittest.TestCase):
    def test_find_tool_prefers_which(self):
        self.assertEqual(
            host.find_tool("lsof", "/fallback", which=lambda name: "/usr/bin/lsof"),
            "/usr/bin/lsof")

    def test_find_tool_uses_fallback(self):
        self.assertEqual(
            host.find_tool("lsof", "/fallback", which=lambda name: None),
            "/fallback")

    def test_llama_server_env_wins(self):
        self.assertEqual(
            host.find_llama_server(
                env={"LMM_LLAMA_SERVER": "/custom/llama-server"},
                which=lambda name: "/usr/bin/llama-server"),
            "/custom/llama-server")

    def test_llama_server_path_wins(self):
        self.assertEqual(
            host.find_llama_server(env={}, which=lambda name: "/usr/bin/llama-server"),
            "/usr/bin/llama-server")

    def test_llama_server_macos_and_arch_fallbacks(self):
        with mock.patch.object(host, "IS_MACOS", True):
            self.assertEqual(
                host.find_llama_server(env={}, which=lambda name: None),
                "/opt/homebrew/bin/llama-server")
        with mock.patch.object(host, "IS_MACOS", False):
            self.assertEqual(
                host.find_llama_server(env={}, which=lambda name: None),
                "/usr/bin/llama-server")

    def test_upgrade_hint_is_os_specific(self):
        with mock.patch.object(host, "IS_MACOS", True):
            self.assertIn("brew upgrade llama.cpp", host.llama_cpp_upgrade_hint())
        with mock.patch.object(host, "IS_MACOS", False):
            self.assertIn("Arch", host.llama_cpp_upgrade_hint())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_host -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'llm.host'`.

- [ ] **Step 3: Implement `llm/host.py`**

```python
"""Operating-system and external-tool discovery for llm."""
import os
import shutil
import sys

IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


def find_tool(name, fallback, which=None):
    which = shutil.which if which is None else which
    return which(name) or fallback


def find_llama_server(env=None, which=None):
    env = os.environ if env is None else env
    configured = env.get("LMM_LLAMA_SERVER")
    if configured:
        return configured
    fallback = ("/opt/homebrew/bin/llama-server" if IS_MACOS
                else "/usr/bin/llama-server")
    return find_tool("llama-server", fallback, which=which)


def llama_cpp_upgrade_hint():
    if IS_MACOS:
        return "brew upgrade llama.cpp"
    return "update llama.cpp (Arch/AUR package or source build)"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_host -v`

Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add llm/host.py tests/test_host.py
git commit -m "feat: add OS and tool discovery layer"
```

### Task 2: Route existing runtime paths through the host layer

**Files:**
- Modify: `llm/config.py`
- Modify: `llm/service.py`
- Modify: `llm/diagnose.py`
- Modify: `llm/cli.py`
- Modify: `tests/test_service.py`
- Modify: `tests/test_diagnose.py`

**Interfaces:**
- Consumes: `host.find_llama_server`, `host.find_tool`.
- Produces:
  - `config.LLAMA_SERVER`
  - `config.PS`
  - `config.LSOF`
  - `diagnose.llama_server_version(llama_server=None)`
  - `diagnose.explain_failure(log_text, upgrade_hint=None)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_service.py`:

```python
    def test_server_args_uses_config_llama_server(self):
        profile = {"id": "x", "alias": "x", "path": "/tmp/x.gguf", "context": 1,
                   "sampling": {"temperature": 0, "top_p": 1, "min_p": 0,
                                "repeat_penalty": 1}}
        with mock.patch.object(service.config, "LLAMA_SERVER", "/custom/llama-server"):
            self.assertEqual(service.server_args(profile)[0], "/custom/llama-server")

    def test_identity_uses_config_ps(self):
        with mock.patch.object(service.config, "PS", "/custom/ps"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = "identity"
            service.identity(123)
        self.assertEqual(run.call_args[0][0][0], "/custom/ps")

    def test_listener_uses_config_lsof(self):
        with mock.patch.object(service.config, "LSOF", "/custom/lsof"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = "42\n"
            self.assertEqual(service.listener_pid(), 42)
        self.assertEqual(run.call_args[0][0][0], "/custom/lsof")
```

Add `from unittest import mock` to `tests/test_diagnose.py`, update the
existing `test_unknown_architecture_inkling` to call
`diagnose.explain_failure(log, upgrade_hint="brew upgrade llama.cpp")`, and add:

```python
    def test_explain_failure_accepts_an_upgrade_hint(self):
        log = "error loading model: unknown model architecture: 'inkling2_loop'\n"
        self.assertIn("custom hint", diagnose.explain_failure(log, upgrade_hint="custom hint"))

    def test_llama_server_version_uses_config_path(self):
        with mock.patch.object(diagnose.config, "LLAMA_SERVER", "/custom/llama-server"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = "version: 1.1.0"
            diagnose.llama_server_version()
        self.assertEqual(run.call_args[0][0][0], "/custom/llama-server")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_service tests.test_diagnose -v`

Expected: FAIL because `config.LLAMA_SERVER`, `config.PS`, `config.LSOF`, and the hint parameter do not exist.

- [ ] **Step 3: Implement the wiring**

In `llm/config.py`, import `host` and add after `PORT`:

```python
LLAMA_SERVER = host.find_llama_server()
PS = host.find_tool("ps", "/bin/ps")
LSOF = host.find_tool("lsof", "/usr/sbin/lsof")
```

In `llm/service.py`:

- remove the module-level hardcoded `LLAMA_SERVER`;
- replace `["/bin/ps", ...]` with `[config.PS, ...]`;
- replace `["/usr/sbin/lsof", ...]` with `[config.LSOF, ...]`;
- change `def server_args(profile, llama_server=LLAMA_SERVER)` to
  `def server_args(profile, llama_server=None)` and start with
  `args = [llama_server or config.LLAMA_SERVER]`.

In `llm/diagnose.py`:

- import `config` and `host`;
- remove the hardcoded `LLAMA_SERVER`;
- `def llama_server_version(llama_server=None):` uses
  `llama_server or config.LLAMA_SERVER`;
- `def explain_failure(log_text, upgrade_hint=None):` uses
  `hint = upgrade_hint or host.llama_cpp_upgrade_hint()` and substitutes
  `hint` for both `brew upgrade llama.cpp` strings.

In `llm/cli.py`, change `cmd_doctor` to use `config.LLAMA_SERVER` instead of
`service.LLAMA_SERVER`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_service tests.test_diagnose -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm/config.py llm/service.py llm/diagnose.py llm/cli.py \
  tests/test_service.py tests/test_diagnose.py
git commit -m "refactor: use OS-discovered tool paths"
```

### Task 3: Add Linux hardware detection

**Files:**
- Modify: `llm/hardware.py`
- Modify: `llm/cli.py`
- Modify: `tests/test_hardware.py`

**Interfaces:**
- Produces:
  - `Hardware.available_ram: int`
  - `detect_hardware(..., platform_name=None, linux_cpu=None, linux_ram=None, linux_gpu=None)`
  - `_parse_meminfo(text) -> dict`
  - `_linux_gpu(sysfs_root=None, runner=None) -> (str|None, int|None)`
  - `_disk_free(path, runner=None) -> int|None`

- [ ] **Step 1: Write the failing tests**

Update the two existing `detect_hardware` tests to pass
`platform_name="darwin"` so they exercise the macOS path on any host. Then add
to `tests/test_hardware.py`:

```python
    def test_memory_budget_uses_available_ram(self):
        self.assertEqual(
            hardware.memory_budget(36 * hardware.GIB, unified=True,
                                   available=24 * hardware.GIB),
            int(24 * hardware.GIB * 0.8))

    def test_parse_meminfo(self):
        values = hardware._parse_meminfo(
            "MemTotal:       1000 kB\nMemAvailable:    500 kB\n")
        self.assertEqual(values["MemTotal"], 1000 * 1024)
        self.assertEqual(values["MemAvailable"], 500 * 1024)

    def test_linux_gpu_prefers_nvidia(self):
        def runner(command, timeout=5):
            if command[0] == "nvidia-smi":
                return "NVIDIA RTX 4090, 24564, 24000\n"
            return ""
        self.assertEqual(
            hardware._linux_gpu(runner=runner),
            ("NVIDIA RTX 4090", 24564 * 1024 * 1024))

    def test_linux_gpu_reads_amd_sysfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = Path(tmp) / "card0" / "device"
            device.mkdir(parents=True)
            (device / "mem_info_vram_total").write_text("8589934592")
            def runner(command, timeout=5):
                return ""
            self.assertEqual(
                hardware._linux_gpu(sysfs_root=Path(tmp), runner=runner),
                ("AMD GPU", 8589934592))

    def test_linux_gpu_falls_back_to_lspci(self):
        def runner(command, timeout=5):
            if command[0] == "lspci":
                return "00:02.0 VGA compatible controller: Intel Corporation Xe\n"
            return ""
        name, vram = hardware._linux_gpu(runner=runner)
        self.assertIn("Intel", name)
        self.assertIsNone(vram)

    def test_linux_gpu_returns_none_when_all_probes_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                hardware._linux_gpu(sysfs_root=Path(tmp),
                                    runner=lambda command, timeout=5: ""),
                (None, None))

    def test_detect_hardware_linux_uses_injected_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            machine = hardware.detect_hardware(
                platform_name="linux",
                linux_cpu=lambda: "AMD Ryzen 9",
                linux_ram=lambda: (32 * hardware.GIB, 24 * hardware.GIB),
                linux_gpu=lambda: ("NVIDIA RTX 4090", 24 * hardware.GIB),
                models_dir=Path(tmp))
        self.assertEqual(machine.total_ram, 32 * hardware.GIB)
        self.assertEqual(machine.available_ram, 24 * hardware.GIB)
        self.assertEqual(machine.chip, "AMD Ryzen 9")
        self.assertFalse(machine.unified)

    def test_detect_hardware_macos_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            machine = hardware.detect_hardware(
                platform_name="darwin",
                sysctl=lambda name: {"hw.memsize": str(36 * hardware.GIB),
                                     "hw.model": "Mac15,8",
                                     "hw.optional.arm64": "1"}.get(name, ""),
                displays=lambda: ("Apple M3 Max", None),
                models_dir=Path(tmp))
        self.assertTrue(machine.unified)
        self.assertEqual(machine.available_ram, 0)

    def test_detect_hardware_linux_with_empty_probes_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            machine = hardware.detect_hardware(
                platform_name="linux",
                linux_cpu=lambda: "",
                linux_ram=lambda: (0, 0),
                linux_gpu=lambda: (None, None),
                models_dir=Path(tmp))
        self.assertTrue(machine.chip)
        self.assertEqual(machine.total_ram, 0)
        self.assertFalse(machine.unified)

    def test_disk_free_uses_btrfs_when_available(self):
        with mock.patch.object(hardware.host, "IS_LINUX", True), \
             mock.patch.object(hardware, "_filesystem_type", return_value="btrfs"), \
             mock.patch.object(hardware, "_btrfs_free", return_value=123), \
             mock.patch.object(hardware.shutil, "disk_usage") as usage:
            usage.return_value.free = 999
            self.assertEqual(hardware._disk_free("/"), 123)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_hardware -v`

Expected: FAIL for the missing `available_ram`, `_parse_meminfo`, `_linux_gpu`, `_disk_free`, and platform parameters.

- [ ] **Step 3: Implement the Linux probes**

In `llm/hardware.py`:

- add `from . import host`;
- add `available_ram: int = 0` at the end of `Hardware`;
- add `_read_text(path)` returning `""` on `OSError`;
- add `_parse_meminfo(text)` parsing `key: value kB` into bytes;
- add `_linux_ram()` returning `(MemTotal, MemAvailable)` from `/proc/meminfo`;
- add `_linux_cpu()` reading `model name` from `/proc/cpuinfo`;
- add `_linux_gpu(sysfs_root=None, runner=None)`:
  - try `nvidia-smi --query-gpu=name,memory.total,memory.free
    --format=csv,noheader,nounits`; parse name and MiB total;
  - else scan `Path(sysfs_root or "/sys/class/drm").glob(
    "card*/device/mem_info_vram_total")`, take the largest integer, return
    `("AMD GPU", bytes)`;
  - else run `lspci`, match `VGA compatible controller`, `3D controller`, or
    `display controller`, and return the text after the first `": "` with
    `None` VRAM;
  - return `(None, None)` if nothing is found.
- add `_filesystem_type(path, runner=None)` using `stat -f -c %T` and
  `_btrfs_free(path, runner=None)` parsing `Free (estimated):` from
  `btrfs filesystem usage -b`; both return `None` on failure.
- add `_disk_free(path, runner=None)`:
  - use `shutil.disk_usage(_existing_parent(path)).free`;
  - when `host.IS_LINUX` and the filesystem is `btrfs`, prefer
    `_btrfs_free` when it returns a value;
  - otherwise return the statvfs value.
- change `memory_budget(total_ram, unified=True, available=None)` to use
  `available` when it is positive, else `total_ram`.
- extend `detect_hardware` with `platform_name=None`, `linux_cpu=None`,
  `linux_ram=None`, `linux_gpu=None`:
  - macOS path keeps the existing sysctl/display code and `available_ram=0`;
  - Linux path uses the injected or real CPU/RAM/GPU probes, sets
    `unified=False`, and `model=platform.machine() or "unknown"`.
- change `preflight` to call
  `memory_budget(machine.total_ram, machine.unified, machine.available_ram)`.
- change `cli.cmd_doctor` to append `", X available"` when
  `machine.available_ram` is positive.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_hardware -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm/hardware.py llm/cli.py tests/test_hardware.py
git commit -m "feat: add Linux hardware detection"
```

### Task 4: Make build/install/update scripts OS-aware

**Files:**
- Modify: `build.sh`
- Modify: `install.sh`
- Create: `update.sh`
- Modify: `tests/test_build_install.py`

**Interfaces:**
- Produces:
  - `build.sh` accepts macOS and `linux-gnu*`; rejects others.
  - `install.sh` accepts macOS and `linux-gnu*`; rejects others.
  - `update.sh` accepts macOS and `linux-gnu*`; rejects others.

- [ ] **Step 1: Write the failing tests**

In `tests/test_build_install.py`, add `"LMM_SKIP_LLAMA_CPP_UPGRADE": "1"` to
`env_for`, then add:

```python
    def test_build_rejects_unsupported_os(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(BUILD)], cwd=ROOT,
                env=env_for(tmp, OSTYPE="plan9"),
                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported OS", result.stderr)

    def test_install_rejects_unsupported_os(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(ROOT / "install.sh")], cwd=ROOT,
                env=env_for(tmp, OSTYPE="plan9"),
                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported OS", result.stderr)

    def test_update_script_is_executable_and_rejects_unsupported_os(self):
        script = ROOT / "update.sh"
        self.assertTrue(os.access(script, os.X_OK))
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(script)], cwd=ROOT,
                env=env_for(tmp, OSTYPE="plan9"),
                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported OS", result.stderr)

    def test_build_skips_package_check_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            bindir = Path(tmp) / "bin"
            bindir.mkdir()
            marker = Path(tmp) / "brew-called"
            brew = bindir / "brew"
            brew.write_text("#!/bin/sh\ntouch %s\n" % marker)
            brew.chmod(0o755)
            env = env_for(tmp)
            env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(["bash", str(BUILD)], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_build_install -v`

Expected: FAIL because `update.sh` is missing and the scripts do not reject unsupported OSes.

- [ ] **Step 3: Add the shared OS detector**

Add this block immediately after `set -euo pipefail` in `build.sh` and
`install.sh`, and in the new `update.sh`:

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

- [ ] **Step 4: Add the macOS/Arch package check to `build.sh`**

After the `python3` check and before the tests, add:

```bash
if [ "${LMM_SKIP_LLAMA_CPP_UPGRADE:-0}" != "1" ]; then
    if [ "$OS" = "macos" ] && command -v brew >/dev/null 2>&1; then
        if brew list --versions llama.cpp >/dev/null 2>&1; then
            echo "Updating Homebrew llama.cpp..."
            if ! brew upgrade llama.cpp; then
                echo "warning: brew upgrade llama.cpp failed; continuing." >&2
            fi
        else
            echo "Homebrew llama.cpp is not installed; skipping upgrade." >&2
        fi
    elif [ "$OS" = "arch" ]; then
        if command -v llama-server >/dev/null 2>&1; then
            echo "llama-server: $(llama-server --version 2>&1 | head -n 1)"
        else
            echo "llama-server not found; install llama.cpp (AUR, e.g. yay -S llama.cpp-cuda-git for CUDA or yay -S llama.cpp-git for CPU) or set LMM_LLAMA_SERVER." >&2
        fi
    fi
fi
```

- [ ] **Step 5: Add the Arch warning to `install.sh`**

After the successful install and before the PATH warning, add:

```bash
if [ "$OS" = "arch" ] && ! command -v llama-server >/dev/null 2>&1; then
    echo "warning: llama-server not found; install llama.cpp (AUR) or set LMM_LLAMA_SERVER." >&2
fi
```

- [ ] **Step 6: Create `update.sh`**

```bash
#!/usr/bin/env bash
# Fetch, build, and install the latest llm from source.
set -euo pipefail

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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$OS" = "macos" ]; then
    default_dir="/private/tmp/llm-update"
else
    default_dir="/tmp/llm-update"
fi
export LMM_UPDATE_DIR="${LMM_UPDATE_DIR:-$default_dir}"
cd "$ROOT"
exec python3 -m llm.update "$@"
```

Then `chmod +x update.sh`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_build_install -v`

Expected: PASS. The build/install tests must not call brew or pacman because
`LMM_SKIP_LLAMA_CPP_UPGRADE=1` is in `env_for`.

- [ ] **Step 8: Commit**

```bash
git add build.sh install.sh update.sh tests/test_build_install.py
git commit -m "feat: support macOS and Arch in shell scripts"
```

### Task 5: Add the update module and CLI command

**Files:**
- Create: `llm/update.py`
- Modify: `llm/cli.py`
- Create: `tests/test_update.py`

**Interfaces:**
- Produces:
  - `update.REPO_URL`
  - `update.UpdateError`
  - `update.temp_root(platform_name=None) -> Path`
  - `update.default_checkout(root=None) -> Path`
  - `update.update(checkout=None, repo_url=REPO_URL, runner=subprocess.run) -> Path`
  - `update.main(argv=None) -> int`
  - `llm update [--dir PATH]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_update.py`:

```python
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm import update


class FakeRunner:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, command, cwd=None, check=False):
        self.calls.append((list(command), cwd))
        if self.fail_on is not None and list(command) == list(self.fail_on):
            raise subprocess.CalledProcessError(1, command)


class UpdateTest(unittest.TestCase):
    def test_temp_root_is_os_specific(self):
        self.assertEqual(update.temp_root("darwin"), Path("/private/tmp"))
        self.assertEqual(update.temp_root("linux"), Path("/tmp"))

    def test_default_checkout_honors_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "custom"
            with mock.patch.dict(os.environ, {"LMM_UPDATE_DIR": str(target)}):
                self.assertEqual(update.default_checkout(), target)

    def test_update_clones_then_builds_and_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            runner = FakeRunner()
            result = update.update(checkout=checkout, runner=runner)
        self.assertEqual(result, checkout)
        self.assertEqual(runner.calls, [
            (["git", "clone", update.REPO_URL, str(checkout)], None),
            (["./build.sh"], str(checkout)),
            (["./install.sh"], str(checkout)),
        ])

    def test_update_pulls_existing_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            (checkout / ".git").mkdir(parents=True)
            runner = FakeRunner()
            update.update(checkout=checkout, runner=runner)
        self.assertEqual(runner.calls[0],
                         (["git", "-C", str(checkout), "pull", "--ff-only"], None))

    def test_update_refuses_non_git_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            checkout.mkdir()
            runner = FakeRunner()
            with self.assertRaises(update.UpdateError):
                update.update(checkout=checkout, runner=runner)
        self.assertEqual(runner.calls, [])

    def test_update_reports_failed_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            runner = FakeRunner(fail_on=["git", "clone", update.REPO_URL, str(checkout)])
            with self.assertRaises(update.UpdateError) as caught:
                update.update(checkout=checkout, runner=runner)
        self.assertIn("git clone", str(caught.exception))

    def test_update_reports_failed_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            (checkout / ".git").mkdir(parents=True)
            runner = FakeRunner(
                fail_on=["git", "-C", str(checkout), "pull", "--ff-only"])
            with self.assertRaises(update.UpdateError) as caught:
                update.update(checkout=checkout, runner=runner)
        self.assertIn("pull --ff-only", str(caught.exception))

    def test_update_reports_failed_build_and_does_not_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            runner = FakeRunner(fail_on=["./build.sh"])
            with self.assertRaises(update.UpdateError) as caught:
                update.update(checkout=checkout, runner=runner)
        self.assertIn("./build.sh", str(caught.exception))
        self.assertNotIn(("./install.sh"), [call[0][0] for call in runner.calls])

    def test_update_reports_missing_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(update.shutil, "which", return_value=None):
                with self.assertRaises(update.UpdateError) as caught:
                    update.update(checkout=Path(tmp) / "llm-update",
                                  runner=FakeRunner())
        self.assertIn("git", str(caught.exception))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_update -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'llm.update'`.

- [ ] **Step 3: Implement `llm/update.py`**

```python
"""Fetch, build, and install the latest llm from GitHub."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/gluonstream/llm.git"
CHECKOUT_NAME = "llm-update"


class UpdateError(RuntimeError):
    pass


def temp_root(platform_name=None):
    name = sys.platform if platform_name is None else platform_name
    return Path("/private/tmp" if name == "darwin" else "/tmp")


def default_checkout(root=None):
    configured = os.environ.get("LMM_UPDATE_DIR")
    if configured:
        return Path(configured).expanduser()
    return (Path(root) if root is not None else temp_root()) / CHECKOUT_NAME


def _run(command, cwd=None, runner=subprocess.run):
    try:
        runner(command, cwd=str(cwd) if cwd is not None else None, check=True)
    except subprocess.CalledProcessError as exc:
        raise UpdateError("command failed: %s" % " ".join(command)) from exc


def update(checkout=None, repo_url=REPO_URL, runner=subprocess.run):
    if shutil.which("git") is None:
        raise UpdateError("git not found; install git and retry")
    checkout = Path(checkout) if checkout is not None else default_checkout()
    if checkout.exists():
        if not (checkout / ".git").is_dir():
            raise UpdateError("%s is not a git checkout; remove it and retry" % checkout)
        _run(["git", "-C", str(checkout), "pull", "--ff-only"], runner=runner)
    else:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo_url, str(checkout)], runner=runner)
    _run(["./build.sh"], cwd=checkout, runner=runner)
    _run(["./install.sh"], cwd=checkout, runner=runner)
    return checkout


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llm update",
        description="Fetch the latest source, build it, and install it.")
    parser.add_argument("--dir", help="checkout directory")
    args = parser.parse_args(argv)
    try:
        checkout = update(checkout=args.dir)
    except UpdateError as exc:
        print("llm update: %s" % exc, file=sys.stderr)
        return 1
    print("Updated llm from %s" % checkout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add the CLI command**

In `llm/cli.py`, import `update` and add:

```python
def cmd_update(args):
    checkout = Path(args.dir).expanduser() if args.dir else None
    print("Updating llm from %s" % update.REPO_URL)
    path = update.update(checkout=checkout)
    print("Updated llm from %s" % path)
    return 0
```

Register it before `doctor` or `help`:

```python
    command = add("update", cmd_update,
                  "fetch the latest source, build, and install")
    command.add_argument("--dir", help="checkout directory")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_update -v`

Expected: PASS, 9 tests.

- [ ] **Step 6: Smoke-test the CLI help without network**

Run: `./bin/llm update --help`

Expected: exit 0, shows `--dir`.

- [ ] **Step 7: Commit**

```bash
git add llm/update.py llm/cli.py tests/test_update.py
git commit -m "feat: add llm update"
```

### Task 6: Add Arch diagnostics and doctor output

**Files:**
- Modify: `llm/diagnose.py`
- Modify: `llm/cli.py`
- Modify: `tests/test_diagnose.py`

**Interfaces:**
- Produces:
  - `diagnose.llama_cpp_package_status() -> str|None`
  - `doctor` prints `Homebrew:` on macOS and `Package:` on Arch.
  - `doctor` warns when `lsof` is unavailable.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_diagnose.py`:

```python
    def test_arch_package_status_uses_pacman_owner(self):
        with mock.patch.object(diagnose.host, "IS_MACOS", False), \
             mock.patch.object(diagnose.host, "IS_LINUX", True), \
             mock.patch.object(diagnose.config, "LLAMA_SERVER", "/usr/bin/llama-server"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = (
                "/usr/bin/llama-server is owned by llama.cpp-git r1-1\n")
            self.assertIn("llama.cpp-git", diagnose.llama_cpp_package_status())

    def test_macos_package_status_uses_brew(self):
        with mock.patch.object(diagnose.host, "IS_MACOS", True), \
             mock.patch.object(diagnose, "brew_llama_cpp_status",
                               return_value="Homebrew llama.cpp 1.1.0"):
            self.assertEqual(diagnose.llama_cpp_package_status(),
                             "Homebrew llama.cpp 1.1.0")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_diagnose -v`

Expected: FAIL because `llama_cpp_package_status` does not exist.

- [ ] **Step 3: Implement package status**

In `llm/diagnose.py`, add:

```python
def llama_cpp_package_status():
    if host.IS_MACOS:
        return brew_llama_cpp_status()
    if not host.IS_LINUX:
        return None
    path = config.LLAMA_SERVER
    try:
        completed = subprocess.run(
            ["pacman", "-Qo", path], capture_output=True, text=True,
            check=False, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (completed.stdout or "") + (completed.stderr or "")
    match = re.search(r"is owned by (\S+) (\S+)", text)
    if match:
        return "Arch package %s %s" % (match.group(1), match.group(2))
    return None
```

- [ ] **Step 4: Update `doctor`**

In `llm/cli.py`:

- replace the `brew = diagnose.brew_llama_cpp_status()` block with
  `package = diagnose.llama_cpp_package_status()` and print
  `"Homebrew:     "` when `host.IS_MACOS` else `"Package:      "`;
- add `from . import host`;
- after the hardware line, if `not os.access(config.LSOF, os.X_OK)`, print
  `"warning: lsof not found; port ownership checks are limited"`;
- include `machine.available_ram` in the hardware line when it is positive.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_diagnose -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add llm/diagnose.py llm/cli.py tests/test_diagnose.py
git commit -m "feat: add Arch package diagnostics"
```

### Task 7: Document Arch support and self-update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Requirements**

Change the platform requirement to macOS on Apple Silicon or Arch Linux.
Mention:

- Python 3.9+, Bash, and `git` for `llm update`;
- `llama-server` from Homebrew on macOS, or from an AUR package/source build
  on Arch;
- `lsof` on Arch;
- optional `pciutils` (`lspci`), `nvidia-smi`, and `btrfs-progs` for richer
  hardware reporting;
- `LMM_LLAMA_SERVER` if `llama-server` is not on `PATH`.

Include Arch install examples:

```sh
# NVIDIA/CUDA:
yay -S llama.cpp-cuda-git
# or CPU/other build:
# yay -S llama.cpp-git
sudo pacman -S lsof pciutils
```

- [ ] **Step 2: Add the Updating section**

After Installation, add:

```markdown
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
```

- [ ] **Step 3: Add the command and environment entries**

Add `llm update [--dir PATH]` to the commands table.

Add to the environment table:

```markdown
| `LMM_UPDATE_DIR` | unset | Override the self-update checkout directory. |
| `LMM_SKIP_LLAMA_CPP_UPGRADE` | unset | Set to `1` to skip the build-time llama.cpp package check. |
| `LMM_LLAMA_SERVER` | detected | Path to `llama-server`; overrides `PATH` detection. |
```

- [ ] **Step 4: Update architecture and platform notes**

Add `llm/host.py`, `llm/update.py`, and `update.sh` to the architecture tree.
Add a short note that Arch hardware reporting is best-effort: CPU/RAM are
reliable, AMD/NVIDIA VRAM is read from sysfs/`nvidia-smi`, and Intel integrated
GPUs report shared RAM.

- [ ] **Step 5: Verify docs**

```sh
grep -n '## Updating' README.md
grep -n 'LMM_UPDATE_DIR' README.md
grep -n 'LMM_LLAMA_SERVER' README.md
grep -n 'Arch' README.md
git diff --check
```

Expected: all commands succeed.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document Arch support and self-update"
```

### Task 8: Final verification

**Files:**
- Verify: all changed files
- Add: `docs/superpowers/specs/2026-11-27-self-update-design.md`
- Add: `docs/superpowers/plans/2026-11-27-self-update.md`

- [ ] **Step 1: Run the full test suite**

```sh
python3 -m unittest discover -s tests -q
```

Expected: all tests pass.

- [ ] **Step 2: Build and smoke-test without touching Homebrew**

```sh
LMM_SKIP_LLAMA_CPP_UPGRADE=1 ./build.sh
./bin/llm help
./bin/llm update --help
```

Expected: build succeeds; both help commands exit 0.

- [ ] **Step 3: Verify OS rejection on all three scripts**

```sh
err="$(mktemp)"
trap 'rm -f "$err"' EXIT
for script in build.sh install.sh update.sh; do
  OSTYPE=plan9 bash "$script" >/dev/null 2>"$err" && exit 1
  grep -q 'unsupported OS' "$err"
done
```

Expected: each script exits non-zero with `unsupported OS`.

- [ ] **Step 4: Check whitespace and status**

```sh
git diff --check
git status --short
```

Expected: no whitespace errors; only the design/plan files are untracked.

- [ ] **Step 5: Commit the design and plan**

```bash
git add docs/superpowers/specs/2026-11-27-self-update-design.md \
  docs/superpowers/plans/2026-11-27-self-update.md
git commit -m "docs: add self-update and Arch support design"
```

- [ ] **Step 6: Confirm the final state**

```sh
git log --oneline -8
git status --short
```

Expected: all feature commits are present and the working tree is clean.

### Task 9: Optional Arch container verification

**Files:** none.

- [ ] **Step 1: If Docker or Podman is available, run the script checks in Arch**

```sh
docker run --rm -v "$PWD":/src -w /src archlinux:latest \
  bash -lc 'pacman -Sy --noconfirm python git lsof pciutils >/dev/null && \
  python3 -m unittest discover -s tests -q && \
  LMM_SKIP_LLAMA_CPP_UPGRADE=1 ./build.sh && ./bin/llm help'
```

Expected: tests pass and the build succeeds. If no container runtime is
available, record that this was not executed and rely on the injected Linux
unit tests.
