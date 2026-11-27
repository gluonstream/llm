# Build and Install Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `build.sh` that packages the CLI as a single executable zipapp and `install.sh` that installs it into `~/.local/bin/llm`.

**Architecture:** `build.sh` copies the package into `build/app`, compiles it to legacy `.pyc` files with `compileall -b`, adds a `__main__.py`, and emits `dist/llm` with `python -m zipapp`. `install.sh` atomically copies that file to `${LMM_BIN_DIR:-$HOME/.local/bin}/llm` and points `lmm` at it. Both scripts take directory overrides so tests never touch the real home directory.

**Tech Stack:** Bash, Python 3 stdlib (`compileall`, `zipapp`), `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-27-build-install-scripts-design.md`

## Global Constraints

- Scripts must run from any working directory; resolve the repo root from `$0`.
- Build output must never pollute the source tree: stage under `build/`, emit to `dist/`, both ignored by git.
- The installed command needs `python3` on `PATH`; no bundled interpreter.
- Re-running either script must be safe and idempotent.
- Directory overrides: `LMM_BUILD_DIR`, `LMM_DIST_DIR`, `LMM_BIN_DIR`.
- All paths must work when they contain spaces.

## Review Focus

- **Destination with spaces** — build, dist, and bin directories containing spaces must still build and install.
- **Existing file or symlink at the target** — `install.sh` must replace `~/.local/bin/llm` whether it is an old symlink or a regular file.
- **Missing artifact** — running `install.sh` before `build.sh` must fail with a message naming `./build.sh`.
- **Python version change** — the zip must contain `.py` sources next to `.pyc` so a newer interpreter can fall back instead of failing.
- **Repeated runs** — a second build or install must leave a working command and no stray temp files.

---

### Task 1: Build script producing a single-file zipapp

**Files:**
- Create: `build.sh`
- Create: `tests/test_build_install.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `llm/` package and its `cli.main()` entry point.
- Produces: executable `${LMM_DIST_DIR:-<root>/dist}/llm`; honors `LMM_BUILD_DIR` and `LMM_DIST_DIR`. Tests for Task 2 call `build.sh` and read the resulting `llm`.

- [ ] **Step 1: Write the failing build tests**

Create `tests/test_build_install.py` with a `BuildTest` class. Use a per-test temp root for the overrides, and put spaces in the names to pin the path handling:

```python
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build.sh"


def env_for(tmp, **extra):
    env = dict(os.environ)
    state = Path(tmp) / "state"
    state.mkdir(exist_ok=True)
    env.update({
        "LMM_STATE_DIR": str(state),
        "LMM_LEGACY_STATE_DIR": str(Path(tmp) / "legacy-missing"),
        "LMM_BUILD_DIR": str(Path(tmp) / "build dir"),
        "LMM_DIST_DIR": str(Path(tmp) / "dist dir"),
    })
    env.update(extra)
    return env


class BuildTest(unittest.TestCase):
    def test_build_produces_executable_zipapp_with_bytecode_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(["bash", str(BUILD)], cwd=ROOT,
                                    env=env_for(tmp), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = Path(tmp) / "dist dir" / "llm"
            self.assertTrue(artifact.is_file())
            self.assertTrue(os.access(artifact, os.X_OK))
            self.assertTrue(zipfile.is_zipfile(artifact))
            with zipfile.ZipFile(artifact) as archive:
                names = set(archive.namelist())
            for expected in ("__main__.py", "llm/cli.pyc",
                             "llm/cli.py"):
                self.assertIn(expected, names)

    def test_build_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = env_for(tmp)
            for _ in range(2):
                result = subprocess.run(["bash", str(BUILD)], cwd=ROOT,
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(tmp) / "dist dir" / "llm").is_file())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_build_install.BuildTest -v`
Expected: FAIL — `bash` cannot find `build.sh` (exit 127) or `FileNotFoundError` from `subprocess`.

- [ ] **Step 3: Implement `build.sh`**

Create `build.sh` with `#!/usr/bin/env bash` and `set -euo pipefail`. It must:

1. Resolve `ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`.
2. Default `BUILD_DIR="${LMM_BUILD_DIR:-$ROOT/build}"`, `DIST_DIR="${LMM_DIST_DIR:-$ROOT/dist}"`, `STAGE="$BUILD_DIR/app"`, `ARTIFACT="$DIST_DIR/llm"`.
3. Fail with `python3: command not found` on `PATH` if `python3` is absent.
4. `rm -rf "$STAGE"`, `mkdir -p "$STAGE" "$DIST_DIR"`, `rm -f "$ARTIFACT"`.
5. `cp -R "$ROOT/llm" "$STAGE/llm"` then remove staged `__pycache__` dirs with `find "$STAGE" -type d -name __pycache__ -prune -exec rm -rf {} +`.
6. Write `"$STAGE/__main__.py"` (heredoc, quoted delimiter) with exactly:

```python
import sys

from llm.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

7. `python3 -m compileall -b -q "$STAGE"`.
8. `python3 -m zipapp "$STAGE" -p "/usr/bin/env python3" -c -o "$ARTIFACT"`, then `chmod +x "$ARTIFACT"`.
9. Smoke-test with a throwaway state dir: `state="$(mktemp -d)"`, a `trap` that removes it on `EXIT`, then `LMM_STATE_DIR="$state" "$ARTIFACT" help >/dev/null`.
10. Print `Built $ARTIFACT` on success.

- [ ] **Step 4: Run the build tests to verify they pass**

Run: `python3 -m unittest tests.test_build_install.BuildTest -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Ignore build output and commit**

Add `/build/` and `/dist/` to `.gitignore`, then:

```bash
git add build.sh tests/test_build_install.py .gitignore
git commit -m "feat: add build.sh producing a single-file zipapp"
```

---

### Task 2: Install script for the zipapp

**Files:**
- Create: `install.sh`
- Modify: `tests/test_build_install.py` (add `InstallTest`)
- Modify: `README.md:86-92`

**Interfaces:**
- Consumes: the artifact produced by Task 1, `${LMM_DIST_DIR:-<root>/dist}/llm`.
- Produces: executable `${LMM_BIN_DIR:-$HOME/.local/bin}/llm`, plus a relative symlink `lmm` pointing at `llm`.

- [ ] **Step 1: Write the failing install tests**

Add an `InstallTest` class to `tests/test_build_install.py`:

```python
class InstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.env = env_for(self.tmp.name,
                           LMM_BIN_DIR=str(self.tmp_path / "bin dir"))
        self.bin_dir = self.tmp_path / "bin dir"
        built = subprocess.run(["bash", str(ROOT / "build.sh")], cwd=ROOT,
                               env=self.env, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)

    def install(self):
        return subprocess.run(["bash", str(ROOT / "install.sh")], cwd=ROOT,
                              env=self.env, capture_output=True, text=True)

    def test_installs_llm_and_relative_aliases(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        llm = self.bin_dir / "llm"
        self.assertTrue(llm.is_file())
        self.assertTrue(os.access(llm, os.X_OK))
        for alias in ("lmm",):
            link = self.bin_dir / alias
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), "llm")
        self.assertEqual(self.run_help(llm), 0)

    def test_replaces_existing_regular_file(self):
        self.bin_dir.mkdir(parents=True)
        (self.bin_dir / "llm").write_text("old")
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(zipfile.is_zipfile(self.bin_dir / "llm"))
        self.assertEqual(self.run_help(self.bin_dir / "llm"), 0)

    def test_replaces_existing_symlink(self):
        self.bin_dir.mkdir(parents=True)
        (self.bin_dir / "llm").symlink_to("/dev/null")
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.bin_dir / "llm").is_symlink())
        self.assertTrue(zipfile.is_zipfile(self.bin_dir / "llm"))
        self.assertEqual(self.run_help(self.bin_dir / "llm"), 0)

    def test_install_is_idempotent(self):
        for _ in range(2):
            self.assertEqual(self.install().returncode, 0)
        self.assertEqual(self.run_help(self.bin_dir / "llm"), 0)

    def test_missing_artifact_names_build_script(self):
        empty = self.tmp_path / "empty dist"
        env = dict(self.env, LMM_DIST_DIR=str(empty))
        result = subprocess.run(["bash", str(ROOT / "install.sh")], cwd=ROOT,
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("./build.sh", result.stderr)

    def run_help(self, executable):
        return subprocess.run([str(executable), "help"], env=self.env,
                              capture_output=True, text=True).returncode
```

- [ ] **Step 2: Run the install tests to verify they fail**

Run: `python3 -m unittest tests.test_build_install.InstallTest -v`
Expected: FAIL — `install.sh` does not exist, so every test fails to launch it.

- [ ] **Step 3: Implement `install.sh`**

Create `install.sh` with `#!/usr/bin/env bash` and `set -euo pipefail`. It must:

1. Resolve `ROOT` as in `build.sh`.
2. `DIST="${LMM_DIST_DIR:-$ROOT/dist}/llm"`, `DEST="${LMM_BIN_DIR:-$HOME/.local/bin}"`.
3. If `"$DIST"` is not a file, print `install.sh: $DIST not found; run ./build.sh first` to stderr and exit 1.
4. `mkdir -p "$DEST"`.
5. Copy atomically: `tmp="$(mktemp "$DEST/.llm.XXXXXX")"`, `trap 'rm -f "$tmp"' EXIT`, `cp "$DIST" "$tmp"`, `chmod 755 "$tmp"`, `mv -f "$tmp" "$DEST/llm"`, then clear the trap.
6. `ln -sfn llm "$DEST/lmm"`.
7. If `":$PATH:"` does not contain `":$DEST:"`, print a warning to stderr.
8. Print `Installed $DEST/llm`.

- [ ] **Step 4: Run the install tests to verify they pass**

Run: `python3 -m unittest tests.test_build_install.InstallTest -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Update the README development section**

Replace the three `ln -sf` lines under "Install the CLI and compatibility shims into `~/.local/bin`:" with:

```sh
./build.sh
./install.sh
```

State that this installs `llm` plus the `lmm` alias, that the artifact is `dist/llm` (self-contained bytecode but requires `python3` on `PATH`), and that `LMM_BIN_DIR` overrides the destination.

- [ ] **Step 6: Run the whole suite and commit**

Run: `python3 -m unittest discover -s tests -v`
Expected: all tests PASS.

```bash
git add install.sh tests/test_build_install.py README.md
git commit -m "feat: add install.sh for the llm zipapp"
```
