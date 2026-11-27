# `llm version` Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `version` command that reports the installed llm release (1.1.1) and states it is the latest version.

**Architecture:** `llm/__init__.py` holds the single source of truth (`__version__` and `LATEST_VERSION`). `llm/cli.py` adds `cmd_version`, reading both values at call time so tests can replace them, and registers the subcommand alongside the existing ones. No network access.

**Tech Stack:** Python 3 standard library (`argparse`, `unittest`, `unittest.mock`), existing zipapp build scripts.

**Spec:** `docs/superpowers/specs/2026-11-27-version-command-design.md`

## Global Constraints

- Both version constants live in `llm/__init__.py` and equal `"1.1.1"` for this release.
- `llm version` performs no network or git calls.
- First output line is exactly `<program> 1.1.1`, where `<program>` is `llm` or `lmm` from `_program_name()`.
- Up-to-date line is exactly `last version is 1.1.1. you are up to date.`
- Existing commands and tests must keep passing.
- Work happens on branch `fix/version-command-1.1.1`.

## Review Focus

- `lmm version` must say `lmm`, not `llm`; `_program_name()` already handles this, and the test must pin it.
- Constants must be read at call time, so patching `llm.LATEST_VERSION` in a test changes the output.
- A newer `LATEST_VERSION` must not claim the user is up to date; it must point at `update`.
- `llm help` must list `version`, and `llm help version` must work through the existing `_subparsers` map.
- The output must not gain extra blank lines or tracebacks on success.

---

### Task 1: Add the `version` command and 1.1.1 constants

**Files:**
- Modify: `llm/__init__.py`
- Modify: `llm/cli.py` (add `cmd_version`, register it in `build_parser`)
- Test: `tests/test_cli.py`
- Modify: `README.md` (command table)

**Interfaces:**
- Consumes: `llm.cli._program_name() -> str` (existing) and the `add(name, handler, help_text)` helper inside `build_parser`.
- Produces: `llm.__version__: str`, `llm.LATEST_VERSION: str`, and `llm.cli.cmd_version(args) -> int`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py` (the file already imports `io`, `sys`, `unittest`, `redirect_stdout`, `mock`, `config`, `cli`):

```python
class VersionCliTest(unittest.TestCase):
    def _output(self, argv):
        buffer = io.StringIO()
        with mock.patch.object(config, "migrate_state", return_value=False), \
             redirect_stdout(buffer):
            self.assertEqual(cli.main(argv), 0)
        return buffer.getvalue()

    def test_version_prints_installed_and_up_to_date(self):
        self.assertEqual(self._output(["version"]).splitlines(), [
            "llm 1.1.1",
            "last version is 1.1.1. you are up to date.",
        ])

    def test_version_uses_invocation_name(self):
        with mock.patch.object(sys, "argv", ["/Users/x/.local/bin/lmm"]):
            lines = self._output(["version"]).splitlines()
        self.assertEqual(lines[0], "lmm 1.1.1")

    def test_version_points_to_update_when_latest_is_newer(self):
        with mock.patch("llm.LATEST_VERSION", "1.2.0"):
            lines = self._output(["version"]).splitlines()
        self.assertEqual(lines, [
            "llm 1.1.1",
            "last version is 1.2.0. run `llm update` to update.",
        ])
```

Also add `"version"` to the command tuple in the existing
`test_help_exits_zero_and_names_commands`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_cli.VersionCliTest -v`
Expected: FAIL — `version` is not a valid choice / constants missing.

- [ ] **Step 3: Implement the constants**

In `llm/__init__.py`, after the docstring:

```python
__version__ = "1.1.1"
LATEST_VERSION = "1.1.1"

__author__ = "Davide Listello <davide.listello@gmail.com>"
```

- [ ] **Step 4: Implement `cmd_version` and register it**

In `llm/cli.py`, add near `cmd_update`:

```python
def cmd_version(args):
    from . import __version__, LATEST_VERSION
    program = _program_name()
    print("%s %s" % (program, __version__))
    if __version__ == LATEST_VERSION:
        print("last version is %s. you are up to date." % LATEST_VERSION)
    else:
        print("last version is %s. run `%s update` to update."
              % (LATEST_VERSION, program))
    return 0
```

Inside `build_parser`, after the `update` command is registered, add:

```python
add("version", cmd_version, "show the installed version and whether it is current")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_cli.VersionCliTest -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Update the README command table**

Add this row after the `llm update` row:

```markdown
| `llm version` | Show the installed version and whether it is the latest release. |
```

- [ ] **Step 7: Run the full suite and build artifact check**

Run: `python3 -m unittest discover -s tests`
Expected: all tests pass (`OK`).

Run: `LMM_SKIP_LLAMA_CPP_UPGRADE=1 ./build.sh`
Expected: `Built .../dist/llm`; the script's final `help` smoke test exits 0.

Run: `./dist/llm version`
Expected exactly:

```text
llm 1.1.1
last version is 1.1.1. you are up to date.
```

- [ ] **Step 8: Commit**

```bash
git add llm/__init__.py llm/cli.py tests/test_cli.py README.md \
        docs/superpowers/specs/2026-11-27-version-command-design.md \
        docs/superpowers/plans/2026-11-27-version-command.md
git commit -m "feat: add llm version command and release 1.1.1"
```
