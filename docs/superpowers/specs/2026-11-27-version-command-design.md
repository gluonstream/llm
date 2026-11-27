# `llm version` command and 1.1.1 release — design

Date: 2026-11-27
Status: approved

## Intent

`llm version` currently fails with `invalid choice: 'version'` because the
command was never registered. Users have no way to ask which release they are
running. Add a `version` command and release it as 1.1.1.

## Goals

- `llm version` prints the invocation name and installed version, then states
  whether that is the latest release.
- Single source of truth for the version: `__version__` in `llm/__init__.py`.
- The command appears in `llm help`, `llm help version`, and the README
  command table.

## Non-goals

- No network call from `llm version`.
- No change to `llm update`; it remains the real updater.
- No packaging metadata, PyPI publishing, or release pipeline.

## Approaches

1. **Bundled latest version (chosen).** `llm/__init__.py` carries both
   `__version__` and `LATEST_VERSION`, currently `1.1.1`. `version` compares
   them locally. Fast, deterministic, testable, works offline.
2. **Live GitHub check.** Fetch `llm/__init__.py` from `main` on each run.
   More meaningful over time, but adds latency, timeout handling, and an
   offline failure path. Rejected for this small change.
3. **Check the temp update checkout.** Reuse `/private/tmp/llm-update`.
   Rejected: the checkout is often absent or stale, and absence would make the
   command lie.

## Design

- `llm/__init__.py`: `__version__ = "1.1.1"`, `LATEST_VERSION = "1.1.1"`.
- `llm/cli.py`:
  - `cmd_version(args)` uses `_program_name()` and prints exactly:
    ```
    llm 1.1.1
    last version is 1.1.1. you are up to date.
    ```
    (`lmm version` uses `lmm` for the first line.)
  - If the versions differ, print the installed version and
    `last version is <latest>. run \`llm update\` to update.`
  - Register `version` with a one-line help string; no arguments.
- `llm update` is unchanged; bumping the release means editing the two
  constants together.
- README: add `llm version` to the command table and the quick start/commands
  context.

## Tests

- `version` prints `<program> 1.1.1` and the up-to-date line.
- `version` prints an update hint when `LATEST_VERSION` is newer.
- `help` lists `version`.
- Existing suite stays green.

## Left to the builder

- Exact wording of the newer-version line.
- Whether to read constants at call time so tests can monkeypatch them
  (preferred).
