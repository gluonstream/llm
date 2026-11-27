# Version and update commands — design

Date: 2026-11-27
Status: proposed

## Intent

`llm` / `lmm` has no way to say which version is installed or whether the
checkout is current. Add two small commands so the user can ask both questions
without leaving the tool.

## Goals

- `llm version` prints `llm 1.0.1` (the invocation name, then the version).
- `llm update` updates the checkout from its git remote and reports the result,
  including "You have the latest version (1.0.1)." when already current.
- Both commands appear in `llm help` and the README.

## Non-goals

- Packaging, PyPI publishing, or a release pipeline.
- Updating llama.cpp, models, or anything outside this repository.
- Auto-updating without the user running `llm update`.

## Approach

Single source of truth for the version: `llm/__init__.py`
exports `__version__ = "1.0.1"`, and the CLI reads it. No packaging metadata is
introduced.

`llm version` uses the existing `_program_name()` helper, so `llm version`
prints `llm 1.0.1` and `lmm version` prints `lmm 1.0.1`.

`llm update` locates the repository root from the package location
(`Path(__file__).resolve().parents[1]`) and runs `git -C <root> pull --ff-only`:

- Already current → `You have the latest version (1.0.1).`, exit 0.
- Pulled new commits → re-read `__version__` from the updated
  `__init__.py`, print `Updated to <version>.`, exit 0.
- The repository is not published yet, so it currently has **no git remote**.
  In that case `llm update` prints `You have the latest version (1.0.1).` plus
  a short note that no update source is configured, and exits 0. With no
  remote there is nowhere newer to fetch, so the local copy is the latest.
- Once a remote is added, the same command performs the pull described above
  with no code change.
- Git missing, not a git checkout, dirty tree, or network/merge failure →
  one clear error line on stderr and exit 1. Never claim "latest" when a
  configured update source could not be reached.
- No network call is invented: the checkout's configured git remote is the
  only update source.

## Tests

- `version` prints `<program> 1.0.1`.
- `update` reports up-to-date when `git pull` succeeds without new commits.
- `update` reports the new version when the pull changes `__init__.py`.
- `update` with no configured remote prints "You have the latest version
  (1.0.1)." and exits 0.
- `update` returns 1 and does not print "latest" when a configured remote is
  unreachable or the pull fails.

## Left to the builder

- Exact wording of the failure messages.
- Whether to run `git pull` via `subprocess.run(..., capture_output=True)` and
  how to detect "Already up to date" across git versions (prefer exit status
  plus comparing `git rev-parse HEAD` before and after).
