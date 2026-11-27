# Build and install scripts (zipapp) — design

## Intent

Replace the current "symlink the repo launcher" install with a real build
artifact: a single-file, bytecode-bearing executable for `llm`, plus a
repeatable installer.

## Current state

`bin/llm` puts the repo root on `sys.path` and calls
`llm.cli.main()`. `~/.local/bin/{llm,lmm}`
are symlinks to it, so the CLI runs directly from the source tree and there
is no build or install step.

## Goals

- `./build.sh` produces `dist/llm`, a single executable zipapp containing
  compiled `.pyc` bytecode (with `.py` sources as a fallback) and a
  `#!/usr/bin/env python3` shebang.
- `./install.sh` atomically installs `dist/llm` as `~/.local/bin/llm`.
- `lmm` remains available as a symlink to the
  installed `llm`.
- The installed command works after the repo moves or is deleted, as long
  as `python3` is on `PATH`.
- Re-running both scripts is safe and idempotent.

## Non-goals

- Bundling a Python interpreter (no PyInstaller/standalone binary).
- Supporting Python 2 or Windows.
- Publishing to PyPI or installing package metadata.

## Constraints

- A bare `.pyc` is not directly executable and is tied to the Python
  version that produced it; a zipapp with a shebang plus source fallback
  handles both problems.
- The package is stdlib-only today; keep the scripts dependency-free.
- Build output must not pollute the source tree: stage in `build/`, emit to
  `dist/`, and add both to `.gitignore`.

## Approaches considered

1. **Executable zipapp (chosen)** — one file, needs `python3`, repo-independent,
   trivially installable.
2. **Keep symlinks** — no build, live edits, but coupled to the repo path.
3. **PyInstaller onefile** — no Python dependency, but heavy, slow, and
   awkward to sign on macOS; overkill for a stdlib-only CLI.

## Design

`build.sh`:

1. Resolve the repo root from `$0` and fail if `python3` is missing.
2. Create a clean staging tree `build/app/` containing a copy of
   `llm/` (no `__pycache__`).
3. Run `python3 -m compileall -b` on the staged package, producing legacy
   `.pyc` files next to the `.py` sources.
4. Write a `__main__.py` that calls `llm.cli.main()` and
   exits with its return code, then compile it too.
5. Build `dist/llm` with `python3 -m zipapp build/app -p "/usr/bin/env python3" -c`
   and `chmod +x`.
6. Smoke-test `dist/llm help` with `LMM_STATE_DIR` pointed at a temp dir.

`install.sh`:

1. Resolve the repo root; require `dist/llm` (tell the user to run
   `./build.sh` if absent).
2. Install to `${LMM_BIN_DIR:-$HOME/.local/bin}` with a temp file +
   `mv -f`, so a running command can never see a partial file.
3. Point `lmm` at `llm` (relative symlink).
4. Warn if the install directory is not on `PATH`.

## Calls made by the builder

- Install both names, not only `llm`: the README and the existing
  setup promise `lmm`.
- Keep `.py` sources inside the zip as a fallback for Python upgrades;
  `.pyc` is preferred when the version matches.
- Do not run any of this automatically as part of tests.
