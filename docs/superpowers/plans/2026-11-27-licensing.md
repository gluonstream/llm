# Licensing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AGPL-3.0-only licensing, a commercial-license notice, and a lightweight CLA so the Python project can be offered under both AGPL-3.0 and a commercial license.

**Architecture:** Documentation-only change. The canonical AGPL text lives in `LICENSE`; README explains the dual-license model; `CLA.md` and `CONTRIBUTING.md` establish the contribution terms needed for commercial relicensing. No runtime code changes.

**Tech Stack:** Markdown and plain-text license files; existing `unittest` suite as a regression check.

**Spec:** `docs/superpowers/specs/2026-11-27-licensing-design.md`

## Global Constraints

- SPDX identifier is exactly `AGPL-3.0-only`; do not use `AGPL-3.0-or-later`.
- Copyright holder is `Davide Listello <davide.listello@gmail.com>`.
- Commercial licensing contact is `davide.listello@gmail.com`.
- The future Rust implementation is a separate product and is not part of this repository.
- Do not modify runtime code.
- `.DS_Store` files are ignored and untracked.

## Review Focus

- The project is described as AGPL-3.0-only, not or-later: README and LICENSE preamble must both say so.
- The commercial path is explicit: proprietary/cloud use that cannot comply must contact the copyright holder.
- The CLA must actually permit commercial relicensing: it needs a sublicensable copyright grant, not just a contribution license.
- The Rust product must not be described as covered by this repository's AGPL license.
- Every relative link added to `README.md` must resolve.

---

### Task 1: Add the canonical AGPL-3.0 license

**Files:**
- Create: `LICENSE`

**Interfaces:**
- Consumes: nothing.
- Produces: `LICENSE`, linked from `README.md` in Task 2.

- [ ] **Step 1: Fetch the canonical license text**

```sh
tmp="$(mktemp)"
curl -fsSL https://www.gnu.org/licenses/agpl-3.0.txt -o "$tmp"
```

Expected: the file begins with `GNU AFFERO GENERAL PUBLIC LICENSE` and contains `END OF TERMS AND CONDITIONS`.

- [ ] **Step 2: Create `LICENSE` with a project preamble followed by the canonical text**

```sh
{
  printf '%s\n' \
    'llm' \
    'Copyright (C) 2026 Davide Listello <davide.listello@gmail.com>' \
    '' \
    'This project is licensed under the GNU Affero General Public License,' \
    'version 3 only (AGPL-3.0-only). The full license text follows.' \
    ''
  cat "$tmp"
} > LICENSE
rm -f "$tmp"
```

Do not edit the canonical license text itself.

- [ ] **Step 3: Verify the file is complete and correctly identified**

```sh
sed -n '1,8p' LICENSE
grep -q 'AGPL-3.0-only' LICENSE
grep -n 'GNU AFFERO GENERAL PUBLIC LICENSE' LICENSE
grep -n 'END OF TERMS AND CONDITIONS' LICENSE
wc -l LICENSE
```

Expected: the preamble contains `AGPL-3.0-only`, the title appears, the end marker appears, and the file is at least 600 lines.

- [ ] **Step 4: Commit**

```sh
git add LICENSE
git commit -m "docs: add AGPL-3.0 license"
```

### Task 2: Document dual licensing in the README

**Files:**
- Modify: `README.md` (append a new `## License` section at the end)

**Interfaces:**
- Consumes: `LICENSE` from Task 1.
- Produces: the public licensing explanation linked to `CLA.md` and `CONTRIBUTING.md` from Task 3.

- [ ] **Step 1: Append the License section**

Add this exact section to the end of `README.md`:

```markdown
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
```

- [ ] **Step 2: Verify the section and links**

```sh
grep -n '^## License' README.md
grep -n 'AGPL-3.0-only' README.md
grep -q 'separate written commercial license' README.md
grep -q 'separate product' README.md
test -f LICENSE
grep -q '](LICENSE)' README.md
grep -q '](CLA.md)' README.md
grep -q '](CONTRIBUTING.md)' README.md
```

Expected: all commands succeed.

- [ ] **Step 3: Commit**

```sh
git add README.md
git commit -m "docs: explain AGPL and commercial licensing"
```

### Task 3: Add the CLA and contribution guide

**Files:**
- Create: `CLA.md`
- Create: `CONTRIBUTING.md`

**Interfaces:**
- Consumes: the licensing terms from Task 2.
- Produces: the contribution terms referenced by `README.md`.

- [ ] **Step 1: Create `CLA.md`**

```markdown
# Contributor License Agreement

This Contributor License Agreement ("CLA") is between you ("Contributor") and
Davide Listello ("Project Owner") for contributions to the `llm` project.

"Contribution" means any original work of authorship, including source code,
documentation, tests, and other material, that you submit to the project.

You agree to this CLA for your present and future contributions when you sign
it as described in section 5. Submitting a pull request without signing does
not by itself grant the licenses below.

## 1. Copyright license

You retain copyright in your contributions. You grant the Project Owner and
its successors and assigns a perpetual, worldwide, non-exclusive, royalty-free,
irrevocable copyright license, with the right to sublicense, to reproduce,
prepare derivative works of, publicly display, publicly perform, and distribute
your contributions and derivative works under any license, including the GNU
Affero General Public License, version 3 only (AGPL-3.0-only), and proprietary
or commercial licenses.

To the extent permitted by applicable law, you waive and agree not to assert
any moral rights or similar rights in your contributions against the Project
Owner, its successors and assigns, its sublicensees, or their respective users.

## 2. Patent license

You grant the Project Owner and its successors and assigns a perpetual,
worldwide, non-exclusive, royalty-free, irrevocable patent license, with the
right to sublicense, to make, have made, use, offer to sell, sell, import, and
otherwise transfer your contributions. This license applies only to patent
claims licensable by you that are necessarily infringed by your contributions
alone or by the combination of your contributions with the project. If any
entity institutes patent litigation alleging that your contribution or the
project infringes a patent, the patent license granted here terminates.

## 3. Representations

You represent that:

- you are legally entitled to grant the licenses above;
- each contribution is your original work, or you have received permission from
  the copyright holder to submit it under this CLA;
- if your employer has rights in your contributions, you have permission to
  make the contribution or your employer has waived those rights; and
- your contributions do not knowingly infringe any third party's copyright,
  patent, trademark, or other rights.

## 4. No warranty

Your contributions are provided "as is", without warranties or conditions of
any kind, express or implied, including warranties of title,
non-infringement, merchantability, or fitness for a particular purpose.

## 5. Signing

To sign this CLA, comment on your pull request with:

> I have read and agree to the CLA.
>
> Name:
> Employer (if applicable):
> Employer representative and title (if applicable):

Your pull request can be merged after the Project Owner confirms the CLA
sign-off. For contributions made on behalf of an employer, an authorized
representative of the employer must sign by including their name, title, and
employer in the comment. The Project Owner may require a separate corporate
CLA.
```

- [ ] **Step 2: Create `CONTRIBUTING.md`**

```markdown
# Contributing

Thanks for your interest in `llm`.

## Before you start

- Open an issue to discuss non-trivial changes before writing code.
- Keep changes focused and include tests where they make sense.
- Run the test suite before opening a pull request:

  ```sh
  python3 -m unittest discover -s tests -v
  ```

## Contributor License Agreement

The project is offered under AGPL-3.0-only and, separately, a commercial
license. To keep that possible, every contributor must agree to the
[Contributor License Agreement](CLA.md).

In your first pull request, add a comment with:

> I have read and agree to the CLA.
>
> Name:
> Employer (if applicable):
> Employer representative and title (if applicable):

You keep copyright in your contribution; the CLA gives the project owner the
right to distribute it under both licenses.

## Pull requests

- Describe what changed and why.
- Link the issue the change addresses, if there is one.
- Make sure the test suite passes.
- Keep the pull request focused; separate unrelated changes.
```

- [ ] **Step 3: Verify the CLA grants and sign-off text**

```sh
test -f CLA.md
test -f CONTRIBUTING.md
grep -n 'with the right to sublicense' CLA.md
grep -n 'moral rights' CLA.md
grep -n 'proprietary or commercial licenses' CLA.md
grep -n 'Employer representative' CLA.md CONTRIBUTING.md
grep -n 'I have read and agree to the CLA' CLA.md CONTRIBUTING.md
```

Expected: all commands succeed.

- [ ] **Step 4: Commit**

```sh
git add CLA.md CONTRIBUTING.md
git commit -m "docs: add contributor license agreement"
```

### Task 4: Final verification and commit the plan/spec

**Files:**
- Verify: `LICENSE`, `README.md`, `CLA.md`, `CONTRIBUTING.md`
- Add: `docs/superpowers/specs/2026-11-27-licensing-design.md`
- Add: `docs/superpowers/plans/2026-11-27-licensing.md`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: a clean, verified licensing change.

- [ ] **Step 1: Run the existing test suite**

```sh
python3 -m unittest discover -s tests -q
```

Expected: all tests pass. If the suite has unrelated environment failures, record the exact failure; do not change runtime code to fix it.

- [ ] **Step 2: Check links and whitespace**

```sh
test -f LICENSE
test -f CLA.md
test -f CONTRIBUTING.md
grep -q '](LICENSE)' README.md
grep -q '](CLA.md)' README.md
grep -q '](CONTRIBUTING.md)' README.md
git diff --check
```

Expected: all commands succeed with no output from `git diff --check`.

- [ ] **Step 3: Review the diff**

```sh
git diff --stat
git status --short
```

Expected: only the two `docs/superpowers/` planning files are untracked; the licensing files are already committed and no `.DS_Store` files are tracked.

- [ ] **Step 4: Commit the design and plan**

```sh
git add docs/superpowers/specs/2026-11-27-licensing-design.md docs/superpowers/plans/2026-11-27-licensing.md
git commit -m "docs: add licensing design and plan"
```

- [ ] **Step 5: Confirm the final state**

```sh
git log --oneline -4
git status --short
```

Expected: the licensing commits are present and the working tree is clean.
