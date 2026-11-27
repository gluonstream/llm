# Licensing — design

Date: 2026-11-27
Status: approved

## Intent

Make the Python `llm` CLI publicly available as free/open-source software under
AGPL-3.0 while preserving the copyright holder's ability to sell commercial
licenses, including for a future separate Rust product.

## Goals

- Add canonical AGPL-3.0-only license text.
- Explain dual licensing and the commercial contact in the README.
- Establish contribution terms that allow commercial relicensing.
- Clarify that the Rust version is a separate product.

## Non-goals

- No change to runtime code.
- No custom "no cloud" license addendum.
- No per-file SPDX headers or license automation now.
- No claim that the future Rust product is covered by this repository.

## Approaches

1. **AGPL-3.0 only, no CLA.** Simple, but the project could not grant
   commercial exceptions for outside contributions. Rejected.
2. **Custom non-commercial/cloud-restricted license.** Not OSI-approved open
   source. Rejected.
3. **AGPL-3.0-only + commercial license + lightweight CLA.** Chosen.

## Design

- `LICENSE`: canonical AGPL-3.0 text with a short project copyright and
  `AGPL-3.0-only` preamble.
- `README.md`: License section covering the open-source terms, commercial
  licensing, Rust separation, and contributions.
- `CLA.md`: contributor retains copyright and grants the project owner a
  perpetual, worldwide, non-exclusive, royalty-free, irrevocable copyright and
  patent license, including the right to sublicense and distribute under any
  license. Includes representations and an "as is" disclaimer, and is signed by
  PR comment.
- `CONTRIBUTING.md`: short contribution guide and CLA sign-off instructions.
- Copyright holder: Davide Listello <davide.listello@gmail.com>.
- SPDX identifier: `AGPL-3.0-only`.

## Decided details

- AGPL-3.0-only, not or-later.
- CLA sign-off via PR comment with full name and employer if applicable.
- Commercial license contact: davide.listello@gmail.com.
- No license headers in source files for now.

## Left to the builder

- Exact wording of the CLA and README, keeping the legal grants intact.
- Whether to add a PR template later.
