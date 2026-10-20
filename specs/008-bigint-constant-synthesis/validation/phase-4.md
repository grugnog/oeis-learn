# Phase 4 migration and retirement (T081–T088)

**T081–T087 code work is complete. T088 remains pending actual clean-image
execution.** Entry is merged Phase 3 PR #56 at
`66d13fdb93bc7d107c0f1ac8fe90cf030dea47e9`; its T080 evidence is preserved.
[PR #57](https://github.com/grugnog/oeis-learn/pull/57) was opened with the failing
retirement audit before migration. Commits are added as work proceeds; new CI
runs are deliberately left unmonitored.

## Implemented disposition

| Tasks | Change and owning checks |
| --- | --- |
| T081 | Source and clean installed-distribution audits reject retired modules/imports, literal dynamic imports (including aliases), old callable names and nontrivial unchanged function bodies copied under another name. Installed checks use `python -I` outside the checkout, inspect importability/package dependencies and require the packaged native artifacts. The initial source audit reported 52 violations; the final source audit passes. |
| T082 | CLI, legacy diagnostic evaluation, qualified activation and G7 use `synthesis.fitting` and the owning native client/independent admission. Native records have an explicit new kind. Unsupported scalar/placeholder/wrapping/bitwise/recurrence fitting forms return explicit migration outcomes; concrete raw legacy diagnostics retain their execution semantics and mismatch checks. |
| T083 | RL fitting uses the same facade. Verified training-archive admission remains; retired legacy teacher ingestion fails explicitly. Old JSON/qualified grounding records remain read-only `historical_unreplayed`. New solution replay checks actual native execution plus independent admission; UNSAT replay rederives the scoped native problem. The independent loop-certificate checker retains its existing initiation/consecution/ranking/exit and checked-range obligations, serializes SMT-LIB, and uses the pinned Rust/Z3 certificate checker with the same 1500ms per-obligation allowance. It has no model/assignment API; missing/failed/unknown native execution cannot prove a certificate. |
| T084 | Owning tests are ported to the public service and independent reference. Polynomial/square16-versus4, shared/large coefficients, singular prime pivots, nonintegral/inconsistent/rank-deficient systems, carried-state Fibonacci arithmetic, intermediate overflow, visible-only fitting, and provisional/tampered evidence rejection remain covered. Old scalar/bitwise/control fitting now has explicit unsupported tests. Historical schema tests retain old record meanings without callable compatibility solvers. |
| T085 | Eight old implementation modules and four public skeleton/solver/filter result dataclasses are removed. `fresh.search.grounded_variants` is removed; the worker already used Phase 3's qualified native route. |
| T086 | run008/009/010 launchers and their three exclusive configs are removed. Retained run011/tier1 diagnostic configs have explicit native settings, replacing obsolete solver switches. Retained training loaders reject old settings before model construction. README/quickstart document replacements, unsupported methods and reproduction revision; historical specs/results are unchanged. |
| T087 | Python z3-solver is removed from metadata/lock, and the image installer removes an inherited copy if present. The other locked package pins and native Z3/Wasmtime/toolchain pins are unchanged. SymPy remains for documented independent certificate/discovery consumers. The new native checker reuses the cached dependency closure. |
| T088 | All inventory code dispositions are closed; the clean installed/native/compatibility evidence gate remains open until the new image passes. CI and the public retirement suite execute that gate; no source-only result is labelled clean-install qualification. |

See the completed path inventory in
[retirement-inventory.json](../retirement-inventory.json), the migration rationale
in [retirement.md](../retirement.md), and hash-bound local evidence in
[phase-4-local.json](phase-4-local.json). Signed-256 validators, arithmetic/codec
profiles and checkpoint readers keep their original numerical semantics. Native
fitting/abstraction/ranking algorithms are unchanged. Bounded discovered-set
ranking does not claim a global optimum. Certificate serialization is a backend
migration of the existing verifier, not a new proof policy or Python fitter.

## Local validation and limits

Python3.12.14/Torch2.9.1+cpu affected regression set: **200 passed, 27 explicit
native/installed-image skips, one GPU deselection; zero failures/errors**.
The JUnit hash, each skip and implementation/snapshot identities are in the
machine-readable record. Source retirement, current caller imports, symbolic
certificate request bounds/failure behavior, raw legacy diagnostics, bigint
encoder/decoder, CPU checkpoint/resume and retained bootstrap controls pass.
Ruff F/E9, Python compilation, shell syntax, lock consistency and `git diff
--check` are checked separately.

A broader collection cannot run here because the compiled native extension is
absent. A contract probe passed 161 tests with two skips before an existing test
failed on unavailable `/proc/sys/kernel/random/boot_id`; a subsequent broader run
was interrupted and is not acceptance evidence. This environment has no usable
Docker/Podman/Rust toolchain, so native compilation, actual certificate solving,
clean installed execution and zero-skip native regression success are not
claimed locally. Required image jobs fail rather than skip missing artifacts.
No HIP execution is claimed from these CPU tests.

## Required clean-image gate and commands

The shared qualification image CI runs:

1. The existing required bigint CPU/native contracts and Phase 3 integration.
2. `python -I -m oeis_learn.synthesis.retirement --installed` and the Phase 4
   audit/caller regressions under arbitrary UID12345 with read-only root.
3. G7's migrated native fitting/replay regressions and retained native checks;
   existing discovery/certificate acceptance uses the new native checker.
4. Existing Docker/Podman portability and installed resource jobs.

To run the complete public retirement suite from this PR checkout:

```bash
python3 scripts/foundation_run.py --engine podman validate \
  --suite bigint-retirement --output tmp-validation/008-phase4-retirement-001
```

Substitute `docker` as appropriate. Omit `--image` to build the current checkout
using cached pinned layers. Use a fresh output suffix for a new attempt. Retain
`validation.json`, stage receipts/logs, installed `audit.json` and `results.xml`.
The suite requires zero failed/error/skipped selected tests and validates the
installed wheel separately from source imports. `qualified: false` does not
negate fixture PASS or grant later research qualification.

Phase 4 introduces no HIP-only task: neural architecture/numerical policy and
GPU selection are unchanged. The previous three-test HIP evidence stays bound
to its submitted image; eventual Phase 5 GPU qualification must identify that
phase's effective image. No extra HIP run, research screen, final evaluation,
merge or automatic promotion is launched by this work.

T088 can be checked only after the actual new-image results satisfy the gate.
`.specify/extensions.yml` is absent; no post-implementation hook is registered.

## CI repair after `d24c7f7`

Review run `39347074139` failed because Ruff named six deleted modules and the
host-only training integration lacked the now-required native worker. Container
run `39347074171` built the image successfully, passed native preparation, the
installed retirement contracts and G7, then failed on a co-training fixture that
still inserted a retired teacher. Both engine jobs reached their public CPU
suite and failed only two assertions that expected the deleted launchers.

The review workflow now lints `synthesis` rather than retired module paths.
Training/replay integration runs in the existing shared qualification image,
with mandatory native artifacts and durable JUnit output; no additional image
build is introduced. Local training tests explicitly require the real native
fixture. Retained wrapper inventory excludes the retired launchers, and tests
require those launchers and exclusive recipes to be absent. The co-training
test exercises on-policy KL with an empty legacy buffer; a separate assertion
requires retired teacher ingestion to fail atomically.

Local repair checks: **38 passed, six native/installed-image skips, zero
failures/errors**; another **55 supporting tests passed** for policy, replay,
handoff, learning gates and container validation. All review-workflow Ruff commands, both changed workflows'
shell blocks, source retirement audit, current review disposition binding and
Spec Kit artifact consistency pass. See
[phase-4-ci-repair.json](phase-4-ci-repair.json) for source/test identities.
The earlier 200-test record remains bound to its original snapshot. This repair
does not close T088; the newly pushed container/native/compatibility checks must
complete successfully first. New CI is deliberately left unmonitored.
