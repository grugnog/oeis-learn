# Phase 4 migration and retirement (T081–T088)

**Phase 4 is complete: T081–T088 are closed.** Entry was merged Phase 3 PR #56
at `66d13fdb93bc7d107c0f1ac8fe90cf030dea47e9`; its T080 evidence is preserved.
The sections before the exit review retain the contemporaneous progression while
[PR #57](https://github.com/grugnog/oeis-learn/pull/57) was open with the failing
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
| T088 | Complete. All inventory code dispositions are closed, and the reviewed clean installed/native/compatibility jobs pass on the recorded image. CI and the public retirement suite supplied the gate; no source-only result is labelled clean-install qualification. |

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

This was the condition for checking T088. The exit review below records the
actual new-image results that satisfied it. `.specify/extensions.yml` is absent;
no post-implementation hook is registered.

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
The earlier 200-test record remains bound to its original snapshot. At this
repair checkpoint T088 remained open until the newly pushed container/native/
compatibility checks completed. The exit review below records their completion.

### Host certificate prerequisite follow-up after `74b7105`

Container run `39349647960` passed every job, including actual G9 certificate
execution and both portable engines. Review run `39349646906` passed lint,
artifact consistency, input diagnostics and quickstart, but its host CPU suite
failed two positive loop-certificate cases because the native certificate
checker is available only in the qualification image.

Loop-certificate tests now declare their native prerequisite, including the
negative ranking case so a missing solver cannot make rejection pass vacuously.
Other certificate families still run on the host. Image qualification explicitly
requires both native artifact flags; missing artifacts fail rather than skip,
and G9 still rejects any skipped owning case. The local affected set passes
**50 tests with three explicit native-certificate skips**. A separate missing
artifact probe confirms required-native execution errors instead of skipping.
The prior CI artifact's digest and G9 results were inspected; no local native
execution is claimed from that evidence. See
[phase-4-certificate-repair.json](phase-4-certificate-repair.json).
At this checkpoint T088 remained pending the new review/qualification run. The
exit review below records its later completion.

## Phase 4 exit review

T088 passes at `8556cce87aec5d5b58f9a163e5c5d266a1f61341`, merged by PR #57
as `be6070bead5f30147774b6517c0f50eb1cb3f6cc` without production source changes.
Review run39355476087 succeeds. Container run39355476126 has successful image,
installed/native, training/replay, G7/G8 software/G9 and both portable-engine
execution jobs; only the separate non-critical Podman rebuild was cancelled.
Verified artifact ZIP digests and individual JUnit counts are recorded in
[phase-4-exit.json](phase-4-exit.json). The arbitrary-UID contract artifact has
163 algorithm, 43 integration and 27 retirement cases with zero failures/errors/
skips, plus a clean isolated installed audit. All native qualification owning
suites pass without skipped cases. The native preparation artifact repeats the
27 source retirement cases with one installed-context skip; that same case
executes successfully in the required arbitrary-UID installed context. No skip
is being counted as installed evidence. Historical reports remain bound to
their original revisions; this closure grants no Phase 5 or research pass.
