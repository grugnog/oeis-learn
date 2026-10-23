# Focused Phase 2 correction: bounded fitting completion

The user authorized the focused correction after the reviewed replay hit its
two-second worker deadline without returning a result. This reopens the
T032–T036/T058 fitting and containment boundary under Spec 008. It does not
resume or replace the completed negative T096 attempt.

## Fixed scope

Keep the two-second native/production ceiling, independent-reference reservation,
four-second candidate call and sixty-second portfolio. Native work uses the
earlier of the caller deadline and two seconds from entry. Freeze internal
policy `native-fit-deadlines/v1`: reserve 5% of the remaining native allowance
for response delivery and 25% for final production checks. Search and symbolic
work share the preceding 70%. These proportional reserves shrink with a shorter
caller allowance and never extend or restart it. Expired work remains expired.
External supervision remains the hard bound under scheduler/solver overrun.

Preserve the language, abstraction, solver subset/bounds/seeds, maximum eight
assignments, discovered-assignment ranking, production checks and independent
admission. An early SAT assignment may be checked after enumeration runs out of
its search slice; UNKNOWN itself never supplies a model. Incomplete uniqueness
and search outcomes remain explicit. Eight assignments already prove multiple
solutions, so no ninth solver check is needed to establish ambiguity. A
single retained assignment still needs the blocking check to establish unique
versus multiple and cannot be called unique after a timeout.

No raw fallback, threshold change, hidden-value use or training experiment is
included. Binary/source/cache identities change; future qualification must bind
the new image. This correction improves bounded delivery and use of discovered
assignments, without promising success for the previously timed-out seed.

## Work and acceptance

- [X] F001 Add native regressions for deadline nesting, short/expired caller
  allowances, late enumeration with early SAT assignments, production overrun,
  the eight-assignment cap and single-model ambiguity.
- [X] F002 Implement proportional deadline partitions and route symbolic,
  affine/SMT and production work through their existing parent allowance.
- [X] F003 Retain bounded search/deadline telemetry without bypassing independent
  admission, and stop the redundant ninth check after a full ambiguous shortlist.
- [X] F004 Add packaged-native regressions for the observed candidate and
  independent fixtures, normal response delivery and healthy worker recovery.
- [X] F005 Run available local checks, push incremental commits and publish
  container validation/replay commands. Actual native/HIP evidence is reported
  separately; do not wait for PR CI.

These checkmarks record implementation and available local checks. The focused
native execution gates below are now covered by reviewed CI artifacts; full
Phase 5 qualification remains separate:

- [X] N001 Compile and run the new Rust tests in the managed image. The normal
  image build already runs all core tests and retains `core-tests.txt`.
- [X] N002 Run the packaged fitting/admission suite and optional single-case
  replay against that rebuilt image. Review returned outcomes, timing policy,
  transport completion and worker health. A delivered empty timeout is an
  honest result; success on this particular seed is not an acceptance gate.

## Implementation and local validation — 2026-10-21

`fit_budget.rs` computes nested deadlines without extending the caller.
`fit.rs` routes symbolic construction, affine solving and SMT through the search
deadline, checks final source/production within its later deadline, and keeps
all results provisional. `smt.rs` counts actual checks and stops at the eighth
SAT assignment while retaining the blocking check for a single assignment.
Ranking, coefficient scopes, seeds and independent admission are unchanged.
The private clock/search test seam is unavailable through requests or CLI flags.

Rust tests cover nested/tiny/short/expired allowances, early SAT plus late search,
no-model timeout, rejected late production, eight distinct exact assignments
in eight checks, and unique/multiple classification with a one-model cap.
Packaged tests independently admit ordinary affine/nonlinear fixtures, inspect
bounded timing/search metadata and exercise the observed visible polynomial
with its original greedy seed. The latter requires a complete native status
and a healthy subsequent execute call, allowing honest unknown/exhausted output
without manufacturing a fit or replaying hidden scoring.

Available local regression checks: **96 passed, 7 packaged-native skips,
1 engine deselection**, zero failures/errors. Ruff F/E9, Python formatting,
standard-library-only help and whitespace checks passed. The source-bound
[local receipt](phase-2-fit-deadlines-cpu.json) records the exact scope.
No Rust toolchain, container engine or HIP device is available here. Rust tests
and actual packaged delivery have therefore **not** been run locally. These
tests are included in the normal managed image build and fitting qualification;
CI is deliberately left unmonitored.

## Completed CI evidence and binding repair — 2026-10-21

On the user's subsequent continuation, reviewed completed jobs from workflow
`39684406792` for head `bc51f7a93047a228eb3e0aa1e11300bf809c7210`.
The managed Docker image compiled the native correction. Its core suite passed
**20 tests**, including all seven new deadline/enumeration regressions. Packaged
fitting passed **37 tests with zero failures or skips**. The observed candidate's
complete-status/healthy-worker regression passed in 1.942 seconds; that records
delivery and subsequent execution, not a claim that this seed found a fit.
Archive SHA256, JUnit SHA256, source identity and matching native build identity
were independently checked. See the [bounded CI review](phase-2-fit-deadlines-ci.json).
This closes N001/N002 for the focused CPU fitter correction; the optional full
candidate replay had not yet been run at that CI review; it is covered below.

The general CPU review and both public Docker/Podman CLI jobs failed the same
assertion: `current-dispositions.json` still bound the Phase 4 Python package
hash. Reproduced the failure locally, refreshed its current implementation hash
and descriptive revision, and checked all 68 original item/task mappings plus
the historical template hash. All dispositions, pending tasks and the false
full-completion flag remain unchanged. The focused owning tests then passed
**54 tests with zero failures/skips**; the Spec 007 structural/schema validator
also passed. This is a traceability repair, not new runtime qualification.
No Python/Rust implementation bytes, dependencies or source snapshot identities
changed in that repair. The new public-engine CI rerun remains pending; the
long-running Podman image build is not polled or awaited. No HIP, training or
hidden scoring work was executed.

### User Podman validation and exact candidate replay — 2026-10-21

The subsequent user artifacts close that optional replay. Reviewed
`008-fit-deadlines-20261021T221541Z(2).zip` and
`008-fit-deadlines-replay-20261021T225952Z.zip`: verified all seven qualification
and nine replay archive entries, including byte counts and SHA256. Qualification
passes all 37 fitting cases with zero failures/errors/skips, matches the current
source snapshot, and records read-only root, no network, no privilege, only the
output bind and cleanup within its bound. Replay uses that exact qualified
image; its runtime identity matches the qualification worker/guest hashes.
Case, parent contract, original source and mounted utility hashes match the
reviewed original diagnosis. See the [replay review](phase-2-fit-deadlines-replay-review.json).

With original greedy seed 75271729 and the unchanged abstraction/resources,
the fit worker completes in **1.479677 seconds** against **1.999861 seconds**
allowed, with zero restarts and one returned fit result. Search finds five SAT
assignments at 16 bits, then exhausts its search slice on check six (reason
`canceled`). Production checks all five and independent admission verifies all
20 visible indices for each; final outcome is `verified_solution`. Repeated the
independent reference admission for all five during this review. Each assignment
retains the same polynomial `n^2 + 11*n - 15`; no extra indices were scored.
Native fitting takes 1.307627 seconds; the entire raw-plus-fitting diagnostic
takes 2.199450 seconds. Host and nested ledgers retain their full reservations,
and cleanup completes in 0.305635 seconds.

This demonstrates the intended correction: an exhausted enumeration slice no
longer discards its already found, verifiable assignments or consumes response
delivery time. The focused reproduced case and bounded diagnostic/reliability
pass are complete. More same-case replay, HIP execution or a larger synthesis
spec is unnecessary for this defect. The original T096 outcome remains NEGATIVE;
this single visible replay does not grant 32/32 qualification or structural
generalization. Next review the remaining Phase 5 gates and require a separately
preregistered, source-bound attempt before any new qualification experiment.

## Run managed native validation

From the updated checkout, build once, then run only CPU native fitting and
independent admission. The new `--fitting-only` mode checks that packaged source
matches the checkout, fails on skipped/failed mandatory cases, and always writes
a bounded reports archive. It excludes preparation, measurements, GPU requests
and checkpoint/training work. `qualified` remains false for this targeted scope.
Dependency manifests and Docker layer ordering were not changed.

```bash
python3 scripts/foundation_run.py --engine podman image ensure
fit_validation_out="tmp-validation/008-fit-deadlines-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/qualify_bigint_phase5.py --engine podman \
  --image oeis-learn:gpu --fitting-only --output "$fit_validation_out"
```

Share `$fit_validation_out.zip`, including failures. Substitute Docker
consistently if needed. Although the existing image target is named `gpu`, this
suite and replay use CPU roles and require no HIP device. A source rebuild is
required for the native fix; the original image cannot validate it.

Optional new visible-only diagnostic replay, using the same reviewed collection
from the earlier diagnosis:

```bash
python3 scripts/diagnose_bigint_run.py replay --engine podman \
  --image oeis-learn:gpu \
  --evidence "$diagnostic_out/collection" \
  --output "$fit_validation_out/replay-current"
```

`diagnostic_out` is the directory already used for collection in the
[diagnostic commands](phase-5-diagnostic-followup.md#collect-and-replay-the-original-failed-prompt).
The replay publishes `$fit_validation_out/replay-current.zip` separately from
the qualification archive. The explicit image override is essential: without
it replay intentionally uses the original parent image. Expect `same_image`
and `same_runtime` to be false for the changed binary; inspect its new identities,
`timing`, `searches` and full `worker_calls` instead. This does not recollect,
resume or amend T096 and does not authorize training. No HIP validation is
needed to assess this CPU fitter correction; future GPU training/cost work must
obtain matching new-source qualification rather than reuse historical receipts.
