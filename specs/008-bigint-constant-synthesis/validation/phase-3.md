# Phase 3 integration evidence (T059–T080)

**Phase 3 tasks T059–T080 are complete.** The completed CPU/native CI run passes
43 Phase 3 integration tests and 163 bigint contract tests without failures,
errors or skips. See the [exit review](#phase-3-exit-review--2026-10-19) and
[machine-readable evidence](phase-3-exit.json). Earlier pending notes below
describe the evidence available at those earlier commits.
The user-supplied integrated HIP rerun passes all three selected tests; see the
[reviewed evidence](phase-3-hip/review.json) and result below. No research screen,
retirement audit, merge, final-cohort evaluation or promotion was executed.

Entry: merged main `6d284b6597582e8900260c1397b796d274aad22b` (PR #55), whose
Phase 2 exit includes CPU/native and user-supplied actual-HIP evidence.
Algorithms, numerical profiles, ranking, grouping, curriculum, replay and
comparison policies remain the qualified Phase 2 implementations. Semantic
failures return to T058; integration does not invent a fallback solver.

## Implemented boundaries

| Tasks | Integration and evidence |
| --- | --- |
| T059–T062 | Four-byte little-endian framing, header rejection before body allocation, bounded count/byte batches, one owning-role FIFO, shared pending/cache limits, partial results, malformed/duplicate/stale rejection, bounded stderr/RSS/deadline and kill/reap/restart. Native dispatch calls the qualified core; Python only serializes, supervises and independently admits. Compiled modules are ephemeral native runtime objects; no serialized compiled artifacts are cached, and the cache report states this explicitly. |
| T063–T066 | Closed image-side YAML resolution, explicit/hash-bound defaults and artifacts, portable CLI/seed/fitting/ablation/compare/suites, immutable exact-image resume and analysis children, durable debit-before-work state, separate visible generation and private scoring mounts, no role network/socket, final-plan/decision validation and shared exposure/debit before final inputs are read or mounted. |
| T067–T070 | Exact indexed paired archives and full-size checks, native/reference admission records, private synthetic/OEIS truth, new input/codec construction and counts, native fitting in fresh search, immutable retained-checkpoint raw/exact/paired analysis under the original checked signed-256 subset. Unsupported retained candidates remain explicit; parent weights and ledger are unchanged. |
| T071–T074 | Seven checkpoint state components; deterministic fixture resume matches uninterrupted weights; fixed32/32/32 panels, full development loss, visible-only selection, end-of-screen128+128 reporting, training-only independently checked replay, prompt controls/templates/cache parity, strata and bounded diagnostics, qualified comparison/interval policies, five explicit frozen recipes. |
| T075–T077 | Reviewed assisted-only demonstrations are re-executed natively and independently, overlap exclusions and monotone full ancestry are checked, foundation/branch compute is separate, alternatives use qualified retention with8 per prompt and aggregate byte caps, finite-match/proof/novelty statuses remain separate. Bundles reserve core first (minimum16MiB/reserve8MiB/default64MiB); bounded detail omissions retain hash/size/reason. Checkpoint export is an explicit separate archive. |
| T078–T079 | Public CPU/GPU suites and blocked-until-Phase4 retirement registration, concise stage logs/failure summaries, native prepare/check/generate/fit/score fixture tests, actual model update/resume, mount isolation, final retry/debit, corruption and export pressure tests. |

The legacy inert `synthesis-record-stub/v1` entry points remain disabled. The new
explicit worker/role routes use qualified APIs. Full installed solver retirement
is T081–T088, not a claim made by this integration.

## Local results

The committed [machine-readable record](phase-3-local.json) binds implementation
and native source-closure hashes. Local Python3.12.14/Torch2.9.1+cpu results:

- Affected regression set: **142 passed, 3 explicit native-image skips**.
- Final changed-route set: **69 passed, the same 3 native-image skips**.
- Final stdlib/shape-contract recheck: **53 passed, the same 3 native-image skips**.
  This includes `python -S` imports with no site packages after making JSON-schema
  validation an image-side lazy dependency.
- Actual HIP fixture is marked GPU and excluded from CPU runs.
- Ruff F/E9 and `git diff --check` pass.
- Host import/CLI and orchestration retain standard-library-only dependencies;
  YAML, tensor operations and native builds execute in the image roles.

The skipped cases are the packaged framed fitter, native sealed selection versus
private truth, and full512/128/128 native preparation/check integration. They
fail rather than skip with `OEIS_REQUIRE_BIGINT_BUILD=1` in CI/public validation.
This workspace has no Docker/Podman or Rust toolchain, so Rust compilation and
these actual-native tests cannot be claimed from local source inspection.
Historical qualifications are unchanged; the rolling spec007 disposition binds
the current Python implementation without granting release acceptance.

Reproduce the local affected set inside the qualification image with pytest
against `test_bigint_workflow.py` (contract/integration), `test_synthesis_worker.py`,
`test_foundation_local.py`, `test_fresh_cli.py`, `test_fresh_workflow.py`,
`test_bootstrap_workflow.py`, `test_bootstrap_bundle.py`, `test_container_validation.py`,
`test_synthesis_service_limits.py`, `test_synthesis_state.py`,
`test_synthesis_provenance.py`, `test_synthesis_archive.py` and
`test_foundation_task_inventory.py`, using `-m 'not gpu and not docker'`.
CI explicitly runs all three new test files in the arbitrary-UID, read-only
qualification image after Phase 2 contracts; private pytest scratch stays in
`/tmp`, outside uploaded evidence.

## Copy-pastable native and HIP validation

From this PR checkout on the machine with a working engine:

```bash
python3 scripts/foundation_run.py --engine podman validate \
  --suite bigint-cpu --output tmp-validation/008-phase3-cpu-001

python3 scripts/foundation_run.py --engine podman validate \
  --suite bigint-gpu --output tmp-validation/008-phase3-hip-001
```

Replace `podman` with `docker` when appropriate. For multiple GPUs add the global
option `--render-node /dev/dri/renderD128` with the intended node. To reuse an
image add `--image sha256:...` **built from this PR checkout**, rather than an
older image. The normal root Dockerfile retains its pinned cached dependency
layers; the host installs no native toolchain.

Share each output directory's `validation.json`, per-stage `receipt.json`,
`results.xml`, `stdout.log`, `stderr.log`, and the GPU `gpu/encoder-update.json`.
The GPU suite requires actual HIP, large-input finite/update evidence,
cached/full generation parity, and the integrated full-state resume fixture.
Unavailable hardware fails clearly; CPU execution is not substituted as GPU
proof. These commands only run bounded validation fixtures.

## Reviewed inputs and final plans

Assisted recipes intentionally contain null `track.foundation` and
`track.assisted_manifest` and fail until supplied. The workspace JSON packet has
exact keys `manifest` and `demonstrations`. `manifest` is the qualified
`assistance/v1` record. Each demonstration has exactly `id`, `source`, `indices`,
`values`, `verification`; `id` hashes source/indices/values. `verification` binds
source, current native/runtime/reference hashes, all100 exact indices/values and
`admitted: true`; its hash must match the reviewed roster. Source/license/review,
full observed overlap exclusions/exposure groups, and compatible foundation
ancestry are mandatory. No demonstrations were fabricated for this PR.

The `final-evaluation/v1` JSON is the qualified FinalPlan fields plus an inline
`decision_manifest` object for the five named decisions. Each decision record
has status `completed` or `not_planned` and hashes to its corresponding
`decisions` entry. The plan binds cohort, seed/checkpoint/recipe/portfolio/runtime,
controls, denominator and aggregate/per-arm budgets. Its immutable full record
and the shared study ledger live under workspace `run-state/final-study`;
analysis bundles retain the plan and exposure state. Retries keep prior debits
and exposure. Ordinary train/learnability/diagnose finalize with final access
sealed. No final plan was authorized by this implementation run.

## Remaining gates

T080 is closed by the completed new-image CPU/native run reviewed below.
T092 has passing integrated HIP evidence below, but remains a later qualification
gate after Phase 4 retirement and source/image binding review. T081–T088 remain
Phase 4; `bigint-retirement` reports BLOCKED.
The Phase 5 measurement-only preflight harness, numerical/performance and
containment gates are still prerequisites for real screens. Frozen recipes use
`learning.cost_preflight: null` until matching measured p95 costs and non-scoring
fixture identity are supplied. No recipe silently reduces requested coverage.

Requirements checklist:16/16 checked. `.specify/extensions.yml` is absent, so
there are no after-implementation extension hooks to dispatch.

## PR #56 CI repair

Completed run [39252676619](https://github.com/grugnog/oeis-learn/actions/runs/39252676619)
at head `018d323b369f9123ada2eea80075635b1dba5f91` built successfully, passed
all 154 Phase2 CPU/native tests without skips, and passed the separate Rust,
Docker and Podman jobs. Two checks failed:

- The native selection/scoring fixture supplied hand-formatted source alongside
  canonical codec tokens. Exact EOS/source binding correctly rejected it. The
  production fixed-template path had the same integration mismatch. Both now
  obtain source from the codec's decoded tokens; the frozen template bank,
  mathematical ASTs, policies and source-binding rejection rule are unchanged.
- G7 rejected successful Cargo evidence because it assumed exactly three target
  summaries and that the second target was phase7. The new worker binary adds a
  zero-test target between library and phase7. Evidence is now checked by named
  owning targets, with the same minimum owning test counts; additional targets
  must also pass, with no ignored/filtered/failed or unowned summaries. Replaying
  the downloaded actual CI log recognizes all 13 passing Rust tests.

Regression tests cover all eight template AST/token/source identities, old and
new Cargo target layouts, additional passing targets, and missing, duplicate,
failed, ignored, filtered, unowned or underfilled owning evidence. They are
included in the CI native contract step and public bigint CPU suite. The rolling
spec007 disposition is rebound to the changed implementation without promoting
historical evidence. The previous local result JSON remains evidence for its
original source identity. T080 still awaits the corrected full native run;
new CI is deliberately left unmonitored.

Local CI-repair recheck:61 tests passed, one packaged-native skip; Ruff F/E9 and
`git diff --check` passed. No local Docker/Podman/Rust execution is claimed.

## HIP device binding repair (T068/T079)

The user-supplied `008-phase3-hip-001` log reports two passing actual-HIP tests
(large-input encoder update and decoder cache/position growth). The integrated
checkpoint/update fixture failed before its first update because the workflow
passes `cuda` while model parameters report `cuda:0`. The chunk encoder now
resolves an unindexed CUDA/HIP request through `torch.cuda.current_device()`
before comparing it with the concrete parameter device, matching the retained
`IntegerBytesEncoder` convention. Explicit wrong indices and CPU/GPU mismatches
remain errors. No architecture, numerical policy, checkpoint state or CPU
fallback changes are involved.

Nine CPU-host device-binding cases reproduce the original failure and check
current indices 0/1, explicit indices, omitted device and mismatch rejection
without allocating GPU tensors. The affected CPU encoder, ablation, decoding and
workflow set passes **36 tests, with two packaged-native skips and one GPU
deselection**, including exact checkpoint/resume versus uninterrupted weights.
Ruff F/E9 and `git diff --check` pass. This is not actual-HIP validation of the
repair. T080 and T092 remain pending; historical evidence keeps its original
source identity, and only the rolling disposition is rebound.

After pulling the repair on `implement/008-phase3`, rebuild from that checkout
and rerun with a fresh output directory (omit `--image` to avoid reusing the old
image):

```bash
git pull --ff-only
python3 scripts/foundation_run.py --engine podman validate \
  --suite bigint-gpu --output tmp-validation/008-phase3-hip-002
```

CI triggered by this repair is deliberately left unmonitored.

## Integrated HIP rerun — 2026-10-19

**PASS: 3 tests, zero failures, errors or skips.** The submitted JUnit report
includes the large-input encoder update, parameter decoder cache/position growth,
and previously failing `test_integrated_device_checkpoint_update[cuda]`.
The integrated fixture checks all seven checkpoint components, a real encoder
weight update, resumed update/debit state, and exact decoder-weight equality
with two uninterrupted updates. The encoder report confirms finite execution at
256/512/1024/4096-bit inputs and 4097-bit differences with changed projection
weights. The environment reports AMD Radeon Graphics (`gfx1151`),
PyTorch `2.9.1+rocm7.2.1.gitff65f5bc` and HIP `7.2.53211-e1a6bc5663`.

All seven submitted files are preserved byte-for-byte under
[phase-3-hip/](phase-3-hip/review.json). Both stage receipts exactly match the
outer validation record; all four submitted files referenced by the receipts
match their SHA-256 hashes. The two stderr files were not submitted; their
receipts report the empty-file hash. Image identity:
`sha256:41b763236ca5f5131ef8b8f07118f79fc4a22d6efbf5f55ec058fdcdfbcbd570`.
No source manifest or Git revision was attached, so this review does not claim
an independently verified source match to PR head `1fec53f`.

The five PyTorch warnings are advisory and did not fail or skip tests.
`qualified: false` is deliberately emitted by the validation wrapper even on
PASS: bounded fixtures do not grant full research/release qualification.
This evidence confirms the integrated HIP rerun passes; it does not replace
T080's CPU/native gate or T088's installed retirement audit. T092 remains
unchecked for its later phase and source/image binding review. No new CI run
is monitored as part of this evidence update.

## Phase 3 exit review — 2026-10-19

**T080 PASS; all 22 Phase 3 tasks are complete.** Completed
[container/native run39336707352](https://github.com/grugnog/oeis-learn/actions/runs/39336707352)
and [foundation review39336707335](https://github.com/grugnog/oeis-learn/actions/runs/39336707335)
both pass at PR head `cbd225400c1c2f7e52852fbb5c9efeeb1e80cf64`.
The artifact ZIP hashes match GitHub's reported digests. All 295 source hashes
reported by the CPU/native helper match that checkout; GitHub artifact metadata
also binds the run to that exact revision. The image reported by both portable
engine runs is
`sha256:dd29f59d63138f146b41fd2b3a432622013a44c7c516f7f2145a004a796d2082`.
The full reviewed counts, report hashes, native/lock identities, artifact IDs and
scope are retained in [phase-3-exit.json](phase-3-exit.json).

| Required boundary | Reviewed evidence |
| --- | --- |
| CPU/native algorithms and bindings | 163 tests; zero failures/errors/skips, including the nine device-binding regressions. |
| Phase 3 integration | 43 tests; zero failures/errors/skips, including the real packaged native fit/execute worker, full matched preparation, independently admitted selection/private scoring, CPU parameter update and exact resume, hidden-mount rejection, final-exposure/debit persistence and bounded private-free exports. |
| Locked native and retained Rust | 13 core and 13 retained tests pass; no failures, ignored or filtered cases. |
| Docker and Podman portability | Each passes 29 role/lifecycle tests and three public CLI tests; all these cases execute without skips. Public validation receipt contents and file hashes verify; repeat runs reuse only verified successful receipts. Podman build and installed native/G7 jobs also pass. |
| Actual HIP integration | User-supplied run above: three tests pass without skips, including integrated checkpoint/resume; original source-binding limitation is unchanged. |

The broader retained-profile public CPU regression report has 1107 tests and
nine explicit installed/legacy/GPU/container skips on each engine; this is not
described as a zero-skip run. The dedicated required bigint and Phase 3 reports
above execute every selected test. No new algorithm defect remains from the
reviewed integration failures. No code changes or further hardware run are
required to close T080.

This completion update changes documentation/evidence only. It does not perform
Phase 4 retirement, Phase 5 qualification, merge or research runs. CI triggered
by the documentation commit is left unmonitored as requested.
