# Phase 6 code review and remediation

Scope: spec 007 phase 6, T038–T049. Base: main `ea1633ed3b318e9da952f8e06356da76baae3768`
(merged phase 5). No phase 7 repairs are included. The accepted single root
Dockerfile/development-VM/`lab-gpu` approach is retained.

## Findings and changes

1. **Missing strict trainer.** Reusing the old SFT orchestration would admit
   named-family/legacy demonstrations, use a different loss weighting, and save
   insufficient state. The new strict orchestration validates the actual phase 5
   archive, passes only its immutable trainer view, and reuses the existing model
   classes and shared teacher-forcing operation. It uses per-program masked
   next-token loss including EOS, explicit AdamW settings, finite gradient/weight
   checks, actual changed-parameter checks and no device/teacher fallback.
2. **No complete checkpoint producer/recovery.** Added the phase 4 reader's
   eleven-key producer, flush/hash/immutable publication, newest-valid scan,
   quarantine, nonreused generations, two-generation/final retention and disk
   reserve checks. Evaluation does not retain optimizer/RNG tensors unnecessarily;
   the validated payload is returned only when continuation requests it.
   Continuation pins all package source/resources and run settings, including
   resource floors, so matching tensor shapes cannot conceal changed forward
   math or a weakened resume-time limit.
3. **No independent durable budget controller.** Added a one-writer event ledger,
   measured versus estimated charges, conservative crash recovery, clean pauses,
   cross-boot handling and nonrefundable model rollback. A spawned controller
   heartbeats outside learner calls; an outer watchdog can kill the controller's
   process group. The API supports all declared phases; combined pool/evaluation
   arm orchestration remains unfinished and is explicitly not claimed.
4. **Unbounded/ambiguous legacy telemetry.** Strict runs use bounded append-only
   event segments, idempotent event identities and conflicting-duplicate rejection,
   typed measurements, torn-write recovery and actual RSS/host/allocator metrics.
   Legacy telemetry remains historical. Run inspection is read-only and separates
   completed learning progress from retried/uncommitted update events and from
   qualification.
5. **Training CLI absent.** Implemented train/resume/inspect, strict input
   rejection, explicit CPU diagnostics, optional clean stopping and preparation,
   actual artifact paths and newest-valid resume without config overrides.
   Non-diagnostic promotion remains blocked instead of trusting hash-shaped pins
   or a finished lifecycle as provenance evidence.
6. **Resource boundary not equivalent to Docker isolation.** Added bounded disk,
   memory, hung learner/controller and allocator-failure tests, plus a local-image
   role launcher using the existing root image. Its actual Docker probe is marked
   and skipped when unavailable. The launcher does not implement coordinated
   multi-container transport/admission; T046 remains partial. The externally
   installed `lab-gpu` broker is not in this checkout and cannot be audited here.

## Evidence and limitations

See the adjacent phase-6 software report for executed counts and environment.
Tests use real typed generic-program admission and real model/AdamW updates;
small CPU shapes are explicitly diagnostic. Full-state CPU comparison includes
model, optimizer, inactive state, every active RNG, order/cursor/next examples,
manifest counters and deterministic charged time. Separate tests cover corrupt
newest rollback, each publication crash point, final pin retention, discarded
prefetch, duplicate events, clock failures, fake disk quota, actual spawned-worker
OOM replacement and external learner/controller deadlines.

The root Dockerfile and dependency lock do not change. No system dependency or
driver is installed. Actual HIP and Docker are unavailable in this environment.
Hardware tests remain skipped, and software test success does not satisfy G5.

Remaining phase 6 work is substantive and visible: coordinated role-container
transport/resource admission, real image/mount/device/containment evidence,
integrated bootstrap/evaluation arm accounting and corresponding metric adapters,
and a provenance-backed readiness promotion path. No qualifying long training is
enabled. These are not silently marked complete merely because diagnostics pass.
See [the operational guide](../../../docs/foundation.md) for commands and exact
boundaries. Later phase task statuses are unchanged.

The broad regression run passed **458 tests**, with **four explicit HIP/Docker
skips**. The final-source phase-6 run passed **25 tests**, with **two hardware
skips** (overlapping counts). The full configured 8,426,527-parameter CPU model
completed three real updates through a clean pause/resume on the verified
64-program pool. Its ledger charged 12.381899333 seconds of active run time; this
excludes pool preparation and clean pause and is not a GPU performance result.

Spec Kit prerequisites and the project validator passed (108 tasks, 52
requirements, 68 review/decision items, 100% trace coverage). Both checklists
remain read-only and complete; no extension hooks are configured. Changed-code
Ruff E4/E7/E9/F, offline lock consistency, wheel build/install/import outside the
checkout and diff whitespace checks passed. The historical report rewritten by
a legacy CLI test was restored and is not part of this PR.
