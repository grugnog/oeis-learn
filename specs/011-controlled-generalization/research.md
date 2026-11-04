# Implementation research and decisions

**Date**: 2026-10-30. Baseline source: main `fba656c4147484e4a8383eab99a0a0599bf20771`; Spec 011 branch adds documents only. The [primary-source review](research-review.md) remains the scientific rationale. This file resolves implementation questions against code, not report prose.

## 1. Prospective profile rather than historical overrides

**Decision**: Add versioned generalization contracts and preserve the frozen pilot routes. New assessment prompts omit targets, family and protected stratum labels.

**Rationale**: `pipeline_validation/pilot_data.py` assumes historical row counts/OEIS rows and allows development sources through `learner_view`; its prompt publishing also includes family/strata. Those views are not the new sealed-assessment contract.

**Alternatives considered**: Renaming historical development as assessment would leak already exposed evidence; stretching the existing schema would silently alter old results.

## 2. Bounded prepared shards for refresh

**Decision**: Freeze structural descriptors/family roster and numeric sampling law, prepare immutable variant shards under admission budgets, then consume deterministically. Store cursor and pending batch/pre-sampling RNG in checkpoints.

**Rationale**: `data/bigint_programs.BigintSampler` currently derives `family, variant = divmod(counter,16)`. Incrementing its counter expands support rather than refreshing variants within fixed support. The historical coefficient law includes draws in [-16,16]; changing that law would be another factor.

**Alternatives considered**: Unbounded online generation complicates reproducibility and can leak new families; silently recycling exhausted shards changes the intervention. New shard generation is allowed only under the frozen descriptor law, exclusions and aggregate cap.

## 3. Deterministic coverage with finite isolation

**Decision**: Separate new quota-aware preparation from `pilot_selection.select`; reuse exact materialization and window/prefix screening. Coverage counters must measure executed branches/loops, not just syntax. Freeze quota tables only after bounded feasibility preparation, before learning or assessment.

**Rationale**: Current structural ordering favors stage A. A full cross-product of stage, size, family and magnitude is not guaranteed feasible. Multiple forms may agree on finite traces.

**Alternatives considered**: Universal semantic deduplication is unavailable; silently filling sparse cells with simpler rows destroys the declared benchmark.

## 4. Paired parameter targets, unchanged exact fitter

**Decision**: Use existing codecs and Rust `parameters.rs`, `dependence.rs`, `fit.rs`, `affine.rs`. Record explicit parameter-sharing policy and supported subset. Grounding round trips require independent verification.

**Rationale**: Automatic abstraction gives eligible literal occurrences distinct parameters. The generator's explicit parameter source may intentionally share parameters after its bounded parameter allocation. These are different hypothesis spaces and must be disclosed, not described as identical just because numeric constants match.

**Alternatives considered**: Blanket abstraction, a new continuous optimizer or integer solver is unnecessary and risks control semantics. Pilot numbers do not identify a solver defect requiring replacement.

## 5. Complete resume and pinned checkpoints

**Decision**: Base new learner integration on the state machinery in `fresh/bigint_worker.py` and `fresh/bigint_state.py`, extending versioned state for scheduler, shard cursor, pending batch, milestones and ledger. Pin milestone/best/latest/selected checkpoints explicitly before pruning.

**Rationale**: `followup_training.py` exports partial state but declares resume=False. Its saved best/latest is not a functioning resumable experiment. `CheckpointStore.prune` does not automatically retain every new milestone.

**Alternatives considered**: Flipping a resume flag loses state; recomputing milestones from logs cannot recover weights. Charge target tokens after replay substitution, including encoded EOS where present, excluding BOS/input/PAD, matching actual loss targets.

## 6. Versioned task-first replay

**Decision**: Extend `fresh/replay.SynthesisReplay` in a new profile; retain full training-oracle agreement from `bigint_worker.admit_replay`, add origin checkpoint and two-alternative/task cap, task-first sampling and deterministic eviction.

**Rationale**: Existing replay already has exact training-only admission and capacity 4,096, but samples globally across alternatives and refuses representation_comparison. It is not absent, nor universally qualified for new combinations.

**Alternatives considered**: Generic off-policy RL adds unsupported assumptions. Keep first replay comparison literal-only; combining parameter targets and replay is a separately registered interaction and must not bypass existing restrictions.

## 7. Qualification identity separate from study identity

**Decision**: Reuse original PASS producer receipts only, with a new sealed qualification-specific binding. Consumers retain full study identity independently and freshly inspect actual images/device/runtime before reuse.

**Rationale**: `pipeline_validation/runner.py` currently reuses host/CPU stages, not HIP. `adapters.py` binds CPU images/resources, while HIP lacks analogous device binding. Current bindings include whole invocation contracts; `diagnostics.py` and `followup_study.py` clear reuse. Shell flags cannot repair this boundary. `reuse.py` accepts original PASS receipts rather than blindly chaining REUSED receipts.

**Alternatives considered**: Globally relaxing experiment binding would mix cohorts/budgets. A skip flag or prior stage ordinal proves nothing. A bounded receipt-chain resolver is deferred; all consumers reference the original producer directly.

## 8. Conservative source closure and serial orchestration

**Decision**: Extend existing runner/receipts/bundle/recovery. Include new policy modules, adapters, helper and authoritative contracts in `experiments/source_snapshot.py` and the whole-package closure; keep prospective recipes independently hashed.

**Rationale**: Current source snapshot explicitly enumerates contract directories through earlier features and must include 011 authority. `closures.py` invalidates unmapped changes. The temporary helper is source and must be frozen before qualification. Existing `pilot_hip_serial.py --start-step` skips commands without proving qualification reuse.

**Alternatives considered**: Duplicating qualification in a helper produces a second acceptance implementation. Run canonical endpoints serially and let each validate its evidence.

## 9. Research progression and feasibility

**Decision**: Name E0 evaluation extension, P0 two-seed uniform replication, P1 fixed-support refresh comparison, then P2 structural support, P3 parameter targets, P4 replay. Use selection-only feedback for adaptive progression; freeze or renew assessment before any confirmatory claim. Keep all capabilities in implementation scope.

**Rationale**: The prior spec exposed too much as an undifferentiated experiment menu. A weak baseline remains informative; failed accounting/coverage cannot support a learning conclusion. No automatic balancing/decay reruns, full sweep or long-run extension.

**Alternatives considered**: Combining all changes would obscure their effects. Requiring improved accuracy to accept software confuses feature correctness with an uncertain research outcome.

## Resolved questions and execution limits

The design requires no new dependency or constitutional exception. Resource defaults and commands are specified in the contracts. Actual quota feasibility, throughput and hardware acceptance remain measured gates with explicit shortfall/failure behavior. They are not assumed successes or reasons to leave interface decisions unresolved.


## 10. Checkpoint throughput without losing recovery boundaries

Decision: reuse unchanged immutable tensors at the debit boundary and cache verified
blob identities only within the active writer, keyed by full inode/time/size facts.
Retain per-update persistence and cold full verification. Reserve terminal work
inside the scientific allowance plus a separate process-exit gap. Measure actual
paired training before choosing longer research limits.
Rationale: the audit proves8% of the target consumed the entire time allowance;
early saves dominate updates. Link deduplication alone still serializes tensors
and rehashes large pinned states repeatedly. Pending debits are not trained exposure.
Alternatives rejected: infrequent checkpoints or asynchronous snapshots introduce
rollback/tensor-race semantics; increasing time alone projects roughly75h/arm at
observed aggregate throughput; deleting old evidence or resetting clocks loses
provenance. A writer cache never replaces cold resume/export integrity validation.


### T157–T160 implementation review (2026-11-04 UTC)

Implemented the planned unchanged-tensor debit snapshots, bounded writer-local
verification cache, terminal publication/process-exit reserves, committed versus
charged token reporting, bounded timing windows and temporary paired probe.
Cold restore/export remains byte-verified; no update-boundary recovery point was
removed. Exact CPU continuation, pending debits, expired saved clocks, linkless
fallback, failed copies and same-size tampering were independently exercised.
The original timeout checkpoint is not resumed or reinterpreted by this work.

The source/CI implementation binding changed. Old HIP qualification and throughput
receipts cannot authorize the new source; the documented command rebuilds and
requalifies before running tens of updates with the original model and recipe.
Its 256-row shards are deliberately a runtime diagnostic, not the full data
performance or scientific comparison. Large-shard I/O, long-run variance and
actual HIP timing remain unmeasured; the forecast includes a twofold window cost
and 1,800-second allowance and can only propose whether six hours looks feasible.
Original receipt inventories and recipe/descriptor/exclusion bytes are verified
before using historical candidate provenance; new native preparation is mandatory.

Local evidence: 77 affected Phase B checks passed with one dependency skip; the
expanded probe/policy suite passed 29 checks. A broader local run passed 306 checks,
with three unrelated handoff imports blocked by absent wasmtime and one installed
model check skipped. All 68 review dispositions still bind to the implementation.
No HIP command or CI wait occurred. The next evidence is the current-source paired
training probe report, not another full research attempt.
