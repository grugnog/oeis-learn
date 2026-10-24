# Research decisions: bounded pipeline throughput

**Date**: 2026-10-23  
**Baseline**: Spec 009 `fb22d653bdcb0d7c1ab9d78d07caace8fbe91c22`; production/evidence inherited from `ef240bf8e5ad04df93555fa43097565250d9eed5`.  
**Method**: Repository-source inspection and three focused research-agent reviews of inference, scheduling and validation. No hardware runs or dependency upgrades were performed. All design choices below are resolved; hardware sufficiency and gains remain validation questions, not missing specifications.

## R1 — Extend the public validator

**Decision**: Add `validate --suite pipeline` to the existing portable foundation CLI. A small static stage graph supports named phases and `all-ready`; one invocation prepares shared prerequisites, executes ready stages, reports progress and produces one archive. Research execution uses the same entry point with a separate explicit study manifest, never the validation scope.

**Rationale**: [foundation_local.py](../../scripts/foundation_local.py) already owns `Host.ensure`, engine/resource policy and command dispatch. [container_validation.py](../../src/oeis_learn/container_validation.py), [qualify_bigint_phase5.py](../../scripts/qualify_bigint_phase5.py) and [profile_bigint_pipeline.py](../../scripts/profile_bigint_pipeline.py) contain reusable qualification, audit, watchdog and packaging behavior. They currently abort on first failure and package separately. Factor these boundaries into reusable functions while retaining their old public adapters and historical readers.

**Alternatives considered**: A new command per experiment perpetuates the manual handoff problem. A general workflow service, database or distributed scheduler adds unnecessary state. Calling each complete existing script would duplicate image checks, measurements and nested archives. Use a sequential topological coordinator initially; concurrency belongs inside the measured pipeline, not between competing benchmark runs.

## R2 — Reuse evidence conservatively

**Decision**: Reuse only successful immutable stage receipts with matching dependency closures, artifact hashes and relevant hardware/resource conditions. Always inspect the current packaged source and binaries. Restart failed bounded stages in a fresh invocation; do not implement mid-test resume. Real learning resume remains the existing separate, qualified workflow.

**Rationale**: [source_snapshot.py](../../src/oeis_learn/experiments/source_snapshot.py) hashes broad code/build inputs; a new machine-readable workload manifest must also be bound explicitly. Reuse can avoid repeated long builds/tests without disguising old qualification as new. When closure completeness is uncertain, invalidate rather than guess.

**Alternatives considered**: Full-source-only reuse is safe but unnecessarily reruns unrelated CPU gates after decoder edits. Fine-grained automatic dependency inference is fragile. Use a short versioned explicit closure registry with negative invalidation tests and conservative fallback.

## R3 — Optimize cached inference before batching policy

**Decision**: First hoist token-step position, row, active-index, length and attention-mask bookkeeping out of the decoder layer loop, with bounded cache-owned scratch buffers. Keep batch eight, FP32, attention backend, weights, grammar, random streams and current deadlines fixed. Remove redundant greedy log-probability transfers only where the trajectory contract already returns no value. Preserve finite-logit checks. Batch sixteen is a later separately identified generation/scheduling comparison.

**Rationale**: [inference_cache.py](../../src/oeis_learn/decoder/inference_cache.py) repeats GPU boolean indexing and metadata creation per layer. [sampler.py](../../src/oeis_learn/decoder/sampler.py) still needs chosen tokens on the CPU for grammar tracking. The [reviewed trace](../008-bigint-constant-synthesis/validation/phase-5-pipeline-profile-rerun-review.json) records 224 `aten::nonzero` events and 7,168 kernels in eight steps, supporting a bounded bookkeeping intervention, not a proven unique cause. [bigint_generation.py](../../src/oeis_learn/decoder/bigint_generation.py) caps batch size at eight and clamps to the earliest request deadline; changing batch size changes scheduling too.

**Alternatives considered**: Attention backend upgrades, reduced precision, graph compilation, projection fusion, a GPU grammar engine and multiple GPU owners change too many factors. They are excluded from this initial plan. Finished-row compaction is deferred unless the fixed workload identifies material waste; metadata hoisting must already handle inactive/reordered rows correctly.

## R4 — Use one bounded native supervisor

**Decision**: Extend [worker_pool.py](../../src/oeis_learn/synthesis/worker_pool.py) from one slot to 1/2/4 supervised native slots under a single owning-role capacity ledger. Keep one active batch per process, existing framing, native solver and independent reference. Use bounded host job threads for blocking transport/reference work; the coordinator alone publishes shared evidence/cache mutations. Do not fork an initialized Torch process.

**Rationale**: The current pool explicitly rejects more than one inflight batch. [service_policy.py](../../src/oeis_learn/synthesis/service_policy.py) already supplies bounded FIFO reservations and caches. Creating N complete clients would multiply pending/cache allowances and conflict with exclusive disk-cache ownership. Reference admission currently executes bounded Python code synchronously; its small measured fraction makes separate reference processes premature, but thread/GIL contention must be measured. A stuck noncooperative host task is contained by the external role watchdog, not a claim that Python threads can be killed safely.

**Alternatives considered**: Native internal solver parallelism changes solver behavior; unbounded executors multiply memory. Eight/sixteen workers are excluded from the first comparison set. A later manifest may investigate more workers only with new aggregate limits and evidence that four workers still scale.

## R5 — Version timing-dependent selection before overlap

**Decision**: Add explicit `visible-only/v2` and `structure-pareto/v2` policies. Rank independently admitted final programs by canonical source byte length, source hash and stable attempt ordinal; measured wall cost stays telemetry. For alternative retention, use deterministic source-size and exact-integer assignment-complexity dimensions, then source hash/ordinal; preserve the existing structural grouping and count limits. Run a serial v1/v2 policy-only comparison first. All scheduler comparisons then use v2 on both sides. Preserve v1 readers, sealed selections and historical outcomes.

**Rationale**: [synthesis_policy.py](../../src/oeis_learn/fresh/synthesis_policy.py) currently ranks by source length, measured wall cost and hash. [candidate_archive.py](../../src/oeis_learn/fresh/candidate_archive.py) uses timing in Pareto retention and training representatives; [bigint_worker.py](../../src/oeis_learn/fresh/bigint_worker.py) merges duplicates by cost. Completion-order sorting alone cannot remove contention-dependent choices. Also, `Portfolio.begin` includes a monotonic start timestamp in IDs: add stable logical job IDs without rewriting old records.

**Alternatives considered**: Keeping v1 while asserting scheduler-independent equivalence is incorrect. Using “fastest finished” is a new biased selection policy. Retaining all alternatives is unbounded. Rust fitting's deterministic coefficient shortlist ranking stays unchanged; budget-limited discovered sets can still differ and must be reported.

## R6 — Bound overlap without asynchronous learning

**Decision**: One GPU owner drives visible generation while up to four CPU slots process completed candidates from the same frozen model snapshot. Publish outcomes in stable logical order; at most one active attempt per prompt/arm; schedule ready prompts fairly in a frozen round-robin order. The checkpoint/update boundary drains or cancels all work and records terminal dispositions before weights change. Share one sixteen-CPU role envelope with up to90% of measured physical RAM, as subsequently confirmed by the user; preserve16 GiB as a historical matched comparison profile; hidden scoring is a separate CPU-only role.

**Rationale**: Current [bigint_diagnostics.py](../../src/oeis_learn/fresh/bigint_diagnostics.py) yields decode requests but does raw/fitting work synchronously. A typed decode/native future boundary can expose overlap without new model algorithms. Queue residence consumes the same parent deadline. Deadline-limited completed sets may differ; fixed completed work and fixed budgets therefore require distinct reports.

**Alternatives considered**: Overlapping weight updates with candidate work creates stale-policy training semantics. Separate GPU processes contend for a shared device. Separate concurrent containers each receiving the full memory quota violate aggregate intent. These are excluded.

## R7 — Preserve cache semantics and evidence ownership

**Decision**: Use one owning-role cache manager, globally bounded reservations, immutable publication and per-comparison-arm namespaces. Cache identity includes numerical/runtime/solver/reference/fit-policy settings and new scheduling/selection contract where relevant. Do not persist unknown/resource-exhausted or truncated shortlist results as complete reusable fitting evidence; retain them as diagnostics. Cold comparisons disable reuse; warm comparisons use an identical predetermined prewarm stream.

**Rationale**: Current [client.py](../../src/oeis_learn/synthesis/client.py) can store a provisional shortlist when any assignment verifies, including a budget-truncated discovery set. Concurrent reuse needs explicit completeness/provenance, not only a candidate hash. A cache hit still requires independent acceptance under the current request's deadline.

**Alternatives considered**: Sharing warmed results between baseline/candidate arms creates an unfair comparison. Replicated disk owners risk races and hidden capacity multiplication. No new persistent compiled-code cache is required: current native caches are not a proven shared compilation service.

## R8 — Use conservative finite comparisons

**Decision**: Retain the proposed 10% material-gain and 5% per-stratum regression thresholds. Use one unscored warmup and five complete paired confirmation runs, with fixed alternating order, plus separate baseline-only calibration. Require all five paired primary gains to reach the 10% threshold and all mandatory stratum latency ratios to remain at or below 1.05; report median/range and partial outcomes. No adaptive extra samples. Primary metric is complete pipeline elapsed time; equal-budget verified throughput/coverage is mandatory supporting evidence, not an alternative selected after results.

**Rationale**: A conservative repeated-pair rule is easy to audit and avoids claiming precision from a tiny bootstrap. It may yield INCONCLUSIVE for noisy or modest changes; that is preferable to extra unregistered retries. Calibration fixes workload sizes and resource reservations before candidate timing; confirmation cannot discard slow rows or incomplete arms.

**Alternatives considered**: Utilization alone, aggregate partial-work ratios, arbitrary best-of timing and repeated testing until positive are unsuitable. Broad model retraining does not answer the present scheduling question.

## R9 — Keep profiling light and budgets explicit

**Decision**: Ordinary confirmation retains aggregate counters, external wall time and 1 Hz resource sampling; no per-token device synchronization or detailed byte/growth tracing. Short optional trace windows are separate, capped and excluded from headline timing. Measure telemetry overhead with the same complete work. Keep a lightweight synchronized instrumentation compatibility mode to compare the old path; changing instrumentation must be an explicit measurement-policy change.

**Rationale**: [PhaseTimer](../../src/oeis_learn/tracking/foundation_metrics.py) synchronizes device boundaries. Carrying every old synchronization into an overlapped path can destroy the intended overlap. [tracking/pipeline_profile.py](../../src/oeis_learn/tracking/pipeline_profile.py) and [pipeline_report.py](../../src/oeis_learn/fresh/pipeline_report.py) already distinguish intervals, resources, missing sensors and partial projections. The former “long” trace stopped near offset 73; add controlled cache replays through 513/1024 tokens, labelled microdiagnostics rather than successful generation.

**Alternatives considered**: Always-on detailed tracing can change the behavior under measurement. Removing all observability prevents attribution. Use host stage intervals, bounded device events where needed and interval unions, never add overlapping spans as if exclusive.

## R10 — Carry research forward without automatic admission

**Decision**: Preserve the complete [carryover register](carryover.md). Engineer and qualify the pipeline first, then use new prospective study identities and scope-specific cost admission for any learning attempt. Combine all already authorized validations; stop only at actual capability, failure, resource or research-decision boundaries. Defer T093 and preserve negative T096.

**Rationale**: The existing diagnostic shares are not whole-training costs, and the old transfer-screen forecast failed its admission ceiling. [The experiment contract](../008-bigint-constant-synthesis/contracts/experiments.md) already defines strict provenance, learning thresholds, matched comparisons and final-cohort exposure. We retain these rather than redesigning the scientific question.

**Alternatives considered**: Re-running T096 automatically, reducing denominators to make runs fit, or interpreting 32/32 diagnostic rescue as a corrected historic result would break the evidence chain. An optimization no-go can still produce a complete engineering investigation and a blocked research report.

## R11 — Use the confirmed workstation memory budget

**Decision**: The user confirmed128 GB RAM with90% available during planning. Default the new workstation policy to an aggregate ceiling of `floor(0.90 * physical_memory_bytes)`, further limited by effective engine/ancestor limits and checked against current availability. This is approximately115 GB on the stated nominal capacity; actual byte/GiB limits come from measurement. Reserve the remainder for the host. Do not sum CPU/GPU allocations as if they were independent physical pools.

**Rationale**: The previous16 GiB role limit was a diagnostic configuration, not a hardware constraint. It must not suppress useful batch/cache choices. Keep16 GiB only as an explicitly named historical comparison, and evaluate64 GiB and workstation90 capacity where representative workloads demand it. Both sides of a comparison use the same memory envelope; a capacity-plus-batch/cache change is separately identified.

**Alternatives considered**: Retaining16 GiB as the default despite confirmed capacity would leave useful options unexplored. Allocating115 GB simply to raise utilization would not help a small model. Use a bounded candidate set, measured memory/bandwidth/throughput and no swapping/OOM-based tuning; no automatic model enlargement follows.
