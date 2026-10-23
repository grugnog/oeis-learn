# Contract: bounded synthesis scheduling and deterministic selection v2

**Status**: Planned. Existing v1 artifacts retain their original meaning. Native arithmetic, solver shortlist algorithms, candidate ABI and independent reference are unchanged.

## Policy identities and comparison boundaries

Record separate hashes for numerical/runtime profile, fit policy, generation policy, selection policy, archive policy, scheduling policy and measurement policy. The run/checkpoint contract includes all effective policies. Do not change the signed-256 historical identity shape or claim a new numerical profile solely because scheduling changes.

The first decoder intervention uses existing `visible-only/v1`, current batch eight and current deadlines on both sides. Before parallel scheduling, compare serial v1 and v2 selection explicitly on the same complete visible candidate records and then on a bounded serial workload. Report every selected-source, provenance, archive and training-representative difference. Neither policy-only nor scheduler comparisons can use hidden continuation results to select the policy. Diagnostic continuation may be reported separately after sealing.

### Visible selection v2

For independently admitted candidates matching all twenty visible terms, select the lexicographically smallest tuple:

1. canonical final-source UTF-8 byte length;
2. canonical final-source SHA-256;
3. stable attempt ordinal within the registered prompt/arm;
4. stable logical job ID as the final duplicate/provenance tie-break.

Hash the source under the existing final-source canonicalization; do not invent normalization that changes the language. Measured cost is still reported, but cannot rank candidates or enter the semantic selection seal. Same-source duplicates use the stable earliest eligible attempt and its corresponding original/transformed source, assignment and verification. A separate evidence receipt hashes timing and other observations; the semantic seal binds source, prompt, snapshot, policies and selected provenance only. Full generation/fitting/scoring results remain separately inspectable.

`transfer_success` explicitly dispatches v1/v2 seal validation. A reader must not silently rebuild an old seal under v2. Unknown policy versions fail closed.

### Alternative retention v2

Preserve the existing structural grouping and cap of eight alternatives. Deduplicate equal final sources by stable eligible attempt identity, with verification/track/profile compatibility checks; conflicting evidence is an error. Within each structural group compute the nondominated frontier on:

- final-source byte length;
- sum of `max(1, abs(value).bit_length())` across the canonical integer assignment values, with an empty assignment charged zero.

Assignment dimension measures representation complexity, not runtime or uniqueness. Keep one representative per structure ordered by `(source_bytes, assignment_bits, source_hash, attempt_ordinal, job_id)`, then fill remaining capacity with other frontier entries using that order. Report dominated/capacity omissions. Select a training representative by the same deterministic key from retained entries. Runtime observations remain telemetry. Record `structure-pareto/v2`; preserve v1 reader behavior and legacy omission interpretation.

Use this rule at every relevant path: portfolio duplicate collection, `_seal`, archive retention, cross-control archive merging and training/replay representative selection. Rust fitting's `(nonzero coefficients, total coefficient bits, coefficient vector)` shortlist ranking and first-independently-admitted fit remain unchanged. An incomplete solver search can discover different assignments under different deadlines; never call that full semantic parity.

## Native slot supervisor

One owner contains the capacity ledger and cache manager. Factor the existing process, pipes, selector, stderr tail, generation and RSS watchdog into slot objects. Each slot executes one native request at a time using the existing wire protocol, maximum 32 candidates and 4 MiB frames. Allocate a monotonically unique integer `worker_generation` across all slot starts/restarts; keep slot ID in coordinator evidence, not an unqualified new wire field.

Initial policy: workers in `{1,2,4}`, up to N active native requests, eight pending requests / 32 MiB aggregate, each active request separately charged against a maximum 16 MiB active payload pool. A candidate/job larger than the existing frame/source cap is rejected before reservation. Result staging is globally capped at eight records / 32 MiB, with output-size reservations made before dispatch and released only after committed evidence or cancellation acknowledgement. These are role totals, not per-slot allowances.

Use stable whole-prompt affinity `prompt_ordinal mod N`, with FIFO admission inside that logical slot queue and a global capacity check. Do not work-steal initially: reproducible cache/dispatch behavior is more valuable than maximizing instantaneous occupancy. Empty slots may idle while another prompt has long work; measure the cost before proposing a new scheduling policy. Every comparison records queue residence, active/idle occupancy, restarts and result publication delay.

Use at most N bounded host task threads for blocking client/independent-reference calls. Do not share mutable per-request state or fork initialized Torch/HIP. Existing role caches have one owner and bounded reservations; access/publication is serialized by that owner. Completed tasks return immutable records; the coordinator owns the job ledger, candidate collection, sealing and all cross-arm publication.

The initial native per-slot RSS watchdog remains 1 GiB, with a conservative 4 GiB aggregate native RSS ceiling and the outer shared-memory limit resolved by `workstation90` (or an explicitly selected historical/capacity comparison profile). Report shared-page double counting in summed RSS. Candidate memory/fuel/integer/source limits remain unchanged. Torch uses two intraop threads and one interop thread; native solver/numerical threads remain one where configurable. Record actual thread/process counts and inherited environment. Do not multiply hidden thread pools by N.

## Generation, overlap and admission

One GPU owner runs the decoder. The initial overlap policy works within the existing prompt group and conditioning mode; it does not mix private truths, model snapshots or controls. Keep raw/rescue/equal-budget and conditioning/template semantics explicit. Raw/rescue use the same first greedy candidate, and per-arm evidence is never borrowed from another control.

Each prompt/arm has at most one active attempt; other prompts may decode while completed candidates undergo native fitting/admission. After an attempt's required visible stages terminate, its next seeded attempt becomes eligible. Process eligible prompts in fixed round-robin prompt order; a readiness decision and its time are recorded. The GPU may produce at most two batches of lookahead, also constrained by tighter native/result reservations. Reserve maximum candidate staging before launching a decoder batch, not after an unbounded executor queue has filled.

The baseline uses batch eight. Batch-one and batch-sixteen diagnostics are separately named generation/scheduling policies. Batch sixteen cannot silently override a current eight-row contract or training minibatch. Active rows retain stable prompt ID, row RNG, cache state and actual length through batching, EOS and reordering. Decode calls use the earliest applicable absolute deadline; expired rows are explicitly classified, not dropped to make a faster denominator.

### Deadline hierarchy

A registered arm/group gets an absolute monotonic deadline when admitted into the portfolio scheduler, before queue waiting. For equal-budget comparisons, the manifest fixes group membership/order, global scope allowance, group allocations and per-prompt ceiling before execution; groups do not recalculate a larger allowance from previous favorable completions. A prompt's deadline is the minimum of its group/arm/global deadline and its start plus the existing 60-second portfolio ceiling. All its generation, queue residence, native execution, fitting and visible reference work share that deadline. Worker count never multiplies it.

A visible raw/fit job retains the existing maximum four-second service allowance, including the native/admission split and reference ceiling of two seconds, clamped by its parent's remaining allowance. No full service allowance is minted again after queue residence. GPU calls cannot reset the parent clock. Record dispatch/expiry and all computed absolute deadlines; durations are telemetry, not identity inputs.

For fixed-completed-work comparisons only, use the separately frozen generous diagnostic deadlines in [comparisons.md](comparisons.md). Those receipts are not evidence of success within a 60-second research portfolio. Scoring occurs after selection under its separate fixed allowance; late/missing independent scoring is incomplete, never inherited from generation completeness.

The eligibility boundary is strict: a result is usable only if all required stages completed and were observed by the owner before the applicable deadline (`observed_at < deadline`). At the boundary or later, record `late_result` and exclude it from selection. Late results cannot reopen a sealed arm. Across schedules, changed completed sets are coverage evidence; only identical complete outcome sets are required to produce identical v2 selection/retention.

### Work and charging

Maintain both elapsed parent allowance and actual child CPU/work counters. Parallel child wall durations may overlap and therefore must not be summed as elapsed time. Native instruction/fuel/attempt limits still apply to each declared operation and the scope's aggregate work ledger. Reserve aggregate work before dispatch and debit actual consumption exactly once on completion/cancellation; release only unused reservations, never consumed allowance. A job cannot be resubmitted under a new ID to evade its attempt cap.

## Cache contract

One role cache owner starts with global existing limits: results 4,096 entries / 64 MiB, compiled metadata 32 entries / 64 MiB, disk 256 entries / 512 MiB. A separately declared phase4 capacity trial may increase the result cache to32,768 entries /512 MiB and disk cache to2,048 entries /4 GiB; it does not change arithmetic or admission, and must be compared independently under the same cold/prewarm conditions. Compiled metadata remains at32 entries /64 MiB until a real use is demonstrated. Do not describe metadata storage as a shared native compilation cache. Inflight reservations, temporary publication and duplicate keys count toward capacity.

Keys bind exact source/prompt, numerical and resource profiles, runtime/guest/engine policy, reference, fitter and fit policy/seed; new run scheduling/selection identities and comparison namespace are recorded where they affect provenance. Candidate raw/native results can be reused only within qualified semantic contracts and current deadlines. A fitting cache entry is complete only when its recorded search/shortlist disposition is complete under the declared policy; verified partial discoveries remain evidence, not a complete replayable fit response. Unknown/time-exhausted/truncated results cannot become complete cache hits.

Comparison arms use isolated namespaces. Cold mode disables reuse; warm mode executes the same manifest-listed prewarm jobs in the same logical order before timing. Do not share baseline-generated successful fits with the candidate. When a cache hit supplies an assignment, independent final-source admission still runs within the current deadline. Qualification explicitly covers cache hit/miss/eviction, worker restart and rejected stale/partial entries.

## Cancellation, failure and update barriers

On cancellation: stop GPU producer/admission; terminally classify queued jobs; signal all affected native process groups concurrently; kill/reap and collect bounded output; then release active reservations. Total role cleanup allowance is five seconds, not N times five. One slot crash retires that generation and classifies its job; it may restart for a later healthy job within the original parent/scope allowance. No transparent retry grants the failed attempt extra work. Aggregate resource breach cancels the role, not merely the largest visible worker.

Independent Python reference checks retain their cooperative instruction/deadline limits. Cancelling a future does not kill its thread. If a task fails to quiesce, fail the role and let the external supervisor terminate its container/process tree within the reserved deadline; never block indefinitely in executor shutdown. The next healthy-request recovery check uses a clean role if role-level containment was required. Fault injection must exercise this fallback.

Before an optimizer update, checkpoint save, selected-checkpoint change, mode switch or control-group transition, drain within the existing allowance or cancel remaining work and record every terminal disposition. A clean checkpoint contains no unaccounted active job and records policy hashes, completed logical work and cumulative budgets alongside the existing full optimizer/RNG/data/curriculum/replay/selection state. Reject resume under changed effective policies; a new policy starts a new experiment. Diagnostic weights-only clones remain nonresumable.
