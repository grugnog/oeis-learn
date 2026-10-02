# Phase 5 generic pool interface

Implemented by T030–T037. This is an offline CPU bootstrap, not a training run
or a language-performance comparison. It reuses the complete foundation codec,
supervised Wasmtime workers and independent interpreter from phases 2–3.

## Frozen sampler

`configs/foundation/generic_sampler.yaml` is the canonical small fixture:
seed 20260913, 64 published output classes, at most 10,000 samples and 300 seconds,
four execution workers, and fixed eight-sample commit chunks. Smaller explicit
test budgets/counts are accepted; increasing these caps requires a new profile.
Unknown keys and altered `generic_sampler_v1` priors fail.

The generator implements the statement, constant, register, expression and
control-depth priors in plan slice D. Additional choices needed to make the
sampler fully reproducible are disclosed here and in its config/code:

- Draw total statement count uniformly in 1–32, counting nested statements.
- At each statement, choose assignment/conditional/counted-loop with weights
  70/15/15, removing unavailable choices and renormalizing. A conditional needs
  two nonempty arms; a counted loop needs one nonempty body. Allocate its child
  statement budget uniformly over the available positive sizes, then divide a
  conditional's budget uniformly into two nonempty arms.
- Before expression depth four, choose leaf versus operator with probability
  one half; at depth four use a leaf. Leaf categories local/index/constant are
  uniform. Value-producing operators are uniform within the explicitly listed
  type-compatible inventory. Statement/control operators are supplied by the
  fixed skeleton, not sampled as value-producing expressions.
- Signed constant draws use -16..16 with probability 7/8; otherwise choose a
  width uniformly among 8/16/32/64/128/256 that fits the operand type, then draw
  uniformly over that signed width's full interval. A zero wide constant uses
  `i256.zero`; other wide constants use `i256.const`.
- Counted-loop bounds choose constant versus `min(n,100)` uniformly; constants
  are uniform in 0–32. Each nesting level reserves a fresh counter/bound pair
  among the eight i32 temporaries. Nested statements cannot overwrite another
  loop's counter. Initial zero registers and the uniform final register choice
  are deliberate generic priors, not a claim of bias-free program sampling.

The generator takes **only** frozen config and sample counter. Its named RNG
identity is `python-mt19937/counter-sha256/v1`. Each sample uses its own seed from
the first 64 bits of the canonical digest of RNG identity, generator source
revision, sampling-config digest, base seed and counter. The sampling projection
excludes worker count, pool target, attempt cap, wall budget and all cohort
identities. Execution settings therefore cannot change the proposal stream.
The full build config and actual Python/runtime identity are separately pinned.
The RNG has no cross-sample mutable state to lose on a crash.

Samples preserve source, complete tokens or a rejection reason, provenance,
statement/leaf/constant counts, sampled expression operators and emitted opcode
coverage. Source/codec caps reject samples; nothing truncates or repairs them.
Changing the generator implementation changes its content-addressed revision.

## Admission and artifact graph

The controller resolves only the cohort manifest and reserved-prefix membership
file for admission. Private truth files are not needed or passed to workers.
The service receives an exact set of prefix digests covering every member of
development/final groups, including nonrepresentatives. Collision responses
contain a reason and membership identity, never a matched target or continuation.

Admission requires `origin=generic_sample` and regenerates the exact sample
from its claimed counter/config/revision. Merely attaching generic provenance
to an imported program is insufficient. It then checks lossless canonical
source/tokens, exact100 production/reference agreement, resource evidence and
reserved-prefix exclusion. Independent value disagreements stop publication;
ordinary syntax, numeric, runtime or resource failures are archived rejections.
If one engine reaches a resource limit earlier, a different output **length**
is a limit, not an arithmetic disagreement; unequal shared-prefix values still
block the gate.

The immutable `program_id` hashes exactly the nine fields specified by the
data model. It does not contain its own outputs/evidence/decision references.
The acyclic artifact graph is:

1. Program identity core.
2. Separate full synthetic outputs, production and reference evidence linked
   to that identity.
3. Program record containing source/tokens, the first twenty outputs and the
   three evidence references.
4. Separate admission decision with status, reason and membership result.
5. Attempt journal, chunk checkpoint and finally the pool manifest.

Every reference carries a relative confined path and final-byte hash. Loading
rechecks provenance, all output/evidence identities, independent agreement,
conditioning and token round trips. Missing/stale/cyclic or extra record fields
fail; no legacy dataset generation is invoked.

## Deterministic construction and recovery

Chunks contain at most eight ascending sample counters regardless of worker
count. Results may finish in any order; decisions publish in counter order.
Finish the bounded chunk before deciding whether the target was reached, so
completion speed cannot change the stopping prefix. All sampled attempts are
counted and archived, including rejected and duplicate candidates.

Exact duplicate sources are rejected without re-execution. Output equivalence
uses the complete exact100 array. Within each output class retain the shortest
body, then smallest source digest (then earliest counter for an identical tie).
Choose the first requested output classes by their first occurrence and publish
their chosen representatives in counter order. The final chunk may contain
surplus classes; these remain archived and never enlarge the trainer view.
Report accepted attempts, source/output duplicates, unique classes, rejections,
published count and the published pool's constant fraction separately.

Each batch's issued counter interval is durably recorded before execution.
Each immutable chunk references that issuance, its attempt records, previous
chunk digest, RNG identity and digests of the reconstructed source/output dedup
state. Recovery replays verified chunks and resumes the same counter stream.
Completed worker requests and partial attempt journals are idempotently reused.
The persisted monotonic/boot/UTC start charges crash gaps; backward clocks fail,
and exhausted time cannot create new workers or samples. There is no clean
pause/refund API for bootstrap. General arm heartbeat accounting remains T042.
If time expires with an interrupted chunk, recover its existing decisions
without execution and report the remaining issued attempts as pending. The
summary distinguishes issued/attempted, decided, admitted, rejected and pending
counts; it cannot conceal work by reporting only fully committed chunks.

Manifest publication is the commit point. A crash after that point resumes the
same manifest/report without resampling. Shortfall is a terminal diagnostic
report with no pool manifest, not permission to inject teachers or retune priors.
A new configuration/output is a new build. Full Docker filesystem/device/network
containment and host quotas remain T046; this phase does not claim those gates.

## Controller and learner APIs

`load_pool(root, expected_pool_id=...)` runs in the controller/admission role.
It verifies the complete referenced journal and archive, recomputes dedup/order
and summary, and returns only approved trainer examples and their identities.
It does not run generated code again during loading.

The learner receives only `trainer/` plus the controller-approved `view_id`:
`load_trainer_view(root, expected_view_id=...)`. This directory contains a pinned
manifest and examples with exactly program ID, codec ID, twenty IntegerText
conditioning values and complete body tokens. It contains no full outputs,
evidence, OEIS IDs, metadata, cohort paths or membership set. The expected view
identity must come from the verified controller result, not an untrusted file
inside the view. The pool references the trainer view, avoiding a circular
pool/view identity. Training integration and updates remain phase 6.

## CLI and reproduction

```bash
python -m oeis_learn.cli.main foundation freeze-cohort \
  --source tests/fixtures/foundation/bootstrap/source \
  --config tests/fixtures/foundation/bootstrap/cohort.yaml \
  --output /workspace/phase5-cohort --json
python -m oeis_learn.cli.main foundation build-pool \
  --config configs/foundation/generic_sampler.yaml \
  --cohort /workspace/phase5-cohort --output /workspace/phase5-pool --json
```

Use new output paths; retrying the same build resumes or validates its immutable
result. Exit 0 means a complete pool; 2 invalid input; 4 correctness/provenance
failure or attempt-limited yield shortfall; 5 resource/infrastructure interruption
or exhausted wall time. No GPU or new system dependency is required. Reports
are diagnostic and unqualified; they do not claim trained accuracy or proofs.
