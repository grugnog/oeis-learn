# Data Model and State Transitions

All records are versioned, canonicalized and content-hashed. Mathematical integers are decimal strings; bounded counts are validated integers. Hashes identify complete effective records, not filenames. Relationships below are planned and must be enforced by contract tests.

| Entity | Required fields / relationships | Validation |
| --- | --- | --- |
| ExecutionProfile | language/codec versions, operations, signed/index rules, runtime WASM hash, native/source/toolchain/lock identities, resource-profile ID | No unknown operations; old-profile records immutable; resource changes alter identity |
| ResourceProfile | input/literal/intermediate/output bits/bytes, arena/memory, fuel/reference steps, control/node/source limits, deadlines/cleanup | Positive finite bounds with sensible nesting; aggregate budgets independent of per-term limits |
| SynthesisServiceLimits | in-flight/pending batch counts and bytes, result/compiled/disk cache counts and bytes, accounting units, FIFO/backpressure/LRU policy | Positive enabled limits; queue wait uses parent deadline; aggregate per-role reservations precede allocation/publication; cancelled/failed work releases reservations without refunding elapsed time |
| LogicalProgram | canonical source, typed AST hash, codec, literals, parameter IDs, original source, abstraction map, structural prior version | Complete source/EOS, types/locals valid, no unresolved parameter reaches execution, no fabricated shared IDs |
| VisiblePrompt | exact indices and values, source snapshot ID, visible-count policy | Strict ordering and explicit missing-index rules; no hidden continuation field in fitter schema |
| FitRequest/Result | protocol/request/worker generation, program/prompt/profile IDs, remaining aggregate budget; assignment/outcome/scope/ambiguity/evidence | Provisional native results cannot be admitted; bounds/coverage/phase required for failures |
| VerificationRecord | final-source hash, production and independent-reference identities, exact indices/results, consumed limits and comparison | Every visible term checked for admission; full horizon checked for continuation scoring; disagreement is a defect |
| ExampleFamily | generator revision, structure key, parameter assignment, concrete/parameterized source, exact outputs, stage, origin | Stage/key reproducible; equal values do not imply shared parameters; simplification cannot drop traps |
| SplitManifest | behavior/prefix/shift group IDs, family IDs, train/parameter-dev/structure-dev/OEIS-dev/final membership, snapshot and overlap audit | No detectable behavior group crosses incompatible splits; family overlap allowed only in the declared parameter-dev experiment |
| AssistedManifest | source/license/provenance, demonstration IDs, verification IDs, permitted recipients, overlap exclusions | Explicitly reviewed before use; no strict recipient; known exposed final data cannot become a fresh held-out test |
| ExperimentContract | data/split/model/input/decoder/solver identities, track, shared foundation, seeds, budgets, selection/stop/curriculum/replay rules | Freeze before execution; cross-profile/recipe changes create new experiments; unknown settings rejected |
| FinalEvaluationPlan | version, cohort/decision-manifest hashes, immutable arm/seed/checkpoint roster, recipe/portfolio/runtime identities, denominators, controls, per-arm/aggregate budgets, shared ledger | Decisions complete before authorization; only registered matching arms run; first final-prompt evaluation durably exposes cohort; identical-plan resume cannot refund allowance or freshness |
| Checkpoint | full weights/optimizer/RNG/data-order/selection/curriculum/replay state, budgets, contract ID and ancestry | Resume is exact-contract only; weights-only artifact is analysis, not resumable state |
| CandidateArchiveRecord | raw/final sources, assignments, source/track IDs, verification, costs, behavioral/structural identity, discovery labels | Retain alternatives within frozen policy; omission decisions logged; finite match never implies novelty/proof |
| Report/BundleManifest | mandatory compact summary, selected checkpoint, split denominators/coverage, paired comparisons, artifacts/hashes, omissions | Core records reserved first; no final/private payload leaks into training-visible bundle |
| RetirementDisposition | old path/public interface, consumers, supported/migrated/retired status, replacement/removal rationale, regression fixtures | Every old solver caller accounted for; no executable Python fitting fallback remains installed |

## State machines

**Candidate**: decoded -> parsed/typed -> parameterized (optional) -> native provisional assignment -> final-source valid -> production visible fit -> independent visible fit -> admitted. Any failure transitions to a classified terminal result for that attempt. Only admitted candidates enter visible-only ranking. Hidden scoring is a separate evaluator transition after candidates are sealed; it cannot revise fitting or selection. A continuation mismatch changes the score, not the earlier claim of visible fit.

**Fit scope**: unstarted -> active bounded scope -> SAT/UNSAT/UNKNOWN/exhausted. A smaller UNSAT scope may advance to the next predeclared bound only with remaining budget. The final record lists every completed scope. Verification rejection/limit is not a domain-wide UNSAT transition.

**Run**: prepared -> validated -> active -> completed/interrupted/cancelled -> resumed only if the identical contract is eligible. Consumed budget is durably debited before work and reconciled conservatively after interruption; missing completion evidence cannot refund time. Finalized runs cannot resume training. New profile/recipe creates a new linked run identity rather than editing the old ledger.

**Track**: random generic initialization -> generic foundation -> generic-only control OR assisted descendant. Assistance is monotone provenance: an assisted checkpoint/replay record and all its descendants can never become strict. Shared foundation cost and ancestry are reported once plus per-branch incremental cost.

**Final cohort**: sealed -> authorized by a complete immutable final-evaluation plan -> exposed before first final-prompt evaluation. Study execution separately records active/completed/interrupted/cancelled arms and the shared ledger. Ordinary training completion cannot authorize this transition. Interruption never restores sealed/fresh status; only the identical plan may resume. Decision/checkpoint changes after exposure require a new cohort for a fresh final claim.

**Discovery record**: finite match, conjecture, bounded verification, proved statement and novelty review are separate fields with evidence links, not an automatic promotion ladder. A proof may establish a property of a known algorithm without establishing novelty; a novelty review may identify an interesting unproved conjecture.

## Identity and isolation details

- Production code identity includes all new crate sources/manifests/locks, compiler flags, emitted runtime artifact and native binding/solver versions. Replace the current hard-coded five-file native source inventory with a deterministic manifest of the supported source closure.
- Keep plaintext visible prompts/fitting evidence only in roles permitted to see them. Final continuations stay in scoring-private mounts. No solver process receives source metadata/family labels to choose a scaffold.
- Structural keys normalize parameter identifiers and local names without pretending to prove semantic equivalence. Generic family metadata is retained; identical first-100 behavior, duplicate prefix and detectable shift relations define separate overlap groups. Report undetected equivalence as a limitation.
- Archive at most eight alternatives per sequence initially, choosing representatives across structure keys and non-dominated measured cost/size, with deterministic ties and an omission ledger. Training may select one shortest verified representative; it does not delete retained alternatives. These retention algorithms are implemented in Phase 2, not invented in reporting.
- Compact summary maxima are enforced by schema: aggregate reports never contain an unbounded copy of every candidate. Detailed rows are chunked artifacts. Bundle's 16 MiB minimum reserves 8 MiB for bounded core records/manifest before traces; failure to fit required records is an explicit export error.
