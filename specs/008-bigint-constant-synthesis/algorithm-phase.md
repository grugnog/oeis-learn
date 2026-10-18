# Phase 2 Handoff — All Hard Algorithm Work

**Owner**: the stronger model selected by the user for implementation. This document does not choose or launch a model. All sub-work below belongs to **one implementation phase**, even if delivered in several commits/sessions. Do not put an additional algorithm phase into `/speckit.tasks`.

## Entry conditions

1. Read `spec.md`, `plan.md`, `research.md`, `data-model.md` and every `contracts/` document, plus repository AGENTS/constitution.
2. Phase 1 has working pinned container builds for standalone bigint WASM and native Z3, identified exact artifacts and locked source tuples, protocol/schema stubs and frozen known-counterexample fixtures.
3. Existing signed-256/old-model regression evidence is available as a compatibility baseline. New profile is disabled until exit gates pass; no Python-solver fallback is acceptable.
4. No training launch or user hardware access is assumed. CPU/native algorithm tests can be run in qualification containers; actual HIP commands/results must be handed back when unavailable.

## Work packages within this single phase

### A. Typed mathematical language and bigint runtime

Implement canonical exact integers, logical bigint/control typing, explicit parameters and complete codec/grammar; cache/uncached generation uses the same token semantics. Lower supported programs to the approved handle ABI, including source-index correctness and checked index/control conversion. Implement immutable arena, host-only literal table/status/serialization, exact add/sub/fullmul/cmp and resource accounting. Wire two modules in one store without exposing runtime memory or privileged exports to candidates. No instance reuse/refcount optimization in v1.

Implement an independently parsed Python reference for the new source language, with exact operations and explicit bounded execution. Expected arithmetic values must not come from the production lowering or runtime. Define correspondence and separate fuel-versus-reference-limit classifications. Preserve old checked-width behavior and explicit unsupported legacy translation outcomes.

### B. Dependence, abstraction and exact fitting

Implement semantic dependence/control-taint analysis, deterministic eligible literal abstraction, shared parameter IDs and final substitution. Build exact affine interpretation, elimination/back-substitution, original-equation checks and inconsistency evidence. Handle rank-deficient integer feasibility with QF_LIA; support bounded parameter-independent control unrolling and nonlinear arithmetic through QF_NIA. Encode checked old-profile intermediates where supported. Reject all unsupported semantics without probe-based guessing.

Implement aggregate bounds/deadlines, bounded assignment enumeration, ranking/uniqueness states and scoped UNSAT replay. Native results remain provisional until independent checking; a failed resource/verification attempt is not global UNSAT. Exact large model numeral conversion must never pass through i64/f64. Keep native solver contexts worker-local and externally killable.

### C. Large-number model inputs and generation correctness

Implement the variable-length shared chunk encoder exactly as the input contract, finite large log/difference/valuation math, padding/length/sign distinctions, device-correct positional buffers and four ablation boundaries. Preserve historical profiles. Verify parameter tokens, teacher-forcing alignment, constrained masks, EOS and cached/full generation on CPU and actual HIP when available. All neural architecture or mask/decoder algorithm decisions belong here, not the later integration phase.

### D. Generic data, split semantics and learning/search policies

Implement typed connected generic generation and parameter-family canonicalization, behavior/prefix/shift grouping, separate parameter/structure splits and concrete/parameterized paired admission. Preserve traps when shortening targets; do not introduce an unqualified optimizer. Implement frozen prompt derangements/unconditioned representation, template baseline construction, candidate ranking/alternative retention, full-denominator score selection, curriculum advancement/retained practice and replay admission/replacement rules. Implement family-bootstrap comparison statistics without treating three seeds as a large population.

This package supplies pure, tested policy functions for Phase 3 to call. If an existing implementation already satisfies a contract, qualify/reuse it rather than rewrite it; record the source identity and tests. Any new semantic policy needed by later integration returns here.

## Mandatory exit evidence

| Domain | Minimum counterexamples/gates |
| --- | --- |
| Exact integers | Zero/negative zero/canonicalization; signed adjacent values at256/512/1024/4096bits; 0!…99!; two-large-operand multiplication; boundary products/zero; large cancellation; exact source offsets |
| Containment | Forged/stale handle rejection; private memory/privileged export inaccessible; source/nodes/loop/arena/linear-memory/fuel/aggregate/deadline failures classified; healthy request after each failure |
| Independent reference | Independently derived expected values and parser; production/reference parity on valid cases; deliberately injected wrong production result rejected; different accounting limits never treated as agreement |
| Affine/SMT | Large negative/zero/shared coefficients; unique, inconsistent, nonintegral and rank-deficient systems; free-variable integer-feasibility counterexample; nonlinear false fit16vs4; intermediate overflow; parameter-tainted control; UNKNOWN/timeouts never UNSAT |
| Admission/evidence | Provisional SAT cannot enter replay; all visible indices required; malformed/mismatched evidence rejected; scope-limited UNSAT labelled; reference timeout prevents admission; no hidden fields accessible |
| Codec/decoder | Source/token round trips, parameter renaming, literal boundaries, EOS, cached/uncached and correct/shuffled/absent path parity; old vocab checkpoint mismatch explicit |
| Encoder | Width-independent parameter shapes; byte/chunk boundary neighbors; 4097-bit differences; masking/padding invariance; finite forward/backward; actual-device parity where available; ablations target documented fusion boundaries |
| Data/policies | Deterministic family/variant sampling; no detectable overlap across forbidden splits; non-affine tiny gate composition; semantic simplification counterexample; hidden-free selection; replay/assistance lineage and budget-preserving policy state |
| Comparisons | Fixed denominators under timeouts, eight-of128 gate arithmetic, paired seed/family statistics, archive alternative retention and omitted-evidence accounting |

Commit and record passing CPU/native suites, required fixture hashes, unresolved actual-GPU checks and effective profiles. A GPU-only failure does not justify silently changing determinism or numeric semantics; report commands/results and reopen the relevant Phase 2 package if algorithm changes are required.

## Exit boundary and return rule

Phase 2 is complete only when typed language, runtime, fitter, reference, encoder and decision-policy interfaces are stable and semantically qualified. Phase 3 may route these APIs and persist records; Phase 4 may migrate/remove callers; Phase 5 may run benchmarks/tests; Phase 6 may run declared recipes. They may not add a new numerical primitive, solver tier, dependence heuristic, encoder architecture, splitting rule, selection metric or mathematical simplifier without explicitly returning ownership to Phase 2.

Operational corrections (paths, permissions, framing, log errors, cache packaging) stay with their integration phase if they do not alter these algorithms. Keep the distinction visible in commits and handoff notes rather than asking a weaker model to improvise semantic repairs.
