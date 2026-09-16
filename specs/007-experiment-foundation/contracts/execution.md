# Execution, language and resource contract

Normative for FR-001–006, FR-013, FR-018–019. Implement in `src/oeis_learn/sandbox/`; this document describes future interfaces, not existing guarantees. IDs in this contract are frozen defaults for 007; changes require a new profile identity and new run.

## Language: `wat_i256_checked_v1`

- Single function `compute(n: i32) -> (i64, i64, i64, i64)`, evaluated at rebased indices 0–99. Results are little-endian two's-complement limbs. Decode each limb modulo `2^64`, combine, then subtract `2^256` if bit 255 is set. Wrong arity/types are rejected, never converted to zero.
- Fixed, task-independent wrapper declares 32 i64 locals `$v0`–`$v31` and 8 i32 locals `$c0`–`$c7`, initialized to zero. `$n` is the only input. No user imports, calls, memories, globals, tables, start functions, recursion, floats or host I/O. Only the trusted lowering stage may add helper functions and an overflow-status global/export.
- Model predicts the body only; wrapper bytes/hash and body/full-module token counts are recorded separately. This scaffolding has no sequence-family information.
- Allowed native operators are exactly the baseline `INSTRUCTION_TOKENS` at commit `aa2c8c4579dd74fdd8114ec9456960a98fcf500e`, excluding `i64.const_?` and the pseudo-token `result_i64_x4`. Concretely: local get/set/tee; i32/i64 constants, add/sub/mul, div_s/div_u/rem_s/rem_u, and/or/xor/shl/shr_s/shr_u, eq/ne/eqz/lt_s/gt_s/le_s/ge_s; i32.wrap_i64 and i64.extend_i32_s/u; drop/nop/unreachable/return/br/br_if; block/loop/if/then/else/end; and the five macros below. No new opcode is admitted implicitly by upgrading a dependency. The implementation stores this inventory in the hashed profile.
- Structured block nesting at most 8, including at most 3 nested loops. Canonical branch operands are numeric depths in active scope. Flat instruction syntax inside explicit structured parentheses; canonical serializer removes comments and whitespace differences, without deleting/reordering instructions. Folded input may be parsed for diagnostics but must serialize to this form before admission.
- Every native i32/i64 opcode follows WebAssembly bit-vector semantics, including wrapping arithmetic, shift masking, truncation-toward-zero signed division and defined traps. This is deliberately distinct from checked logical wide macros. Arbitrary native limb manipulations are bit-vector programs; they are not claimed to be checked 256-bit integer algorithms.

| Macro | Exact semantics and stack effect |
| --- | --- |
| `i256.const C` | Signed decimal C in `[-2^255, 2^255-1]`; pushes four low-to-high limbs. Each emitted i64 literal is a valid signed 64-bit spelling of its bit pattern. |
| `i256.zero` | Pushes four zero limbs. |
| `i256.add` | Pops two four-limb values A,B; computes A+B exactly; pushes result or reports `numeric_limit`. |
| `i256.sub` | Pops A,B in that order; computes A-B exactly; pushes result or reports `numeric_limit`. |
| `i256.mul_scalar` | Pops four-limb A then signed i64 scalar B; computes A×B exactly, including B=`-2^63`; pushes result or reports `numeric_limit`. |

Helpers check logical signed overflow, not individual unsigned carries. Multiplication must retain sufficient high-product/sign information to prove the final signed range; division of a wrapped product is not an acceptable overflow test. An injected private status global is set immediately before a checked-helper overflow trap. Guest source cannot access it or spoof the reserved namespace; traps are classified from this marker, not a text search in runtime error messages.

## Codec: `wat_body_decimal_v1`

Dedicated tokens encode syntax, allowed instructions and fixed local names. A numeric immediate is `<int>`, optional `-`, one or more decimal digit tokens, then `</int>`; canonical zero is `0`, with no leading zeros, plus sign or negative zero. Decimal magnitude is checked against the instruction/branch range. The grammar masker handles partially emitted literals, type stack, local types and branch scope. No arbitrary integer vocabulary enumeration and no `<unk>` substitution are permitted. BOS/EOS/padding are separate; EOS is legal only when the complete body satisfies its return signature. Maximum 1,024 predicted body tokens including EOS, excluding BOS and wrapper; source at most 64 KiB. Hitting the cap with no valid EOS rejects the sample rather than truncating it. Vocabulary order and all special-token IDs are hashed into the codec identity; legacy output weights are incompatible.

## Shared interfaces

`prepare(source, profile) -> PreparedCandidate | Failure` parses and validates the allowed source, applies only explicitly enabled transformations, revalidates the final source, canonicalizes it losslessly, lowers it, and compiles it. The foundation/v1 smoke disables constant solvers/optimization. US5 repairs their owning modules and enables only tested extension profiles. Regex optimization, family scaffolds and unqualified proof hooks remain prohibited. Test injections may supply a proposed grounded/transformed source to demonstrate that final checking rejects it.

`execute(prepared, indices, limits) -> ExecutionEvidence` runs terms in ascending requested order, starting a fresh store/instance for every term. Compiled code/engine may be cached, but mutable instance state must not be reused. A runtime-neutral adapter exposes prepare/execute/identity; The baseline implements Python/Wasmtime; US5 repairs and qualifies the Rust adapter with the same executable corpus. Rust remains unavailable until that mandatory parity gate passes. Backend auto-selection is prohibited.

`verify(evidence, expected_terms, scope) -> CandidateResult` requires complete exact outputs, every required evidence identity and resource compliance. Admission compares production outputs with the independent reference; benchmark verification compares to exact source truth and independent execution. A disagreement is an internal validation failure and cannot produce a match. Prefix verification runs only indices 0–19; full verification runs fresh indices 0–99 after candidate selection is frozen. Reusing prefix results to conceal state-reset differences is prohibited.

The independent evaluator uses Python integers plus separately written native-width bit-vector rules and structured control flow over the parsed AST. It may share the parser, but must not call production lowering, helpers, limb operations or Wasmtime for expected arithmetic. It is bounded by its own AST-step and wall limits; inability to finish produces an explicit reference limit, never acceptance. Parser independence is not claimed. Conformance tests compare known hand-calculated vectors as well as the two execution engines to reduce common-parser blind spots.

## Outcomes

| `outcome` | Meaning |
| --- | --- |
| `invalid_syntax` / `invalid_type` | Parse, static profile/type or return-signature failure. |
| `unsupported` | Well-formed feature outside the frozen language/runtime profile. |
| `numeric_limit` | Exact literal or logical macro result outside its numerical range. |
| `execution_limit` | Fuel, reference steps, wall time, memory, token or aggregate limit reached; `reason` identifies the limit and `stage` its origin. |
| `runtime_failure` | Trap such as division by zero/unreachable, worker crash, runtime/internal disagreement; reason and available diagnostic evidence required. |
| `wrong_values` | Complete in-budget execution disagrees with available exact truth. |
| `incomplete_output` | Execution reports success but outputs are missing; missing benchmark truth instead invalidates the benchmark/run before scoring. |
| `prefix_match` | All 20 visible values agree; no statement about hidden values. |
| `full_horizon_match` | All 100 values agree with exact truth and independent execution within all limits. |

Stop at the first failure in term order. Preserve outputs/cost up to failure and its index. A prefix match later failing is retained as prefix evidence but the full verification outcome is the failure. `proof_status` is always `not_claimed` in CandidateResult; `PROVEN` is not in that finite-result schema. Separate certificate records in the repair contract carry scoped checked claims. Result delivery is idempotent by attempt/stage identity. A duplicate candidate still consumes its generation attempt; cached exact execution cannot add a second success.

## Resources: `ryzen_foundation_v1`

| Limit | Default and enforcement |
| --- | --- |
| Wasmtime fuel | 1,000,000 per term; 50,000,000 aggregate per 100-term candidate. Sum actual consumed fuel; per-call resets cannot reset aggregate accounting. |
| Independent reference | 1,000,000 AST instructions per term; 50,000,000 aggregate. Report separately from Wasmtime fuel; units are not comparable. |
| Candidate deadline | 2 seconds for prepare + production execution + reference verification, excluding queue wait but including cache misses. A stage may inherit a shorter remaining target deadline. Worker kill/replacement acknowledgment within another 2 seconds. |
| Program state | No guest linear memory. Worker address-space/container RSS limit 1 GiB; compilation memory is included. |
| Concurrency | One learner, at most 8 execution workers, one native execution thread each; learner CPU threads 4. |
| Queue / cache | At most 32 pending requests, 1 MiB per request/result; bounded LRU compiled-module cache 1 GiB total across workers. Cache key includes final source, profile, helper, runtime, target architecture and engine flags. |
| Host memory | Learner/controller Docker memory/swap limits both 88 GiB plus at most eight one-GiB worker containers (96 GiB aggregate declared cap); GPU allocation budget at most 64 GiB and 80% of reported device memory, whichever smaller. Monitor host available memory separately: stop new work if below 16 GiB for two one-second samples. Shared APU allocations and cgroups may overlap; do not add their metrics as disjoint memory. |
| Disk / files | Run artifacts at most 200 GiB and stop before host free disk drops below 100 GiB. Checkpoints at most 40 GiB, retain two latest valid plus one pinned final; reserve twice the largest checkpoint size before writing. Logs 2 GiB, rotate 64 MiB segments without deleting required evidence. |
| Lifecycle | No network/GPU devices for candidate workers, read-only runtime/source, bounded output mount, finite PID count (64 per worker, 512 for learner/controller). Docker default seccomp, no privileged mode or host IPC. |

Output-limit violations stop safely rather than discarding provenance to keep training running. The controller's target budget includes queueing and watchdog overhead; parallel execution does not multiply it. Foundation default evaluation permits 16 attempts, a 20-second generation/prefix-selection phase and a 10-second hidden-verification phase per target; later experiment profiles may change these before freezing. Hidden verification proceeds in frozen attempt order, prioritizing the already selected candidate, and charges all independent checks. A final metric therefore means success **within these budgets**, not exhaustive correctness or sequence unsynthesizability.
