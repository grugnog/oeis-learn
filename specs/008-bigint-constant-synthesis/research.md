# Research: Bigint Execution and Native Constant Fitting

Date: 2026-10-18. Source baseline: main `f87dc8089d1a17aa1e01e83fd0e90f2137207d70`; spec commit `074d2e3f6edf2411f7e163239b8f3fd1fe41e94d`.

Research resolves design choices, not empirical qualification. No new dependency build, WASM execution, GPU run or performance benchmark was performed during planning. The version candidates below must pass the explicit Phase 1 build gate before implementation relies on them.

## R1 — Arithmetic placement and library

**Decision**: Use `dashu-int` alone behind a minimal wrapper compiled to `wasm32-unknown-unknown`, with default features disabled, panic abort and no WASI/JS imports. Pin the current researched candidate 0.6.2 only after testing it against the image's Rust 1.90.0 and recording the complete lock. Keep the boundary library-independent so a bounded `ibig` comparison can be made if build/throughput qualification reveals a concrete problem.

**Rationale**: Dashu supplies signed arbitrary integers in pure Rust; its optional defaults include facilities not needed by the guest. WASM execution accounts for arithmetic loops through Wasmtime fuel. Native host arithmetic callbacks would need a different accounting model.

**Alternatives considered**: Bespoke 1,024/4,096-bit arithmetic only moves the ceiling; native bigint callbacks escape instruction metering; `ibig` is credible but two production backends would add unnecessary qualification work before a measured need.

Sources: [Dashu types/features](https://docs.rs/dashu-int/latest/dashu_int/), [Rust target/alloc/features](https://doc.rust-lang.org/rustc/platform-support/wasm32-unknown-unknown.html).

## R2 — Store, handles and isolation

**Decision**: One compiled runtime module plus one generated candidate, instantiated in a fresh fuel-enabled store for every term. Only the runtime owns linear memory. Register candidate-approved functions individually, never all runtime exports. Immutable append-only checked handles; no candidate allocator/free/reset; discard the whole store after success or trap.

**Rationale**: Existing strict code rejects all imports and permits zero memories, one instance and four-i64 returns. It must remain unchanged. A new adapter permits two instances and one private runtime memory; runtime table features, if emitted by the pinned compiler, are inspected and capped. Store limits do not bound all compiler/native allocations, so external process/container memory and deadlines remain necessary.

**Alternatives considered**: Splicing compiled Rust into generated WAT is fragile. Sharing runtime state between terms or introducing refcounts needs extra ownership/alias semantics and is deferred unless measurement warrants reopening Phase 2. Stateful streaming is not silently migrated to per-term handles.

Sources: [Wasmtime linking](https://docs.wasmtime.dev/examples-linking.html), [resource limiter exclusions](https://docs.wasmtime.dev/api/wasmtime/trait.ResourceLimiter.html), [store limits](https://docs.wasmtime.dev/api/wasmtime/struct.StoreLimitsBuilder.html). Current docs describe the design pattern; concrete APIs must be checked against locked Wasmtime 20.0.2.

## R3 — Language and index contract

**Decision**: A typed logical body language owns bigint and bounded control types before lowering both to WASM scalar representations. Mathematical input is the actual signed source index, passed as an exact bigint handle. No implicit 0-based offset rewrite. Explicit index-to-control conversion rejects out-of-range values; bounded loops have explicit resource failure, not silent truncation.

**Rationale**: Native handles are i32 implementation values, not mathematical integers. Current `CompiledProgram.run_term` assumes indices in 0..99, while archive source offsets must remain exact. Typed lowering prevents forged handles and separates structural constants from mathematical parameters.

**Alternatives considered**: Exposing pointers or arbitrary WAT makes ownership and parameter abstraction unsafe. Reusing four-limb tokens for bigint would retain unnecessary source complexity. Existing checkpoints keep their original codec/profile and only supported concrete candidates can be translated explicitly for new-profile experiments.

## R4 — Exact fitter and native bindings

**Decision**: Use a native abstract interpreter for exact affine forms, deterministic fraction-free elimination and exact rational back-substitution. Route rank-deficient integer feasibility to bounded QF_LIA and supported nonlinear parameter arithmetic to bounded QF_NIA. Use Rust Z3 bindings; candidate `z3` 0.21.1 and its resolved `z3-sys`/vendored native source form one tested pinned tuple. Build in cached container layers; no floating system library or runtime downloads.

**Rationale**: A rational free-variable choice cannot decide integer feasibility. Numerical probing cannot establish affineness. Current files called `dixon_solver.py` use rational Gaussian elimination, so the replacement must name its actual algorithm. Rust Z3 exposes exact integer expressions, model queries, timeouts and context interruption; those are not substitutes for a killable worker deadline. Large assignments must use exact string conversion, not `as_i64`/`as_u64`.

**Alternatives considered**: A homegrown general Diophantine solver is unnecessary. cvc5 remains an alternative engine, but this design chooses the existing Z3 semantics/binding ecosystem and avoids maintaining multiple engines. Floating least squares and bounded brute force cannot justify exact global-unsatisfiability claims.

Sources: [Rust Z3 API](https://docs.rs/z3/latest/z3/), [solver outcomes/enumeration](https://docs.rs/z3/latest/z3/struct.Solver.html), [exact integer API](https://docs.rs/z3/latest/z3/ast/struct.Int.html), [binding build strategies](https://github.com/prove-rs/z3.rs/blob/master/z3/README.md), [binding/native compatibility](https://github.com/prove-rs/z3.rs/blob/master/README.md).

## R5 — Supported control, ambiguity and verification

**Decision**: Symbolically unroll bounded control only when guards/trip counts are parameter-independent at each supplied index. Track taint conservatively, retain every intermediate constraint for old checked-width semantics and stop at node/unroll limits. Reject parameter-dependent control; do not guess from samples. Enumerate at most eight assignments under one budget and rank discovered assignments by nonzero count, summed magnitude bit lengths and signed tuple. Check uniqueness only if budget remains, with explicit incomplete/unchecked status.

**Rationale**: An apparently correct structure may produce several fitting constants, hidden overflow or unsupported loops. SMT SAT creates a provisional assignment. Public verification requires final-source validation plus production execution and independently parsed Python reference agreement at every visible index. Python sees shortlisted assignments in a bounded batch, not every solver trial. A final resource failure is not evidence that all coefficient assignments are impossible.

**Alternatives considered**: Solver optimization without bounded completion, silently setting free variables to zero, and using withheld terms to select coefficients all undermine interpretation. SMT replay can support scoped UNSAT, but is not independent theorem certification.

## R6 — Width-extensible hybrid inputs

**Decision**: Preserve original tri-stream/hybrid profiles. Add a new hybrid input profile with shared learned encoding over variable-length signed-magnitude byte chunks, covering values and exact first differences. Use eight-byte chunks, a shared d64 one-layer four-head transformer with sinusoidal chunk positions and masked pooling, then project pooled value/difference features to d256. Include sign, true byte length and final-chunk length. Share weights across widths; constrain resource length in the profile. Exact layout, padding, pooling and feature math are Phase 2 algorithms.

**Rationale**: Current `hybrid.py` flattens 32 value bytes plus 33 difference bytes into a fixed projection, so merely relaxing integer checks breaks shapes. At 4,096 bits, 64 chunks (65 for the widest difference) are tractable qualification targets; longer widths need measured resource budgeting, not new parameter shapes. Source bytes remain exact, neural projections remain lossy. Use bit length and leading bits for log features rather than converting an entire huge integer to float or an unbounded decimal string.

**Alternatives considered**: A larger fixed projection repeats the same migration problem; unbounded byte-level attention is unnecessary. A recurrent byte encoder is possible but adds a different deterministic GPU kernel qualification. The chosen architecture is a testable candidate, not a proven superior representation.

## R7 — Curriculum, attribution and practical assistance

**Decision**: Generic connected structures produce parameter families; keep behavior groups together and publish separate familiar-structure/new-parameter and held-out-structure splits. Parameterize semantic literals with explicit IDs rather than merging all equal literals. Train a shared generic foundation, then compare generic-only and reviewed assisted continuations. Keep raw/fitted and conditioning/template baselines, and stage representation, replay, assistance and input ablations rather than a confounded full factorial.

**Rationale**: Recent matched runs established easier memorization for a joint decoder-gain/dropout candidate on n±k, not broad OEIS transfer or the value of every encoder input. A fixed-template solver could solve that diagnostic without a model. The full pipeline needs a diverse tiny gate followed by held-out structure evidence before another long run.

**Alternatives considered**: Pure generic-only training is retained as a control, not a creativity proxy. Importing historical demonstrations/weights without lineage would violate strict-track interpretation. GRPO and privileged scaffolds remain outside this feature.

## R8 — Retirement and evidence compatibility

**Decision**: Replace the old Python fitting surface completely. Port final checks before deletion: `grounding_reference.py` imports old parser helpers and `grounding_evidence.py` imports solver translation/Z3/recurrence code. Native scoped replay replaces supported solver replay; read-only old records retain historical status without re-executing removed algorithms. Keep independently useful proof/reference libraries only with documented active consumers.

**Rationale**: Merely replacing `constant_solver.py` leaves other Python solvers and fresh literal edits callable. Conversely, deleting files by age breaks independent validation. A disposition manifest and installed-image/import audit define completion.

**Alternatives considered**: Deprecated executable shims, renamed Python solvers and automatic fallbacks perpetuate ambiguity. Version-control history is sufficient for intentionally retired executable workflows.

## R9 — Dependency and performance qualification policy

**Decision**: Concrete design choices are resolved above; feasibility gates remain explicit tests, not unresolved product questions. Phase 1 smoke-builds the pinned candidate dependencies and records exact artifacts. Failed builds require a reviewed pin/toolchain revision, never an undocumented substitution. Phase 5 records latency/memory/throughput before long-run resource settings are frozen.

**Rationale**: Package documentation does not establish standalone WASM compatibility, container ABI compatibility, deterministic HIP operation or acceptable throughput. Native and WASM locks, emitted imports/features, compiler flags and artifact hashes must be part of run identity.
