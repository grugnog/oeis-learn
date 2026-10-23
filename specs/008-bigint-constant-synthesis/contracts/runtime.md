# Contract: Bigint Language, Runtime and Inputs v1

Status: planned interface, not currently implemented. Owned by implementation Phase 2. Initial qualification defaults below must be versioned; tuning them creates a new resource identity, never a silent experiment change.

## Profiles and exact values

- Keep the current signed-256 language/codec/runtime identifiers and validators unchanged.
- New identities: `bigint-body/v1`, `bigint-codec/v1`, `bigint-wasm/v1`, `bigint-resources/v1`, `hybrid-chunks/v1`. Each name resolves to a canonical manifest hash, not just a label.
- Authoritative JSON mathematical integers are canonical decimal strings matching `0|-?[1-9][0-9]*`. No negative zero, leading plus/zeros, float, exponent or JSON-number coercion. Counts and bounded metadata use range-checked JSON integers. Reject overlong strings before parsing; do not globally disable parser safeguards without bounded replacements.
- Native/WASM transfer uses a separate sign and minimal little-endian unsigned magnitude bytes; zero is sign=0 and one zero byte. Normalization/rejection is deterministic and the final canonical bytes are hashed.
- Every sample carries actual signed source indices. `compute` consumes the actual index, never an implicitly rebased position. The first 20 values are visible; 100 total indices is the default horizon. Missing source values are not fabricated.

## Logical language and codec

Bigint locals, exact integer literals, reusable parameter identifiers, immutable mathematical input `$index`, bounded control locals and booleans are distinct logical types. The initial operator set is bigint add/subtract/multiply/compare, control comparison/checked increment, typed select/if and structured bounded loops. Division/remainder/bitwise operations, arbitrary memory, recursion, indirect calls and unvalidated arbitrary WAT are absent. Legacy candidates needing absent operations are explicitly unsupported in this profile.

A bounded index-to-control conversion is explicit: nonnegative value fitting the declared control bound succeeds; otherwise execution fails with `control_range` or `loop_limit`, never truncates/wraps or silently caps the mathematical loop. A program may explicitly define different behavior for negative indices through supported comparisons/branches. Per-term execution is stateless; recurrence algorithms compute their result from the index using carried loop state inside that term. Existing stateful streaming profiles are retained separately, not silently reinterpreted.

Use first-occurrence canonical parameter IDs `p0` through `p7`, a typed `param.get` operation, and explicit literal tokens. Equal literal values do not imply shared parameters: generator metadata or a recorded abstraction map determines identity. Parameter-free concrete programs execute; unresolved parameter programs only enter fitting. Codec uses explicit BOS/PAD/EOS and complete literal delimiters; no UNK or completion repair. Freeze source-to-token round trips, parameter renaming, grammar masking, teacher-forced alignment and cached/uncached parity tests. A new vocabulary never loads an old decoder checkpoint by resizing its embedding silently.

Initial source limit 64 KiB, typed AST 4,096 nodes, nesting 32, 16 bigint locals, 8 control locals, 1,024 loop iterations per loop and 65,536 aggregate loop iterations per term. Generated target limit 1,024 tokens; final substituted sources use source/AST/literal limits rather than pretending a large solved constant was generated within that token budget. Literal/parameterized comparisons admit only the shared dataset subset satisfying both arms' token rules and report exclusions.

## Candidate ABI and privileged boundary

Candidate export: `compute(index_handle: i32) -> i32`. Handle zero is invalid; other handles are checked arena slots. Logical typing prevents passing arbitrary control values as handles even though both lower to i32.

Only these runtime functions are candidate imports under `oeis.big.v1`:

| Function | Parameters/result | Meaning |
| --- | --- | --- |
| `from_i64` | i64 -> handle | Exact small-literal/control conversion where typing permits |
| `constant` | bounded literal ordinal -> handle | Read a preinstalled immutable literal, including large solved coefficients |
| `add`, `sub`, `mul` | handle, handle -> handle | Exact mathematical operation subject to intermediate and allocation limits |
| `cmp` | handle, handle -> i32 | Exactly -1, 0 or 1 |
| `to_control` | handle -> i32 | Checked nonnegative control conversion with the profile bound |

Host-only initialization, literal/input loading, status and serialization exports are never entered in the candidate linker. Runtime memory is host-readable for bounded transfer/status, but never imported by the candidate. Validate function types, modules, exports, imports and forbidden memory/table/global access; link each approved function individually. Do not use a full-export registration shortcut.

Runtime has private immutable values in an append-only arena. No guest free, reset, pointer API, ownership transfer or handle reuse. Compiled modules may be cached; stores/instances are fresh per term and dropped after any result or failure. The store has exactly two instances and one capped memory. Runtime tables/features, if emitted, are audited and capped; candidate tables are forbidden. Reuse/refcounting is not an integration optimization: it would reopen Phase 2.

All runtime arithmetic, initialization and WASM serialization calls share one fuel allocation; no hidden refills. A native parser/compiler is still bounded by size, process memory and deadline. Preflight allocation checks must not reject valid borderline products solely because a loose maximum-bit estimate exceeds the exact result; bounded temporary storage and exact post-operation checking determine the outcome. Zero and cancellation are mandatory tests.

## Initial qualification limits

| Limit | Default |
| --- | --- |
| Visible integer / coefficient / literal magnitude | 4,096 bits (separate sign) |
| Intermediate/result magnitude | 16,384 bits |
| Input first-difference capacity | 4,097 magnitude bits for 4,096-bit signed inputs |
| Per-integer serialized output | 2,048 magnitude bytes plus sign; decimal transport separately byte-capped |
| Arena | 65,536 entries; 32 MiB allocated-value cap, accounting recorded |
| Runtime linear memory | 64 MiB including allocator metadata/temporaries |
| Fuel | 10,000,000 per term; 100,000,000 aggregate per 100-term evaluation |
| Independent reference | 1,000,000 AST steps/term; 50,000,000 aggregate, plus process deadline |
| Native worker total RSS cap | 1 GiB initially, including compiler/solver; parent role has its separate configured limit |
| Wall time | 2 seconds/term; 30 seconds/evaluation, compilation included; parent request may impose less |
| Cleanup | 5 seconds to kill/reap the worker/container, separately recorded, no extra solving |

These defaults are engineering hypotheses to test, not measured capacity. Near-limit counts use a frozen 90% threshold. Input/literal acceptance checks their own bit limits before execution; a much larger intermediate may still be permitted up to the separate result cap. No dynamic escalation except a predeclared profile policy charged to the same parent budget.

Benchmark labels distinguish supplied operand width from generated intermediate/result width. Under this production profile, 64/256/512/1024/4096-bit supplied values are legal subject to their declared role; 16,384-bit values are qualified as intermediates/results constructed from legal input/literal values through supported operations. Large intermediate operands may be reused by later operations only if the exact result stays within16,384 bits; overflowing cases are classified-failure measurements. Do not load a 16,384-bit input/literal or multiply two such values and call the expected rejection a successful arithmetic timing. Any direct wider-input microbenchmark requires a separately identified benchmark-only resource profile and cannot change training/evaluation defaults. Compare old/new backends only on their shared supported small-value semantics; wider cases report new-profile throughput or old-profile exclusion explicitly.

Failures: `invalid_source`, `unsupported_profile`, `invalid_handle`, `integer_limit`, `literal_limit`, `output_limit`, `arena_limit`, `memory_limit`, `term_fuel`, `aggregate_fuel`, `reference_limit`, `control_range`, `loop_limit`, `deadline`, `runtime_failure`. Runtime logical errors set a sticky bounded host-readable status before failure; limiter denial, fuel exhaustion and external kills supply separate evidence. Unknown traps stay `runtime_failure`, not guessed OOM. No numeric result survives a sticky failure. Failure of independent verification is not candidate success.

## Engine allocation policy and execution identity

The reviewed CPU growth comparison motivates `bigint-engine-policy/v2`:
64KiB spare virtual capacity for dynamic linear-memory growth, retaining zero
guards, pinned Wasmtime20.0.2, fresh stores, visible memory/arena caps, fuel and
failure semantics. This does not enlarge `bigint-resources/v1`'s visible64MiB
memory cap or change arithmetic/guest bytes. The immutable manifest in
`src/oeis_learn/synthesis/engine-policy.json` supplies both actual Rust settings
and Python/build metadata. The native build inspector compares the compiled
manifest against packaged data and rejects stale policy bytes.

Bigint execution identity is the canonical hash of binary hash, guest hash,
language profile and canonical `engine_policy_sha256`. Python admission and
native fitting use the same record. Retained signed256 keeps its earlier
three-field identity shape. Changed policy/build/runtime hashes invalidate
cached/prepared/checkpoint contracts; historical receipts remain historical,
not automatically qualified under this policy. Existing numerical profiles and
reference arithmetic identities remain distinct from this engine policy.

The only alternate constructor fixes zero growth reservation for qualification.
Engine configuration/comparison v2 labels64KiB as production and nests the full
zero control under `memory_growth.zero`; v1 archival readers preserve the old
zero-production/64KiB-control interpretation. No worker request or mutable size
flag selects an engine policy. Actual new-image native/container qualification
is required before declaring adoption qualified; T094's remaining slowdown work
and other Phase5 gates stay open.

## Width-extensible hybrid contract

Preserve the historical mathematical streams with the explicit bootstrap prime list `[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]` for modulo and valuation streams, valuation cap32, FiLM enabled, first/second differences, summary tokens disabled and d256 four-encoder/four-decoder backbone. Verify these constants against the frozen current bootstrap contract during implementation rather than constructor defaults. A new input profile replaces only the fixed byte projection; decoder gain/dropout settings are separately recorded.

For value and first difference, split canonical magnitude bytes into eight-byte little-endian chunks. Use shared per-byte embedding width 4, flatten each chunk to 32 dimensions, append true chunk length, project to d64, add sinusoidal chunk positions, apply one shared four-head d64/ff256 transformer layer, and masked mean pooling. Shared weights encode values and differences; their distinct pooled positions, signs and log byte lengths feed a projection to d256, added through the existing form of learned byte gate. Zero, padding and a partially filled final chunk must be distinguishable. Chunk position buffers must grow on the correct device/dtype without trainable width-dependent shapes. Byte padding must not affect pooled output.

Exact input bytes and differences never pass through float. Magnitude/difference log features use bit length and leading-bit approximations with finite documented error, avoiding whole-value float conversion and decimal parser limits. Residues are computed with exact integers; valuations have explicit finite caps/zero handling. Independent fixtures compare old-width mathematical features against the original control within declared numerical tolerance.

Four ablation groups are magnitude, modular, difference/valuation and exact bytes. Zero a group's contribution at its declared fusion boundary; do not silently renormalize other groups or change model width. Retrained variants disable the same boundary from initialization and report active/inactive parameters. Post-training sensitivity and retrained utility are distinct outputs. Variable-length serialization preserves exact source information, not neural injectivity.
