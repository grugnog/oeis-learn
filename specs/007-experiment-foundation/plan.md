# Implementation Plan: Trustworthy Experiment Foundation

**Branch**: `007-experiment-foundation` | **Date**: 2026-09-16 | **Spec**: [spec.md](spec.md)

**Input**: `specs/007-experiment-foundation/spec.md`. Baseline source: `aa2c8c4579dd74fdd8114ec9456960a98fcf500e`.

## Summary

Build a small, trustworthy path from generic program sampling to exact admission, supervised training, complete checkpoints and prefix-only model evaluation. Reuse the existing Python package, encoder/decoder, Wasmtime bindings and CLI. A single versioned execution contract replaces divergent acceptance paths; private evaluator data and immutable manifests prevent accidental answer leakage. Repair the owning implementations of all known non-LODA defects, retaining strict isolation until each repair is qualified. Add KV caching, bounded richer WAT profiles and program-derived discovery. Implement uncertain optimizations as controlled variants and qualify them through the bounded protocol, not an automatic sweep.

The next spec can add a native LODA adapter and run one practical paired comparison within the user's 1–3 days per arm. Only the LODA runtime, cross-language comparison and conditional transpiler are deferred. Non-LODA repairs and conditional measurements remain in this feature; no result or winning variant is assumed.

## Technical Context

**Language/Version**: Python 3.12 in the chosen Docker image; repository compatibility remains Python >=3.11. Bash host launch scripts use installed Docker/Python only. WAT helpers compile inside the container.

**Primary Dependencies**: Existing PyTorch, Wasmtime Python bindings, NumPy, PyYAML and pytest/jsonschema. Baseline GPU userspace is the versioned AMD image `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`, resolved to an immutable digest before use. Pin all resolved Python wheels, transitive versions and hashes in a generated lock during T001; retain the image's tested PyTorch rather than allowing pip to replace it. No unresolved digest is accepted in a qualifying run. The native parity job builds Rust/Rayon in Docker. Existing SymPy/Z3 and exact linear algebra support the repair slices. LODA is not a 007 dependency.

**Storage**: Content-addressed JSON/JSONL files, immutable program chunks and checkpoint blobs, one controller writer. Existing DuckDB may index source data, but manifests are the reproducibility boundary. No new database/service.

**Testing**: Existing pytest directories; independent Python-int arithmetic/reference execution, schema/negative contract cases, process-isolation integration tests, deterministic CPU resume fixture and explicitly hardware-marked GPU acceptance. Documentation validation is separate from these future runtime gates.

**Target Platform**: AMD Ryzen AI Max+ 395, 128 GB shared RAM, 2 TB SSD, NixOS host with Docker and Python. CPU-only diagnostics remain possible and unqualified for GPU readiness.

**Project Type**: Local ML/compiler/runtime library and CLI.

**Performance Goals**: Three actual GPU updates within ten minutes after image availability; bounded two-second candidate deadline with at most two seconds for worker reclamation; finite queues/storage and no hidden clock resets. Record throughput/yield and phase costs, without inventing a universal evaluations-per-second target before measurement.

**Constraints**: Twenty visible terms; 100 total; strict random initialization and generic data; checked logical signed-256 macros; exact native bit-vector semantics; initial FP32; at most eight execution workers and 96 GiB declared aggregate container memory. All budgets are enumerated in [execution.md](contracts/execution.md).

**Scale/Scope**: One learner and one WAT backend in the baseline, followed by explicitly qualified Python/Rust adapters; immutable small diagnostic pool then later experiment-sized pools. Corpus ingestion is streaming/indexed; grouping uses hashed windows and exact confirmation, not an all-pairs corpus scan. Hardware/data budgets can be changed only by making a new frozen profile/run.

## Constitution Check

**Pre-research gate**: The ratified 1.0.0 constitution conflicts with the requested machine, generic bootstrap and future native LODA. These conflicts are addressed in the separate, explicit [RFC](constitution-rfc.md) and proposed 2.0.0 constitution; they are not silently waived. Research and document planning are complete. Adoption and measured feasibility remain visible implementation gates, not claims made by Spec Kit validation.

**Post-design gate**: PASS for internal consistency against the proposed 2.0.0 rules; conditional on its adoption for implementation beyond standalone preflight. T001 gathers the old amendment procedure's hardware evidence; T002 records maintainer adoption. Neither is marked complete by this planning change. If adoption is declined, revise the design rather than weakening checks.

| Proposed principle | Design evidence / implementation gate |
| --- | --- |
| I: exact data and declared numerics | Decimal authoritative values, checked macros, independent oracle; G1/G2 below. Existing neural features explicitly may be lossy. |
| II: supported languages and acceptance | Complete codec/profile; final-source validation; qualified optional paths; G1/G3/G7. |
| III: bounded recovery | Per-call/aggregate fuel, independent watchdog, cache/queue/file caps; G1/G5. |
| IV: workstation reproducibility | Digest-pinned Docker, true GPU probe, complete checkpoint and ledger; G4/G5. |
| V: provenance and isolation | Generic sampler, admission archive, separate process views, frozen cohorts/candidates; G2/G3. |
| VI: scoped discovery | Finite CandidateResult permits only `not_claimed`; separate checked proof certificates require scoped evidence; G6/G7/G9. |

## Project Structure

### Documentation (this feature)

`specs/007-experiment-foundation/` contains `spec.md`, this `plan.md`, `research.md`, `data-model.md`, `quickstart.md`, `tasks.md`, `constitution-rfc.md`, `contracts/`, `checklists/`, `validation/` and `validate_artifacts.py`. The plan is complete; a separate unfinished plan-notes document is unnecessary.

### Source Code (repository root)

| Area | Existing files to adapt | Planned files |
| --- | --- | --- |
| Versioned contracts/config | `data/models.py` legacy adapters; `evaluation/protocol.py` | `experiments/{__init__,models,config,profiles,artifacts}.py`; `configs/foundation/{wat_smoke,wat_profile,generic_sampler}.yaml` |
| WAT acceptance | `sandbox/{lowering,preamble,runner,fallback_runner}.py`, `sandbox/preamble.wat` | `sandbox/{wat_ast,reference,pipeline,worker_pool}.py` |
| Codec/constrained decoding | `decoder/{wat_grammar,grammar_masker,environment_tracker,sampler}.py` | `decoder/program_codec.py` |
| Cohorts/evaluation | `data/benchmark.py`, `evaluation/{synthesis,checkpoint,readiness}.py`, `cli/evaluate_canaries.py` | `evaluation/{foundation_cohort,foundation_synthesis,finalization}.py` |
| Strict bootstrap/admission | `data/synthetic_generator.py` retained as legacy; no automatic use | `data/{generic_programs,program_admission,program_pool}.py` |
| Learning/resume/metrics | `rl/sft_trainer.py` reusable training operations; `tracking/run_manager.py` | `rl/foundation_sft.py`, `tracking/{training_checkpoint,budget_ledger,foundation_metrics}.py` |
| CLI/workstation | `cli/main.py`, `pyproject.toml` dependency extra; existing package | `cli/{foundation,foundation_preflight}.py`; `docker/foundation/{Dockerfile,preflight.yaml,compose.yaml,requirements.lock}`; `scripts/foundation/{build_image,run_container}.sh` |
| Tests/guidance | Existing `tests/unit`, `tests/contract`, `tests/integration`; existing arithmetic/lowering tests | `tests/fixtures/foundation/`, focused `test_foundation_*.py`, `docs/foundation.md` |

**Structure Decision**: Keep one package. New `experiments` types make the strict boundary explicit rather than teaching all legacy dataclasses a permissive mode. Reuse production operations through explicit interfaces; do not duplicate whole encoders/trainers/CLI implementations. New modules above are implementation targets, not files claimed to exist now.

## Phase 0: Research decisions

Resolved in [research.md](research.md): arithmetic/independent execution, final acceptance, complete integer codec, frozen 20+80 data, strict generic SFT, external checkpoint hashes, Docker compatibility and proof isolation. The expanded contracts below settle the repair scope and bounded experiment rules; no experiment winner is assumed. Hardware compatibility and performance are bounded **measurements to perform**, not missing architecture choices: a failed gate reports evidence and stops.

## Phase 1: Design and implementation slices

### A. Feasibility and explicit governance

Implement the standalone preflight first. It can use the existing model architecture and deterministic generic tensor fixtures without any OEIS solution or cohort. Pin and resolve the Docker environment; inspect existing NixOS/AMDGPU/KFD/render-node access; run three FP32 forward/backward/AdamW steps on actual encoder/decoder parameters at visible length 20 and body length 1,024, including zero, alternating signed-256 extremes and adjacent large-integer prefixes. Synchronize, verify finite loss/gradients, changed parameters, GPU tensor/device identity, elapsed time and peak allocator memory. Availability alone is not enough. Ten minutes bounds execution after image availability; downloading/building is separately recorded and does not imply permission for a training run. Record `hardware_not_ready` with specific diagnostics if the host cannot support the image. Never install a host compiler/driver or enable broad container privileges to turn failure into success.

T002 attaches this evidence and obtains adoption of the proposed constitution before the strict implementation is promoted. The prior laptop run does not satisfy this gate. This small preflight is not a third language-comparison arm.

### B. Shared types, profiles and exact runtime (US1)

Implement strict artifact/config validation and immutable identity first. Separate language/numeric/resource profiles from a checkpoint-specific evaluation protocol so candidate generation seeds cannot accidentally depend on hidden truth. Use exact final-source identities and a single prepare/execute/verify service. Adapt current `runner.py` and `fallback_runner.py` into explicit Python/Wasmtime implementation paths; disable auto-native fallback for strict runs. Repair negative scalar multiplication, full signed-256 literal lowering and four-result decoding. Trusted helpers expose an unspoofable overflow marker. The initial smoke disables these paths. US5 repairs solvers, typed optimization and the symbolic prover, then qualifies explicit variant profiles; named-family scaffolds remain prohibited.

Write a small supported-WAT parser/type checker and independent AST interpreter. Share syntax parsing only; arithmetic, native-width semantics and wide logical operators in the oracle use independent Python-int code. Persist production and reference evidence. Compile after macro lowering; current synthesis compiles too early. Reuse one engine per worker and a bounded compiled-module cache; fresh store/instance per index prevents hidden state. Count fuel per call and across the candidate. The controller enforces external wall/memory/request limits and reaps/replaces dead workers, including during compilation/reference execution.

Conformance comprises hand-calculated signed/unsigned boundary vectors, the known defects, 10,000 deterministic randomized arithmetic vectors (seed 7001), and 256 bounded generic structured programs across indices 0–99 (seed 7002). Test intentional wrong arity, traps, loops, worker kill, duplicate messages and aggregate exhaustion. Differential disagreement blocks admission; it is not voted away. US5 must repair the existing Rust adapter and execute the same parity corpus in Docker; unsupported profile features remain explicitly unsupported rather than silently falling back.

### C. Visible data, frozen cohorts and actual model evaluation (US2)

Implement source-index continuity/range census, union-find groups and deterministic selection exactly as [evaluation.md](contracts/evaluation.md). Hashed 80-term windows identify possible shifts up to 20; exact overlap confirms them. Persist verified component witness edges and counts without materializing every duplicate pair. Read private truth only in cohort/scoring processes; expose `VisiblePrompt` through a narrow schema and a generation process without the truth mount. Do not carry `BenchmarkTarget.family`, stage or ID through the model API.

Adapt existing actual-checkpoint synthesis concepts in `evaluation/synthesis.py`; do not route strict evaluation through `evaluate_canaries.py`. Reconstruct the model/codec from validated checkpoint identity and load actual weights. Module wrapper is task-independent. Seal proposal order and selected top-one before hidden access, using shortest body then source hash. Selection deliberately excludes execution time to avoid nondeterministic timing ties. Score all frozen groups, including unsuccessful/limited candidates; infrastructure incompleteness invalidates qualification instead of shrinking the denominator.

Freeze development/final splits before pool generation. Finalization writes a durable decision lock for checkpoint, protocol, selector and stopping record. Final evaluation then generates immutable per-target candidate seals using visible prompts only; scoring requires the seal, and retries reuse it. No hidden-guided retry. Tests use an instrumented model whose output depends on its parameters, so an implementation that silently ignores a checkpoint cannot pass by always running a canonical fixture. Full corpus size and useful synthesis accuracy are not acceptance prerequisites for this slice.

### D. Lossless programs and strict generic pool (US3)

Replace finite integer token enumeration for the strict path with `wat_body_decimal_v1`. Grammar masking understands digit tokens, bounded local identifiers, type stack, scope and EOS completeness; no uint128 vocabulary-mask assumption. Retain legacy codecs for inspecting old runs. Retokenize only newly admitted source; never reuse incompatible legacy vocabulary weights.

`generic_programs.py` samples a typed statement/expression AST, with no sequence-family branches or target metadata. Initial `generic_sampler_v1`: 1–32 statements; expression depth at most 4; at most 3 nested counted loops and 8 structured levels; 8 four-limb register groups backed by the 32 i64 locals and 8 i32 temporaries. At each statement choose assignment/conditional/counted-loop with probabilities 0.70/0.15/0.15, renormalizing when depth/size makes an option unavailable. Expression leaves choose same-typed local/index/constant uniformly when available; operators choose uniformly from type-compatible profile operations. Counted-loop bounds are sampled nonnegative constants 0–32 or `min(n, 100)`, with fresh counters; loop bodies are generic statements, not named recurrence templates. Integer constants choose uniform -16..16 with probability 7/8, otherwise a uniform signed bit-width from {8,16,32,64,128,256} restricted by operand type. Final output selects one wide register group uniformly. The fixed control skeleton and these priors are recorded; they are not claimed bias-free or complete over every allowed program.

Each sample is serialized, round-tripped, validated, independently executed for 100 terms, checked against reserved prefixes and deduplicated by source/output. Reject unsupported/overlength/limited/disagreeing cases rather than fabricating labels. Publish results in sample-counter order independent of worker completion. Freeze a pool before SFT; no online generator/replay state inside the foundation/v1 smoke. Pool construction checkpointing records counter, RNG identity, dedup set/chunks and decisions. A foundation fixture requests 64 unique admitted outputs within 10,000 attempts and five minutes; a shortfall is a diagnostic failure with yield statistics, not permission to inject familiar sequence teachers. Report constant-output fraction and coverage of sampled operators; no language-performance inference follows from this smoke fixture.

### E. Complete SFT and recovery (US4)

Reuse the current small tri-stream encoder and Transformer shape (d=256, four encoder/four decoder layers, four heads, feed-forward 1024) with a new codec-sized output layer. Start random, FP32, eager attention, dropout 0.1, FiLM enabled and summary tokens disabled. Freeze explicit modulus/prime lists and every remaining constructor default in the effective model profile; no implicit default can change on resume. Model inputs are twenty exact source values processed by the existing feature pipeline; record that the resulting representation may be lossy. This is the unchanged smoke control. US6 adds exact-byte conditioning and qualified cache/batching variants; the bounded decision protocol selects the experimental configuration.

Strict SFT consumes only the immutable trainer view; a missing pool fails. Initial objective is the mean of per-program mean next-token cross-entropies over predicted nonpadding tokens including EOS, followed by the batch mean; this avoids changing sample weight merely because a language spelling is longer. AdamW, learning rate 3e-4, weight decay 0.01, gradient norm cap 1.0; smoke uses batch four and three updates, with scheduler/scaler disabled explicitly. US6 repairs the legacy RL path and adds a separately versioned objective profile with complete active state. SFT remains the baseline; repaired RL need not win the conditional pilot to count as correctly implemented.

At completed updates, atomically persist all active states and an external final-byte hash manifest. Named RNG streams isolate initialization, data order and candidate sampling. The frozen pool/permutation/cursor makes asynchronous prefetch disposable; rerunning prefetched work must not change order. Maintain the independent heartbeat/budget ledger so a rollback cannot refund spent time; conservatively charge an unobserved crash gap through recovery and label the uncertainty. Deterministic CPU testing injects clock/crash boundaries and compares complete state, not just model weights. GPU continuation records determinism settings and environment; no cross-device/version bitwise guarantee is claimed.

Metrics are measured in the phase that owns them: generation attempts/yield, reference/runtime/compilation time, tokens, updates/examples, cache state, RSS and GPU allocator peaks. Logging is append-only bounded segments, not a growing JSON array rewritten each step. Full exact artifacts remain available under quota; exhausting a required-evidence quota stops the run.

### F. Integration and handoff

Expose the documented foundation CLI; legacy canaries explicitly say reference-runtime diagnostics. Retain the finite-result proof boundary and require the repaired original prover to pass its shifted-identity counterexample before architectural experiments. General proofs are restricted to explicitly supported certificates; UNKNOWN is a valid outcome outside those classes. Run the complete acceptance suite and the bounded workstation smoke, retain evidence and record remaining limitations. Only then can 008 use these profiles as an experimental foundation. Changes suggested by failed gates are focused fixes followed by the affected checks, not automatic new experiment arms.

## Validation gates and stop rules

| Gate | Required evidence | Blocks |
| --- | --- | --- |
| G0 | T001 real-device report plus explicit adoption record for RFC 007 | Strict implementation promotion and all long experiments |
| G1 | Zero unexplained conformance disagreements; final-source, arity, status, state-reset and single/batch parity tests | Training admission and evaluation qualification |
| G2 | Complete/range-correct groups, fixed denominator, hidden/metadata perturbation invariance, actual checkpoint dependency, finalization isolation | Model scores |
| G3 | Negative provenance corpus rejected; lossless boundary constants/locals; verified immutable generic pool | First strict update |
| G4 | Deterministic CPU state/cursor/budget equivalence; corrupt-newest rollback; no self-referential hashes | Resumable run claim |
| G5 | Real strict-model GPU updates, worker kill/recovery, quota stops and truthful resource evidence | Ryzen readiness / 008 |
| G6 | `PROVEN` and novelty labels cannot enter foundation artifacts | Archive/discovery handoff |

No gate substitutes a reference program, synthetic metric or CPU fallback. If source snapshots/hardware are unavailable, software tests may pass independently, but those runtime gates remain pending. No 24–72-hour language-comparison training is part of this plan. Non-LODA diagnostics and at most one targeted paired pilot per the experiment contract are bounded within 007.

## Complexity Tracking

| Departure from ratified 1.0.0 | Why needed | Simpler alternative rejected because |
| --- | --- | --- |
| Explicit proposed 2.0.0 governance | User-requested experiment contradicts old mandatory architecture | Silent exceptions leave future implementers with incompatible instructions. Adoption is a recorded gate. |
| Separate strict types/path with thin legacy adapters | Existing defaults import teachers, metadata and incomplete checkpoints | Adding a permissive flag to each call site makes information leaks and silent legacy admission easy. |
| Independent bounded reference execution | Production helper errors previously passed self-generated labels | Reusing Wasmtime/helpers as the only oracle reproduces the same defect. |
| Container/process boundaries | Candidate failure and evaluator truth require enforceable separation | Threads and omitted dictionary fields cannot contain native crashes or prevent accidental truth access. |

No additional microservice, universal compiler, database migration platform or unrestricted theorem prover is introduced. Richer profiles and certificate checkers have finite, enumerated scope.

## Expanded implementation slices — supersede the initial narrow scope

The user's latest instruction requires actual repairs, not permanent quarantine. The `foundation/v1` SFT smoke remains a fixed control. `qualified/v1` extension profiles are additional explicit modes in 007, with distinct codec/model/objective/runtime identities and mandatory qualification. Historical teacher data and unqualified proof labels never become allowed. Only LODA work moves to 008.

### G. Repair owning synthesis/proof implementations (US5, P0 correctness)

Implement [repairs.md](contracts/repairs.md): conservative typed parameter-dependence analysis, exact linear/integer algebra, genuine bounded nonlinear dispatch, structural recurrence recognition, measured solver outcomes and final verification. Replace both training/evaluation dispatch paths with the shared service. Repair local.tee-sensitive optimization using typed liveness/effects, and qualify Rust/Python arity/state/resource parity in Docker. Proof repair uses controlled symbols, domains/poles and real witnesses in both old public APIs. These repairs may start immediately after US1, in parallel with US2–US4; they must finish before any architectural experiment. A small polynomial identity certificate checker is sufficient to repair initial false promotion; US7 broadens supported certificate classes.

### H. Correct learning, faster generation and complete state (US6)

Implement [learning.md](contracts/learning.md) as explicit variants: prefill/decode self/cross KV caches, batched masks/buffers, exact signed-byte input, structural generic splits, real behavior-policy probabilities, complete frozen references, mode restoration and honest trace evidence. Repair all archive admission paths, CGI length handling, active replay and per-visit adaptive curriculum. Add bounded beam/repair search and verified round-based self-training using training-only continuations, with no development/final hidden feedback. Extend resumable state for every active replay/scheduler/reference/search component. Implement pluggable sinusoidal/RoPE/prefix-ancestor positions and one 25M-class model constructor; conditional comparison selects at most one challenger. Correctness is required even if the baseline wins.

### I. Broader WAT and generated-program discovery (US7)

Add a typed logical WAT IR without changing the baseline body codec silently. Separate checked-wide/array/streaming profiles declare exact arithmetic, memory, reset and cost semantics; Python reference and Rust adapters qualify each advertised profile. Unsupported arbitrary programs remain explicit failures. Archive multiple implementations and retain long verified source outside the trainer context. Derive program/dataflow features and bounded transform proposals, exact multi-index relation validation, class-specific certificates and post-seal novelty evidence. Typed macro mining and one bounded value-guided repair proposer use only permitted training archives; compiled expansion receives the same final checks. Certificate claims never conflate a mathematical identity with a machine theorem or an OEIS definition.

### J. Packaging, source policy and bounded decisions (US8)

Repair horizon-specific census and source metadata/eligibility; freeze the final cohort only after affine/shift grouping and structural split rules are installed. Build the wheel and native extension in Docker; clean-install CPU and actual native CI exercise resources and parity, rejecting unexpected skips. Consolidate historical launchers, extraction/census and qualification into shared services while preserving diagnostic provenance and exact configuration. Archive redundant wrappers only after parity fixtures pass.

Run the single consolidated engineering session and at most one bounded learning comparison in [experiments.md](contracts/experiments.md). All numerical thresholds there are frozen planning defaults, not claimed prior user approvals. The total cap is two engineering hours plus six training hours, not another set of multi-day arms. Make every not-triggered or inconclusive decision explicit. No script starts these experiments merely because document validation passes.

### Additional source targets

- Existing repairs: `decoder/{constant_solver,dixon_solver,qfnia_solver,mersenne61_filter}.py`, `sandbox/optimizer.py`, `crates/oeis_wasm_evaluator/src/sandbox.rs`, `discovery/{symbolic_prover,pipeline,vector_search,pslq_solver,numerical_validator,relation_identity}.py`, `data/symbolic_definitions.py`.
- New services: `decoder/grounding.py`, `experiments/{qualified_models,decisions}.py`, `encoder/integer_bytes.py`, `data/structural_splits.py`, `evaluation/{search,experiment_runner}.py`, `sandbox/{logical_ir,streaming}.py`, `rl/self_training.py`.
- Discovery: `discovery/{program_archive,program_features,transforms,certificates,certificate_checker,polynomial_certificate,recurrence_certificate,state_certificate,loop_certificate,macro_mining,novelty}.py`.
- Integration: `configs/foundation/{qualified,experiments}.yaml`, `scripts/foundation/test_wheel.sh`, `.github/workflows/foundation.yml`, `tests/{unit,integration,contract}/test_foundation_*.py`.

### Additional acceptance gates

| Gate | Required evidence | Blocks |
| --- | --- | --- |
| G7 | Owning solver/prover/optimizer counterexamples repaired; exact statuses and Rust/Python parity; no imported-label promotion | Architectural measurements and activation of repaired services |
| G8 | Cache equivalence, exact input round trips, analytic policy correctness, archive/replay/curriculum and complete active-state resume | Qualified learning/search modes |
| G9 | Wide/array/streaming independent parity; ranked relation negatives; replayable supported certificates and immutable novelty boundary | Rich profile and discovery claims |
| G10 | Clean-wheel CPU and real native CI; metadata/census/wrapper parity; bounded measurement decision records; complete review traceability | Full 007 completion and LODA handoff |

G0–G6 establish baseline readiness only. Full 007 requires G7–G10 as well; an implementation cannot mark 007 complete by leaving faulty extensions disabled. Conditional non-LODA experiment branches may finish with a justified not-triggered result, but mandatory interface implementations/tests and known repairs still must pass.
