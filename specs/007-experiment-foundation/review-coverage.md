# Review and decision coverage — expanded 007

**Status:** Design coverage updated under the user's explicit instruction to fix all known issues; implementation and runtime evidence remain pending. Only LODA runtime/comparison/conditional transcoding are deferred. KV caching is included. A mandatory repair cannot be closed by disabling its subsystem; a speculative optimization may finish with an evidence-backed conditional decision.

This supersedes the narrower audit at commit `9be610f5f13410dfde1729426232b56293a861b9`. It maps all20 independent recommendations, A01–A19, P01–P17 and12 grouped decision-session obligations. Exact attachment hashes and machine-readable task references are in [review-inputs.json](validation/review-inputs.json). Source proposals are traced through their individually assessed entries in the independent review; no original probe or new training result is claimed here.

## Closure of the six audit gaps

| Gap | Required design correction now included |
| --- | --- |
| GAP-01 immediate proof repair | T051/T057 repair both original proof APIs; T061 gates experiments. Quarantine remains defense in depth, not the repair. |
| GAP-02 disabled faulty subsystems | T050–061 solvers/optimizer/native; T063/T070–077 RL/replay/curriculum; T081/T084–092 discovery. Each has original-path regression and activation evidence. |
| GAP-03 package/CI | T094 clean wheel outside checkout, packaged WAT, Docker native build and actual parity execution; T102 installed CLI integration. |
| GAP-04 dataset/census | T065/T069 structural and parameter holdouts; T093 all horizons, affine/shift grouping and finite/table/growth/source policy before real cohort freeze. |
| GAP-05 operational details | T073 long-program/Pareto archive and CGI eligibility; T095 wrapper parity; T096 real near-limit/cost telemetry and bounded tiers. |
| GAP-06 experiment protocols | T097–100 implement bounded non-LODA numerical/engineering/learning protocol. LODA-specific comparison alone remains008. |

Task IDs link conceptually to [tasks.md](tasks.md); their detailed contracts are [repairs](contracts/repairs.md), [learning](contracts/learning.md) and [experiments](contracts/experiments.md). Unsupported claims are corrected/rejected rather than turned into speculative requirements.

## Independent review

| ID | Topic | Disposition | Tasks | Concrete coverage / rationale |
| --- | --- | --- | --- | --- |
| R01 | Exact integer arithmetic | included | T009, T013, T014, T019, T082, T097 | Negative scalar correction, full literals, independent oracle, checked logical overflow and wider-intermediate diagnostic. |
| R02 | Genuine checkpoint qualification | included | T021, T023, T024, T025, T027, T029, T049, T095 | Actual weights, full length, selected top-one/any scores, original launcher qualification and warmstart repair; historical evidence stays diagnostic. |
| R03 | Constant grounding | included | T050, T054, T055, T056, T061 | Exact affine/nonlinear classification, modular pivots and rank deficiency, genuine QF_NIA/BV distinction, recurrence structure/cache, measured outcomes and final verification. |
| R04 | Common execution | included | T012, T015, T017, T052, T058, T059, T083 | Lower before compile, four results, pure/batch/native parity and typed optimizer local.tee/control/trap correctness. |
| R05 | Immediate symbolic proof soundness | included | T051, T053, T057, T061 | Repair both existing APIs, controlled symbol table, shifted domains/poles, scoped certificates and concrete disproof witnesses before experiments. |
| R06 | Data eligibility and isolation | included | T020, T022, T024, T065, T069, T093 | Exact indexed truth, finite/table/growth metadata, separate horizons, affine/shift grouping, synthetic structure/parameter/operator holdouts. |
| R07 | RL policy mathematics | included | T063, T070, T071, T077, T079 | Real behavior probabilities/support, frozen encoder+decoder, forced-action exclusion, module modes, entropy/variance and real trace evidence. |
| R08 | Reproducibility and infrastructure | included | T005, T038, T041, T042, T043, T077, T094, T095, T096 | Complete active checkpoints, truthful metrics/config, external hashes, clean wheel/resources, actual native CI and shared lifecycle. |
| R09 | Generation performance | included | T062, T066, T067, T099 | KV and cross-projection caches, batching/buffers, dynamic masks, invalidation and equivalence; actual end-to-end timings. |
| R10 | Typed representation and complete codec | included | T006, T007, T012, T073, T082 | Complete constants/locals, typed logical WAT, consistent EOS/wrapper/length, atomic elite/CGI eligibility and long-program archive. |
| R11 | Generic data and self-training | included | T030, T032, T033, T065, T069, T073, T076 | Diverse generic constructs, parameter/family isolation, multiple implementations/Pareto archive, near-miss training reuse and revalidation. |
| R12 | Bounded search | included | T056, T075, T076, T090, T100 | Sampling/beam/repair, skeleton cache, sound pruning, visible counterexamples and permitted retrieval; equal-budget selection. |
| R13 | Curriculum/replay defects | included | T064, T072, T073, T074, T077 | Per-visit observations, actual adaptive allocations, uncertainty/cost/exploration, nonzero replay gradients and complete state. |
| R14 | Runtime expressiveness and resource calibration | included | T080, T082, T083, T096 | Pure plus streaming/block and arrays, full wide ops, reset/state parity, calibrated bounded fuel tiers and near-limit counts. |
| R15 | Exact input representation | included | T065, T068, T078, T100 | Signed bytes and rebased indices, exact differences/residues, parity/powers-of-two, active heads and bounded prefix/encoder comparisons. |
| R16 | Ryzen configuration | included | T001, T040, T046, T047, T096, T099 | Actual GPU updates, Docker AMD identity/shared-memory accounting, synchronized sustained timing, finite AMP and bounded4/8-worker comparisons. |
| R17 | Generated-program discovery | included | T073, T084, T086, T090, T091 | Archive-derived graph/features/transforms, multiple implementations, checked typed rewrites/learned macros and post-seal novelty. |
| R18 | Relation-search defects | included | T081, T085, T086, T099 | Ranked bounded retrieval, actual configured multi-index exact nullspace, nonempty disjoint validation, honest metrics and negative controls. |
| R19 | Scoped proof capabilities | included | T053, T057, T087, T088, T089, T092 | Polynomial/rational, recurrence, finite-state, restricted-loop and bounded-machine certificates with domain/compiler-link limits and independent replay. |
| R20 | Position and capacity decisions | included | T062, T078, T098, T100 | Keep sinusoidal baseline, fixture-test RoPE/ancestor-path/25M-class options, select at most one triggered paired contrast; no future AST leakage or assumed superiority. |

## Bundled architecture proposals

| ID | Topic | Disposition | Tasks | Concrete coverage / rationale |
| --- | --- | --- | --- | --- |
| A01 | Sinusoidal corruption | not_adopted | — | Unsupported defect diagnosis; no such claim adopted. |
| A02 | Fixed positional-buffer crash | already_present | T062, T078 | Dynamic extension already exists; retain long-length regression instead of inventing a missing fix. |
| A03 | RoPE | included | T062, T078, T100 | Implemented selectable variant, cache-offset tests and bounded conditional comparison. |
| A04 | Unconstrained RoPE extrapolation | corrected | T062, T078 | Reject guarantee; explicit length/cache/memory limits and held-out length evidence. |
| A05 | Depth/sibling Tree-RoPE | corrected | T065, T078, T100 | Replace colliding coordinates with generated-prefix ancestor paths; binding remains statically checked. |
| A06 | Consolidated lifecycle | included | T041, T045, T077, T095 | Shared launch/resume/canaries/stages plus original-wrapper parity. |
| A07 | Extraction/manifests | included | T022, T069, T093, T095 | Indexed sources, checksums, synthetic identity, group split and shared extraction. |
| A08 | Four-horizon census | included | T093 | Separate20/50/100/120 complete denominators and exact signed range. |
| A09 | Qualification CLI | included | T024, T027, T028, T049, T095 | Actual checkpoint synthesis and machine-readable gates; reference diagnostics separate. |
| A10 | All logic in CLI classes | corrected | T008, T045, T095 | Reusable services with thin CLI adapters; classes only for stateful needs. |
| A11 | Additional command aliases | not_adopted | — | Existing oeis-learn command namespace suffices; optional names are not a defect. |
| A12 | Retire obsolete wrappers | included | T095 | Archive after parity with original configs and evidence preserved. |
| A13 | 512 default or RoPE removes limits | corrected | T006, T007, T062, T078 | Existing dynamic buffer plus one explicit context contract; no unlimited length claim. |
| A14 | 85+160 universal token minimum | corrected | T006, T007, T073 | Actual body/wrapper/BOS/EOS accounting; fixed benchmark estimate is not universal. |
| A15 | Elite canonical limit | included | T064, T073 | Repair legacy insertion-before-validation and retain valid long programs outside training. |
| A16 | CGI length guard | corrected | T064, T073 | Existing guard becomes profile-derived with common eligibility, not hardcoded512. |
| A17 | Static helper fuel estimates | corrected | T013, T096, T097 | Actual per-call and aggregate fuel measured; no copied instruction estimate. |
| A18 | 25k–50k fuel thresholds | corrected | T080, T083, T096 | Insufficient thresholds rejected; bounded100k/250k/1M tiers and aggregate accounting. |
| A19 | 75% fuel warnings | included | T096 | Configurable near-limit counts and summaries, not a learned optimum. |

## Bundled performance proposals

| ID | Topic | Disposition | Tasks | Concrete coverage / rationale |
| --- | --- | --- | --- | --- |
| P01 | Historical generation timings | corrected | T096, T099 | Historical G16 bottleneck is evidence; actual Ryzen phase timings still required. |
| P02 | KV caching | included | T062, T066, T067, T099 | Include self/cross caches with bounded state/equivalence and measured benefit. |
| P03 | Attention complexity | corrected | T066, T099 | Correct aggregate dense attention O(T^3) to O(T^2), not O(T^2) to O(T); no universal speed promise. |
| P04 | 3–5x cache speedup | corrected | T099 | Unverified projection rejected; use measured practical-effect rule. |
| P05 | Grammar allocation work | corrected | T067, T096, T099 | Instrument actual sets/loops/synchronization; do not assert143 checks per step universally. |
| P06 | uint128 masks | corrected | T006, T007, T067 | Reject insufficient width; sized boolean/word masks cover the full vocabulary. |
| P07 | Batched grammar masks | included | T067 | Preallocated static components plus dynamic stack/local/scope state. |
| P08 | 1.5–2x grammar gain | corrected | T099 | Measure end-to-end and phase costs without multiplying speculative gains. |
| P09 | CUDA deployment | corrected | T001, T046, T047 | AMD-compatible Docker runtime and actual HIP tensors, not NVIDIA wheels. |
| P10 | Weight/VRAM size | corrected | T040, T043, T078, T099 | Record actual parameters and peak shared-host/device memory; weight storage is not full footprint. |
| P11 | GPU 10x/sub50ms claim | corrected | T099 | Unsupported bound rejected; synchronized steady-state end-to-end measurements. |
| P12 | Entropy/ACR causal claim | corrected | T063, T070, T071, T096 | Repair unrestricted entropy/proxy fields and measure actual group variance; no invented historical causality. |
| P13 | Temperature lower bound | corrected | T070, T098, T100 | 0.8 explicit baseline, not an established optimum; changing schedule requires one selected contrast. |
| P14 | Entropy coefficient0.03 | corrected | T063, T070, T098 | No evidence for optimum; repaired legal entropy with explicit0 default and bounded objective contrast. |
| P15 | top_p0.95 diversity | corrected | T063, T070, T075 | Top-p1 policy baseline; legality and useful diversity measured, unsupported truncated policy rejected. |
| P16 | Recurrence speed/100% claim | corrected | T050, T054, T055, T056, T096 | Exact solver and final checks; actual time and independent continuation evidence, no unsupported rate. |
| P17 | AST solver routing | included | T054, T056 | Verified dependency/recurrence structure and exact cached systems; unsupported/rank-deficient not automatically UNSAT. |

## Decision session

| ID | Topic | Disposition | Tasks | Concrete coverage / rationale |
| --- | --- | --- | --- | --- |
| D01 | Generic bootstrap and no imported teachers | included | T030, T032, T033, T035, T069, T076 | D001 preserved: random roots, mechanically generated bootstrap and later admitted generated hypotheses; no imported solutions/metadata guidance. |
| D02 | Immediate proof/arith repairs | included | T013, T051, T057, T061 | Original decision sequencing restored; owning prover/solver defects fixed before architectural measurements. |
| D03 | Per-experiment contract | included | T098, T099, T100 | Frozen question/hypothesis/baseline/variants/manifests/seeds/budgets/metric/effect/stop/inconclusive rules; no final tuning. |
| D04 | Spec/plan notes/research | included | T103, T104 | Completed plan.md replaces preliminary plan-notes; research and contracts distinguish decisions from outcomes. |
| D05 | Unconfirmed overflow semantics | included | T009, T013, T097 | D002 is not falsely labeled approved. Checked256 is a planning default; independently measure cost/intermediate range and unique rescues. |
| D06 | Native LODA C++ first | deferred_loda | — | Only deferred package: pin native interpreter/subset and data boundaries in008. Later user direction supersedes compiler-first proposal. |
| D07 | Conditional LODA-to-WASM | deferred_loda | — | Transcoder only if a demonstrated issue justifies it; no mandatory compiler assumed. |
| D08 | Minimal language comparison | deferred_loda | — | 008 specifies native LODA versus WAT at roughly1–3 days each, token/primitive/runtime attribution and effect/stopping rules. |
| D09 | NixOS/Python/Docker/Ryzen | included | T001, T046, T047, T094 | Beyond-venv dependencies and native builds in Docker, actual device feasibility, no automatic host changes. |
| D10 | Twenty observed terms | included | T004, T021, T024, T068 | 20 is confirmed;100 total and rebased indices remain explicit planning defaults, diagnostic5/10 prefixes use separate manifests. |
| D11 | Fresh weights/breaking compatibility | included | T007, T023, T033, T044, T078 | New codecs/models may break compatibility; old weights remain diagnostic, strict parent chains root at random. |
| D12 | Only LODA deferred; KV discretionary | included | T061, T066, T079, T092, T101, T105 | Latest instruction supersedes containment-only scope. KV included. Known defects require repair; uncertain optimizations have bounded decisions within007. |

## Other design-review inputs

Preserve the nine earlier design corrections: acyclic program/admission identities; final decision lock before target seals/scoring; conservatively charged crash gaps; visible-only seed projection; blob-before-manifest publication; nonquadratic witness grouping; standalone preflight; correct GPU test role; and expected-failing-before-fixed regressions. T008/T022/T025/T026/T041/T042/T046 and their tests retain these obligations.

Coverage means an explicit requirement, design and task disposition, not executed repairs. Full release requires G0–G10. The proposed constitution still needs its recorded feasibility/adoption process; document scripts cannot ratify it or demonstrate Ryzen compatibility. No scope amendment changes the confirmed information boundary or makes final-test feedback available to learning.
