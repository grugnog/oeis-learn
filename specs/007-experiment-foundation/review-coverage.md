# Coverage audit against the earlier reviews and decisions

Date: 2026-09-16. Audited PR head: `1a764eb6909ea7dc99a5033763a2307a75028024`.

**Conclusion: 007 does not include every earlier concrete fix.** It specifies a substantial correctness and measurement foundation. Some defects are contained by excluding their subsystems, some improvements are deliberately deferred, and several details were not explicitly carried into the backlog. The earlier validator's 26/26 coverage measures requirements already written in 007; it does not measure coverage of the input reviews. Nothing in this document claims implementation is complete.

Inputs: the user-supplied `oeis-learn-independent-review(1).md` (20 recommendations, LODA follow-up, A01–A19 and P01–P17 proposal assessments); `oeis-learn-spec-decisions(1).md`; later visible user decisions; and 007's spec, plan, research, contracts, tasks and validation analysis. The proposal tables below trace the bundled documents through their individually validated entries in the independent review. They are not a new execution of the original probes.

**Status meanings:** Included = a concrete implementation/test obligation exists. Contained = the faulty path cannot qualify under 007, but its implementation is not repaired. Partial = only part of the recommendation is included. Deferred = outside 007, not a claim of completed work. Not adopted = rejected, optional or already addressed in the reviewed source. Task references are to [tasks.md](tasks.md).

## Material discrepancies and recommended disposition

| ID | Finding | Consequence / recommended disposition |
| --- | --- | --- |
| GAP-01 | The decision record explicitly requires immediate repair of known false-proof defects. R7, FR-019 and T018 instead quarantine the legacy prover and defer its repair. | **Unresolved sequencing divergence.** Keep quarantine and add a narrowly scoped repair of symbol identity, valid domains and evidence-backed counterexample status before architectural experiments. This is distinct from building a general proof system. Do not describe the current deferral as user-approved. |
| GAP-02 | Constant-grounding, RL, curriculum/replay and discovery defects are excluded from the strict path, not repaired. | Acceptable containment for an SFT foundation, but each must have a repair/conformance gate before re-enablement. The inventory below preserves the concrete fixes; there are no implementation tasks for most of them yet. |
| GAP-03 | The review requests clean-wheel/resource packaging and actual CI enforcement. T051 runs regression checks but does not explicitly add these deliverables. | Add a CPU CI gate installing the built wheel outside the source tree and exercising packaged WAT resources and foundation regressions. A native-enabled CI job is conditional on enabling that backend. |
| GAP-04 | Multi-horizon census, finite/table metadata policy, growth-label correction and broader synthetic-family/transformation holdouts are not fully specified. | Define source eligibility before freezing the comparison. Preserve the 100-term primary denominator; add separate 20/50/100/120 descriptive denominators if retaining A08. Do not claim all-OEIS or unseen-family coverage from the current split. |
| GAP-05 | Near-limit resource summaries, legacy wrapper migration and retaining valid over-context programs have no complete tasks. | Track these explicitly for operational follow-up; do not count generic logging, new strict admission or disabling old replay as the corresponding repairs. |
| GAP-06 | The requested LODA comparison and overflow/expressiveness measurements are only a handoff to 008. | Write their bounded protocol before either long training arm. Include budgets, outcome thresholds, attribution limits and inconclusive handling. They are not a completed experiment specification in 007. |

These findings amend the earlier broad claim that no further design clarification was needed. This audit records discrepancies; it does not silently expand the 52 implementation tasks or treat proposed repairs as accepted amendments.

## Independent review: all twenty recommendations

| Review | 007 coverage and concrete remaining work | Status |
| --- | --- | --- |
| 1. Exact integer arithmetic | FR-002/003; T009, T013/014, T019 repair negative scalar multiplication, full-width literals, carry/sign boundaries, checked logical overflow and independent reference labels. | Included for the declared WAT subset |
| 2. Genuine qualification | T021, T023–029 and T049 require actual checkpoint generation, hidden continuations, top-one versus any-of-budget and truthful readiness. Reference canaries become diagnostics. Historical warmstart/Run011 launchers are not all rewritten. | Included on the strict path; legacy behavior contained |
| 3. Sound constant grounding | T011/T015 reject the C*C false assignment through final execution. Placeholder solving is disabled. Dependency classification, nonlinear handling, rank-deficient recurrence systems, integer-safe HNF, honest solver statuses, routing and matrix caching remain unimplemented. | Contained, not repaired |
| 4. Shared execution | T012–017 provide typed parsing, lowering before compilation, four-result decoding, single/batch parity, fresh state and common budgets. Regex optimization and unqualified native paths are disabled. Rust parity and migration of every legacy entry point remain later work. | Included for enabled paths; other paths contained |
| 5. False symbolic proofs | T018 rejects imported/legacy PROVEN in foundation evidence. It does not repair assumption-bearing symbol identity, shifted-domain intersection, singularities, or the requirement for an actual counterexample witness. Formula equality, program correctness and OEIS identity remain distinct claims. | Contained; **GAP-01** |
| 6. Data and evaluation | T020–029 preserve exact indexed terms, complete-100 eligibility, signed range, frozen development/final groups, duplicate/prefix/limited-shift handling, hidden truth and synthetic provenance. Broader transformed/family holdouts, finite/table policy, growth labels and multi-horizon census are incomplete. | Partial; **GAP-04** |
| 7. RL probability correctness | T044 selects SFT; the plan explicitly does not repair legacy RL. Before reuse: stored behavior probabilities with the actual mask/temperature/support, scaffold exclusion, frozen encoder+decoder reference, module-mode restoration, valid entropy/reward variance and truthful trace attribution. | Contained; deferred repair |
| 8. Reproducibility and orchestration | T005, T023, T038–049 cover active-state checkpoints, external hashes, exact CPU resume, effective config, budgets, real metrics, Docker and shared services. Clean-wheel packaged-resource tests, CI enforcement and wholesale legacy launcher consolidation are not explicit. | Partial; **GAP-03/05** |
| 9. Decoding velocity | T006/007 avoid fixed 128-bit mask width and enforce dynamic grammar correctness. Prefill/decode self- and cross-attention caches, cache invalidation, equivalence tests, batched prompts, preallocation, synchronization reduction and measured attention backend improvements are deferred. | Partial; performance work deferred |
| 10. Complete program representation | T006/007/T012 define a complete literal/local codec, typed WAT parser, EOS/length contract and lossless admission. Compact learned IR/LODA is later. Legacy elite insertion leakage and CGI hardcoded limits are excluded, not repaired; valid over-context archive retention is not specified. | Partial; **GAP-05** |
| 11. Diverse bootstrap and self-training | T030–037 implement mechanically generated, independently verified, frozen generic data. Broader structural/parameter holdouts, verified self-training, multiple implementations per sequence, Pareto replay, near-miss reuse and archive revalidation are deferred. Current output deduplication retains one preferred program, not a multi-implementation archive. | Bootstrap included; other work deferred |
| 12. Search | T024/025 specify bounded greedy-plus-seeded sampling, deduplication, prefix-only selection and fixed attempts. Beam/best-first/repair portfolios, skeleton reuse, retrieval, sound partial pruning and counterexample-guided search are deferred. | Minimal baseline included |
| 13. Curriculum and replay | T030/T033/T035 reject missing prefixes, fabricated continuations and unaudited replay. They do not repair successes-then-failures bandit ordering, fixed allocations, dormant replay updates or visit-level uncertainty/cost accounting. | Contained; deferred repair |
| 14. Runtime expressiveness and fuel | T016 and execution contracts specify per-call/aggregate limits, fresh state and exhaustion outcomes. Streaming/block interfaces, bounded arrays, broader wide arithmetic, adaptive fuel tiers and near-limit summaries are not implemented by this plan. | Resource correctness included; expansion deferred; **GAP-05** |
| 15. Exact neural conditioning | Exact integers are authoritative in artifacts, but the plan explicitly retains the potentially lossy tri-stream neural encoder. Lossless sign/byte features, parity/powers-of-two features, inactive-head cleanup and variable-prefix ablations are deferred. Exact program tokenization does not solve this input-encoding issue. | Deferred |
| 16. Ryzen deployment | T001/T040/T046/047 qualify the actual AMD GPU, pinned Docker userspace, FP32 updates, shared-memory budgets and no CPU substitution. Sustained thermal/power testing, validated mixed precision and worker/batch sweeps remain later measurements. | Readiness included; tuning deferred |
| 17. Program-based discovery | Artifact provenance and finite-evidence boundaries are included. Decoder-program-driven relation proposals, normalized program/subprogram graphs, multiple implementations, semantic rewrites and posthoc novelty analysis are deferred. | Interfaces only; system deferred |
| 18. Relation-search correctness | No tasks repair first-50 versus ranked retrieval, cubic enumeration, single-index PSLQ, unused configuration, exact multi-index nullspaces, independent validation indices, empty validations or misleading rank/nullity metrics. The discovery path is not used by 007. | Deferred, with concrete repair gate required |
| 19. Scoped proofs | FR-019/SC-006 prevent finite matches from becoming theorem/novelty claims. Polynomial/recurrence certificates, bounded-machine claims, finite-state proofs, invariants, termination/range arguments and translation validation are deferred. | Evidence boundary included; proof capabilities deferred |
| 20. Positions and capacity | The small-model control is retained and length behavior tested. RoPE, structural positions, larger models and controlled length/capacity ablations are deferred. No unsupported positional-corruption or guaranteed-extrapolation claim is adopted. | Deliberately deferred |

## Bundled architecture proposals: A01–A19

| ID | Disposition in 007 |
| --- | --- |
| A01 | Not adopted: embedding corruption was not established. |
| A02 | Already addressed in the reviewed source by dynamic positional-buffer extension; do not invent another crash fix. |
| A03 | RoPE remains a later measured ablation. |
| A04 | No unlimited-length or extrapolation guarantee adopted; explicit budgets remain. |
| A05 | Tree-RoPE deferred; depth/sibling coordinates alone do not establish binding. |
| A06 | Shared strict lifecycle/resume/real evaluation included; migration of every historical launcher deferred. |
| A07 | Exact indexed extraction, manifests, source identities and grouping included in T020–022. |
| A08 | Only primary complete-100 census included; separate 20/50/100/120 reports missing, GAP-04. |
| A09 | Strict evaluation CLI and actual qualification included in T024–029/T049. |
| A10 | Corrected design adopted: reusable services with thin CLI adapters, not all business logic under CLI classes. |
| A11 | Optional aliases not adopted; use the existing command namespace. |
| A12 | Historical-wrapper retirement after parity remains future work, GAP-05. |
| A13 | Unified strict length contract included; RoPE is not treated as removing limits. |
| A14 | Body/EOS/wrapper accounting included; the old 85+160 estimate is not a universal threshold. |
| A15 | Strict admission avoids legacy elite insertion, but does not fix that old entry path or preserve all valid long programs. |
| A16 | CGI is inactive; its hardcoded legacy limit is not repaired. |
| A17 | Actual fuel accounting required; static helper instruction estimates cannot populate measured fields. |
| A18 | Insufficient 25–50k thresholds not adopted; finite higher limits declared. Calibrated escalation tiers are deferred. |
| A19 | Resource metrics included, but configurable near-limit threshold/count summaries are not explicit, GAP-05. |

## Bundled performance proposals: P01–P17

| ID | Disposition in 007 |
| --- | --- |
| P01 | Historical generation bottleneck remains evidence, not a Ryzen forecast or new task. |
| P02 | KV and encoder-projection caching explicitly deferred; no implementation task. |
| P03 | Correct complexity distinction retained by deferring the optimization without promising linear total attention. |
| P04 | No unmeasured 3–5x cache speedup promised. |
| P05 | Grammar correctness covered; its isolated allocation/CPU cost is not yet measured. |
| P06 | Fixed uint128 rejected; T006/007 explicitly cover vocabularies above 128 IDs. |
| P07 | Dynamic legality included; preallocated vectorized grammar-mask performance work deferred. |
| P08 | No unmeasured 1.5–2x grammar speedup promised. |
| P09 | AMD-compatible Docker/PyTorch qualification included; NVIDIA CUDA wheels are not the target deployment. |
| P10 | Actual resource measurement included; weight size is not treated as peak training memory. |
| P11 | No unsupported GPU speedup or sub-50ms claim adopted. |
| P12 | Misleading metric fields prohibited; RL disabled. The historical causal claim remains unestablished. |
| P13 | Fixed baseline sampling is specified; optimal temperature/schedule tuning deferred. |
| P14 | Entropy-bonus tuning irrelevant to initial SFT; any future RL objective must first repair probability/entropy accounting. |
| P15 | Initial top-p 1 avoids a nucleus-support complication; 0.95 is not assumed to guarantee useful diversity. |
| P16 | Final execution verification included; recurrence solver disabled and its speed/generalization claim not adopted. |
| P17 | Typed parsing and final verification included; polynomial/recurrence signature recovery and solver dispatch deferred. |

## Decision-session reconciliation

| Decision / constraint | Coverage and provenance |
| --- | --- |
| D001 generic bootstrap, random weights, no imported solutions or named teachers | FR-012/013 and T030–037/T044 include this. Legacy weights/replay remain historical. Assisted learning requires a later separate decision. |
| Provisional package sequence | 007 combines research contract, trusted strict execution and reproducible infrastructure, plus bounded GPU feasibility. Later package numbers were not reserved in the attachment. The immediate proof repair requirement was not preserved: GAP-01. |
| Per-experiment scientific standard | 007 supplies infrastructure contracts; 008 still needs the complete hypothesis, contrasts, seeds, budgets, minimum useful effect, stopping and inconclusive rules. Do not claim those experiment specs exist. |
| Spec, research and plan-phase notes | 007 has a completed plan.md and research.md instead of preliminary plan-notes.md. That is a completed-plan replacement, not missing implementation planning. |
| D002 overflow decision | The attached record explicitly says unconfirmed. Checked signed-256 in 007 is labeled a planning default, not a user-confirmed answer. Logical checks versus intentional limb wrap and finite LODA capacity remain distinct. |
| Overflow-cost and wider-intermediate measurements | The fixed-program provenance strata, logical-range/magnitude traces, paired uninstrumented timing, fuel, rescued correct candidates and unique-target gains are not fully specified in 007. Carry these into a bounded 008 diagnostic; do not infer intermediate range from output census. |
| Native LODA C++ first | Later user messages supersede the earlier compiler-first review recommendation. 007 defers the native runtime implementation to 008 and defines shared interfaces only. |
| LODA-to-WASM | Conditional on a demonstrated issue under the later user direction; not a prerequisite or already-authorized compiler project. Any future compiler still requires semantic conformance. |
| Minimal comparison and primitive trade-offs | Initial paired comparison is intended, not a broad experiment matrix. Token/action length, primitive inventory, throughput and held-out solves must be measured; native-runtime comparison combines language, primitives, arithmetic and runtime effects. A targeted follow-up contrast is conditional on a decision it can resolve. |
| LODA corpus and seq | No imported solution corpus in strict training. 008 must pin a subset and runtime semantics, bound memory/numbers, and disable seq initially or define an allowed learned dependency closure without access to held-out solutions. Native runtime reuse avoids inventing loop semantics but does not remove these boundaries. |
| Hardware and dependency boundary | Ryzen AI Max+ 395, 128 GB, 2 TB, NixOS/Python/Docker and beyond-venv dependencies in Docker included in FR-017 and workstation tasks. Actual readiness is unmeasured. |
| Roughly 1–3 days per arm | Recorded as user guidance for the later comparison; 007 does not launch it or silently convert the range into an approved exact schedule. |
| Twenty observed terms | Included as the confirmed input count. 100 total / 80 hidden terms, rebased indices and initial SFT are separately labeled planning defaults. |
| Restart compatibility | Fresh random weights and a new codec are explicitly allowed; old weights cannot masquerade as strict continuation. |

## Other review inputs and validation limits

The nine design-review corrections already listed in [validation/analysis.md](validation/analysis.md) are present: digest-cycle removal; finalization ordering; crash-gap budget accounting; visible-only seed projection; blob-before-manifest publication; nonquadratic grouping evidence; standalone preflight dependencies; correct GPU test role; and failing-before-fixed regression sequencing. They do not substitute for the independent review inventory above.

Priority: reconcile GAP-01 and make the packaging/CI obligation explicit before describing 007 as covering all immediate repairs. Keep disabled-subsystem fixes gated before re-enablement. Resolve dataset eligibility and write the bounded 008 protocol before freezing the comparison. Cache optimization remains the leading deferred performance improvement; proof expansion, broader search, encoding and capacity changes remain conditional follow-ups rather than additional mandatory training arms.
