---
description: "Task list for Phase 3: Inductive Algorithmic Generalization, Anti-Shortcut Regularization & Fine-Grained Credit Assignment"
---

# Tasks: Inductive Algorithmic Generalization, Anti-Shortcut Regularization & Fine-Grained Credit Assignment

**Input**: Design documents from `/specs/003-algorithmic-generalization-and-credit-assignment/`  
**Prerequisites**: [plan.md](specs/003-algorithmic-generalization-and-credit-assignment/plan.md), [spec.md](specs/003-algorithmic-generalization-and-credit-assignment/spec.md), [research.md](specs/003-algorithmic-generalization-and-credit-assignment/research.md), [data-model.md](specs/003-algorithmic-generalization-and-credit-assignment/data-model.md), [contracts/](specs/003-algorithmic-generalization-and-credit-assignment/contracts/)

## Format: `[ID] [P?] [Story] Description with file path`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (`[US1]`, `[US2]`, `[US3]`, `[US4]`, `[US5]`)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish configuration profiles, data models, and schemas for Phase 3 anti-shortcut regularization and credit assignment.

- [x] T001 Update training configuration profile with Phase 3 hyperparameters (`beta_sft`, `beta_kl`, `alpha_ent`, `pbrs`, `lexicase`, `non_triviality`) in `configs/train_tier1.yaml`
- [x] T002 [P] Define Phase 3 domain entities (`NonTrivialityEvaluation`, `CoTrainingBatch`, `FineGrainedAttributionSpan`, `PotentialState`, `LexicaseSelectionBatch`) in `src/oeis_learn/data/models.py`
- [x] T003 [P] Update CLI argument parser and command definitions for Phase 3 co-training, PBRS, and non-triviality flags in `src/oeis_learn/cli/main.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core mathematical and structural primitives (variance metrics, attention padding masks, and tracing interfaces) that block user stories.

**⚠️ CRITICAL**: No user story work can begin until this foundational phase is complete.

- [x] T004 Implement empirical output variance $\mathbb{Var}_n[P(n)]$ and input sensitivity $\mathcal{S}_{\text{input}}(P)$ calculator in `src/oeis_learn/rl/reward.py`
- [x] T005 [P] Integrate strict `tgt_key_padding_mask` enforcement and mini-chunk logit projection ($L_{\text{chunk}}=256$) in `src/oeis_learn/decoder/wat_decoder.py`
- [x] T006 [P] Update AST tracer with causal error span mapper and basic block execution coverage tracking in `src/oeis_learn/sandbox/tracer.py`

**Checkpoint**: Foundation ready — user story implementation can now proceed.

---

## Phase 3: User Story 1 - Non-Degenerate Inductive Synthesis & Input Sensitivity (Priority: P1) 🎯 MVP

**Goal**: Condition policy rewards on input-parameter sensitivity and cross-input mutual information, ensuring that candidate programs that ignore `$n` or emit static constant sequences receive zero surrogate reward and a static penalty.

**Independent Test**: Evaluate the model on 100 non-constant sequence tasks; verify that $\ge 95\%$ of generated programs contain active bindings to `$n`, produce non-zero empirical variance $\mathbb{Var}_n[P(n)] > 0$, and achieve non-zero input sensitivity $\mathcal{S}_{\text{input}}(P) > 0$.

### Tests for User Story 1

- [x] T007 [P] [US1] Unit test for empirical output variance, input parameter sensitivity, and non-triviality gating in `tests/unit/test_reward_evaluator.py`
- [x] T008 [P] [US1] Unit test for batch-level cross-input mutual information proxy ($R_{\text{MI}}$) in `tests/unit/test_mutual_information.py`

### Implementation for User Story 1

- [x] T009 [US1] Implement non-triviality reward gate zeroing $R_{\text{dist}}$ and $R_{\text{prefix}}$ with static penalty on constant shortcuts in `src/oeis_learn/rl/reward.py`
- [x] T010 [US1] Implement batch-level cross-input mutual information proxy $R_{\text{MI}}$ over executed minibatch output vectors in `src/oeis_learn/rl/reward.py`
- [x] T011 [US1] Integrate non-triviality checks and active `$n` parameter binding verification into evaluation workflow in `src/oeis_learn/rl/trainer.py`

**Checkpoint**: At this point, User Story 1 is fully functional and degenerate constant shortcuts receive zero surrogate reward and cannot serve as optimization attractors.

---

## Phase 4: User Story 2 - Demonstration Co-Training & Anchor Loss Regularization (Priority: P1)

**Goal**: Co-train reinforcement learning policy updates with an auxiliary Supervised Fine-Tuning (SFT) demonstration loss and bound policy drift with an unbiased Schulman reference model KL divergence penalty.

**Independent Test**: Train the policy for 30 epochs under mixed SFT+RL optimization; verify that the policy's capacity to generate valid `loop`, `block`, and `br_if` constructs does not degrade relative to the initial SFT baseline (token entropy $\mathcal{H}(\pi_\theta) \ge 1.50$, reference perplexity $\text{PPL}_{\text{ref}} \le 1.30$).

### Tests for User Story 2

- [x] T012 [P] [US2] Unit test for unbiased Schulman per-token KL divergence penalty and entropy bonus in `tests/unit/test_kl_regularization.py`
- [x] T013 [P] [US2] Integration test for SFT co-training loss blending ($\mathcal{L}_{\text{RL}} + \beta_{\text{SFT}}\mathcal{L}_{\text{SFT}}$) and padding attention masks in `tests/integration/test_co_training_step.py`

### Implementation for User Story 2

- [x] T014 [US2] Implement unbiased per-token Schulman KL penalty $\beta_{\text{KL}}\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ and entropy bonus in `src/oeis_learn/rl/egca_grpo.py`
- [x] T015 [US2] Implement blended SFT co-training loss over elite demonstration buffer $\mathcal{D}_{\text{elite}}$ in `src/oeis_learn/rl/trainer.py`
- [x] T016 [US2] Ensure exact target key padding masks (`tgt_key_padding_mask`) in teacher-forcing loss passes in `src/oeis_learn/rl/sft_trainer.py` and `src/oeis_learn/rl/egca_grpo.py`

**Checkpoint**: User Stories 1 and 2 are functional — the policy retains inductive loop templates and syntactic idioms without policy drift during online RL.

---

## Phase 5: User Story 3 - Fine-Grained Execution-Grounded Credit Assignment (EGCA) (Priority: P2)

**Goal**: Isolate the exact instruction token where candidate execution state deviated from expected sequence values, zero-masking advantages for all subsequent downstream tokens while conserving total advantage mass.

**Independent Test**: Evaluate gradient updates on candidate programs that correctly compute initial terms $n \in [0, 5]$ but fail at $n = 6$; verify that token advantages for $t > \max T_{k^*}$ are masked to zero, concentrating $\ge 90\%$ of gradient mass on the causal error span.

### Tests for User Story 3

- [x] T017 [P] [US3] Contract test for credit attribution and localized advantage schema in `tests/contract/test_credit_attribution_contract.py`
- [x] T018 [P] [US3] Unit test for downstream token zero-masking and total advantage conservation in `tests/unit/test_egca_credit_assignment.py`
- [x] T019 [P] [US3] Integration test for execution trace divergence localization and basic block coverage masking in `tests/integration/test_egca_coverage_attribution.py`

### Implementation for User Story 3

- [x] T020 [US3] Implement priority-gated failure classification (`SYNTAX`, `CONSTRAINT`, `LOGIC`, `CORRECT`) in `src/oeis_learn/sandbox/tracer.py`
- [x] T021 [US3] Implement sequence divergence to token span mapping ($k^* \to T_{k^*}$) and downstream zero-masking ($t > \max T_{k^*}$) in `src/oeis_learn/sandbox/tracer.py`
- [x] T022 [US3] Update EGCA-GRPO loss kernel to enforce total advantage conservation $\sum a_{i,t} = A_i$ on localized causal error windows in `src/oeis_learn/rl/egca_grpo.py`
- [x] T023 [US3] Integrate coverage-based unexecuted instruction masking (FGO) into training step in `src/oeis_learn/rl/trainer.py`

**Checkpoint**: User Stories 1, 2, and 3 are functional — credit smear is eliminated in stack bytecode updates.

---

## Phase 6: User Story 4 - Potential-Based Reward Shaping (PBRS) & Down-Sampled Lexicase Selection (Priority: P2)

**Goal**: Enforce policy invariance through potential-based state differences ($\gamma \Phi(s') - \Phi(s)$) over AST completion states and evaluate rollouts via down-sampled lexicase selection over randomized test cases.

**Independent Test**: Run 20 iterations comparing PBRS against raw heuristic distance rewards; verify that PBRS eliminates constant shortcut attractors while accelerating convergence toward exact sequence matching.

### Tests for User Story 4

- [x] T024 [P] [US4] Unit test for AST potential-based reward shaping ($F = \gamma \Phi(s') - \Phi(s)$) and telescoping sum in `tests/unit/test_pbrs_reward.py`
- [x] T025 [P] [US4] Unit test for down-sampled lexicase rollout selection over randomized test cases in `tests/unit/test_lexicase_selection.py`

### Implementation for User Story 4

- [x] T026 [US4] Implement Potential-Based Reward Shaping engine with structural phase ($\phi_{\text{comp}}$) and parameter binding ($\phi_{\text{bind}}$) potentials in `src/oeis_learn/rl/reward.py`
- [x] T027 [US4] Implement down-sampled lexicase rollout filtering algorithm across randomized sequence indices in `src/oeis_learn/rl/prompt_weighting.py`
- [x] T028 [US4] Integrate dynamic competence-driven cosine annealing for surrogate potentials in `src/oeis_learn/rl/trainer.py` and `src/oeis_learn/curriculum/sampler.py`

**Checkpoint**: User Stories 1 through 4 are functional — dense shaping signals preserve policy invariance and reward per-input specialists.

---

## Phase 7: User Story 5 - Generalization Extrapolation ($K=100$) & Automated Theorem Discovery (Priority: P3)

**Goal**: Verify synthesized algorithms across an extended extrapolation horizon ($N+K$ terms with $N=20, K=100$) and Minimum Description Length bounds ($M_{\text{MDL}} \le 1.20$), and discover algebraic identities via Kernel VICReg, high-precision PSLQ, and SymPy proofs.

**Independent Test**: Evaluate graduated candidate programs on $K=100$ future terms; verify that $100\%$ of passing algorithms compute exact integer values across the full horizon and generate verified symbolic proofs.

### Tests for User Story 5

- [x] T029 [P] [US5] Unit test for extrapolation verifier ($K=100$) and anti-memorization MDL ratio bounds ($M_{\text{MDL}} \le 1.20$) in `tests/unit/test_extrapolation_verifier.py`
- [x] T030 [P] [US5] Integration test for additive homomorphism loss ($\mathcal{L}_{\text{add}}$), PSLQ integer relation detection ($<10^{-50}$ drop), and SymPy symbolic proof execution in `tests/integration/test_discovery_pipeline.py`

### Implementation for User Story 5

- [x] T031 [US5] Update extrapolation verifier and MDL complexity calculator in `src/oeis_learn/curriculum/extrapolation.py` and `src/oeis_learn/curriculum/mdl_verifier.py`
- [x] T032 [US5] Enhance Kernel VICReg loss with additive homomorphism penalty ($\mathcal{L}_{\text{add}}$) in `src/oeis_learn/discovery/vicreg_loss.py`
- [x] T033 [US5] Update CLI `synthesize` and `discover` commands with Phase 3 extrapolation thresholds ($K=100, M_{\text{MDL}} \le 1.20$) in `src/oeis_learn/cli/main.py`

**Checkpoint**: All user stories (1 through 5) are functional and independently testable.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, contract validation, long-running overnight benchmark, and documentation.

- [x] T034 [P] Update CLI contract test in `tests/contract/test_cli_contract.py`
- [x] T035 [P] Run full 5-tier pre-flight progressive validation hierarchy (`test-progressive --max-tier 3`) in `scripts/run_progressive_validation.py`
- [x] T036 Launch and monitor Phase 3 overnight benchmark tracking under `runs/003_phase3_inductive_generalization/` via `scripts/run_long_e2e_benchmark.py`
- [x] T037 Code cleanup, typing verification, and formatting across `src/oeis_learn/`
- [x] T038 Update project documentation and architecture diagrams in `README.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion — **BLOCKS all user stories**.
- **User Stories (Phase 3+)**: All depend on Foundational phase completion.
  - US1 (P1) and US2 (P1) can proceed in parallel once Phase 2 is complete.
  - US3 (P2) depends on US1 (reward structures) and US2 (co-training batching).
  - US4 (P2) depends on US1 and US3.
  - US5 (P3) can proceed independently after Phase 2.
- **Polish (Final Phase)**: Depends on all user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Depends on Phase 2 — No dependencies on other stories.
- **User Story 2 (P1)**: Depends on Phase 2 — No dependencies on other stories.
- **User Story 3 (P2)**: Depends on US1 (reward/tracer) and US2 (loss updates).
- **User Story 4 (P2)**: Depends on US1 (reward module) and US3 (tracer).
- **User Story 5 (P3)**: Depends on Phase 2 — Independent of online RL optimization pipeline.

---

## Parallel Opportunities

- **Setup Phase**: T002 and T003 can run in parallel.
- **Foundational Phase**: T005 and T006 can run in parallel.
- **User Story 1**: T007 and T008 tests can run in parallel.
- **User Story 2**: T012 and T013 tests can run in parallel.
- **User Story 3**: T017, T018, and T019 tests can run in parallel.
- **User Story 4**: T024 and T025 tests can run in parallel.
- **User Story 5**: T029 and T030 tests can run in parallel.
- **Polish Phase**: T034 and T035 can run in parallel.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T006)
3. Complete Phase 3: User Story 1 (T007–T011)
4. **STOP and VALIDATE**: Run `pytest tests/unit/test_reward_evaluator.py tests/unit/test_mutual_information.py -v`. Confirm that static constants receive zero surrogate rewards and non-triviality penalties.

### Incremental Delivery

1. Setup + Foundational $\to$ Primitives established.
2. User Story 1 $\to$ Non-triviality gating & input sensitivity active (**MVP!**).
3. User Story 2 $\to$ Demonstration co-training + Schulman KL penalty active.
4. User Story 3 $\to$ EGCA downstream zero-masking & advantage conservation active.
5. User Story 4 $\to$ Potential-Based Reward Shaping & down-sampled lexicase selection active.
6. User Story 5 $\to$ $K=100$ extrapolation & PSLQ/SymPy discovery pipeline active.
7. Polish $\to$ Run 003 benchmark execution and documentation updates.
