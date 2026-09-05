---

description: "Task list for Phase 2: Synthesis Bootstrapping, Semantic Soundness & Progressive Optimization"
---

# Tasks: Synthesis Bootstrapping, Semantic Soundness & Progressive Optimization

**Input**: Design documents from `/specs/002-synthesis-bootstrapping-and-soundness/`
**Prerequisites**: [plan.md](specs/002-synthesis-bootstrapping-and-soundness/plan.md), [spec.md](specs/002-synthesis-bootstrapping-and-soundness/spec.md), [research.md](specs/002-synthesis-bootstrapping-and-soundness/research.md), [data-model.md](specs/002-synthesis-bootstrapping-and-soundness/data-model.md), [contracts/](specs/002-synthesis-bootstrapping-and-soundness/contracts/)

## Format: `[ID] [P?] [Story] Description with file path`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (`[US1]`, `[US2]`, `[US3]`, `[US4]`, `[US5]`)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish configuration profiles, data models, and CLI structures for Phase 2 implementation.

- [x] T001 Update training configuration profiles with Phase 2 hyperparameters in `configs/train_tier1.yaml`
- [x] T002 [P] Define Phase 2 domain entities and schema data models in `src/oeis_learn/data/models.py`
- [x] T003 [P] Update CLI argument parser and command definitions for Phase 2 commands in `src/oeis_learn/cli/main.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core grammar symbols, telemetry logging, and dynamic state tracking primitives that block user stories.

**⚠️ CRITICAL**: No user story work can begin until this foundational phase is complete.

- [x] T004 Enhance vocabulary tokens, opcodes, and type signatures in `src/oeis_learn/decoder/wat_grammar.py`
- [x] T005 [P] Define structural phase enum (`StructuralPhase`) and frame structures in `src/oeis_learn/decoder/environment_tracker.py`
- [x] T006 [P] Implement diagnostic telemetry recorder and metric trackers in `src/oeis_learn/rl/telemetry.py`

**Checkpoint**: Foundation ready — user story implementation can now proceed.

---

## Phase 3: User Story 1 - Structurally & Semantically Sound Environment-Indexed Decoding (Priority: P1) 🎯 MVP

**Goal**: Enforce mandatory function signature structures, lexical scoping rules, and stack depth/type constraints at sub-$100\,\mu\text{s}$ latency, so that 100% of generated program candidates assemble into compilable WebAssembly binaries without syntax errors, missing parameter declarations, or unbound variable traps.

**Independent Test**: Generate 1,000 candidate programs under unconstrained temperature sampling ($T=1.0$); verify that 100% of generated programs compile in-memory without `PARSE_ERROR`, `MISSING_ENTRYPOINT`, or scoping traps.

### Tests for User Story 1

- [x] T007 [P] [US1] Contract test for dynamic WAT EBNF grammar completeness in `tests/contract/test_wat_grammar_contract.py`
- [x] T008 [P] [US1] Unit test for dual-layer structural phase, lexical scope, and operand stack tracker in `tests/unit/test_environment_tracker.py`
- [x] T009 [P] [US1] Unit test for dynamic Earley trie logit masking and sub-$100\,\mu\text{s}$ latency in `tests/unit/test_grammar_masker.py`

### Implementation for User Story 1

- [x] T010 [US1] Implement dual-layer dynamic state machine ($\Phi_t, \Gamma_t, \Sigma_t, H_t$) with No-Ghost scoping in `src/oeis_learn/decoder/environment_tracker.py`
- [x] T011 [US1] Implement dynamic logit mask generator with bit-parallel type mask indexing in `src/oeis_learn/decoder/grammar_masker.py`
- [x] T012 [US1] Integrate environment-indexed masking into autoregressive token generation loop in `src/oeis_learn/decoder/wat_decoder.py`

**Checkpoint**: At this point, User Story 1 is fully functional and 100% of generated WAT code compiles without syntax or scoping errors.

---

## Phase 4: User Story 2 - Demonstration Bootstrapping & Supervised Warmup (Priority: P1)

**Goal**: Initialize the policy decoder via teacher-forced Supervised Fine-Tuning (SFT) on forward-generated synthetic and canonical reference programs to establish baseline syntactic fluency and arithmetic templates before RL exploration.

**Independent Test**: Train the Transformer decoder on a synthetic dataset of 5,000 forward-generated sequence-program pairs for 5 epochs; verify that greedy synthesis ($T=0.0$) produces $>80\%$ compilable and mathematically valid programs on Stage 1 polynomial tasks.

### Tests for User Story 2

- [x] T013 [P] [US2] Contract test for synthetic demonstration dataset schema in `tests/contract/test_sft_dataset_contract.py`
- [x] T014 [P] [US2] Unit test for forward synthetic program-sequence generation across all families in `tests/unit/test_synthetic_generator.py`
- [x] T015 [P] [US2] Integration test for SFT pretraining convergence ($\mathcal{L}_{\text{SFT}} < 0.50$) in `tests/integration/test_sft_warmup.py`

### Implementation for User Story 2

- [x] T016 [P] [US2] Implement template-driven synthetic demonstration generator covering polynomial, recurrence, modular, and factorial families in `src/oeis_learn/data/synthetic_generator.py`
- [x] T017 [P] [US2] Implement Elite Seed Demonstration Replay Buffer ($\mathcal{D}_{\text{elite}}$) and query methods in `src/oeis_learn/rl/elite_buffer.py`
- [x] T018 [US2] Implement Supervised Fine-Tuning (SFT) teacher-forcing trainer in `src/oeis_learn/rl/sft_trainer.py`
- [x] T019 [US2] Add CLI commands `generate-sft` and `warmup-sft` in `src/oeis_learn/cli/main.py`

**Checkpoint**: User Stories 1 and 2 are functional — the warmed-up SFT policy achieves $>80\%$ pass rate on Stage 1 polynomials.

---

## Phase 5: User Story 3 - Multi-Tiered Dense-to-Sparse Reward Shaping & S-GRPO Exploration (Priority: P2)

**Goal**: Provide continuous learning gradients during early exploration via composite reward shaping (compilation validity, prefix match distance, execution trace divergence attribution) and Conditional Ground-Truth Trajectory Injection (S-GRPO / CGI), smoothly annealing to strict verifiable binary rewards without zero-advantage collapse.

**Independent Test**: Run 20 training steps across a batch containing hard prompts where generated completions fail; verify that Advantage Collapse Rate remains bounded ($\text{ACR} \le 0.15$), non-zero gradient updates are applied, and pass rate scales monotonically.

### Tests for User Story 3

- [x] T020 [P] [US3] Unit test for multi-tiered composite reward computation and cosine annealing in `tests/unit/test_reward_evaluator.py`
- [x] T021 [P] [US3] Unit test for S-GRPO trajectory injection and asymmetric prompt weighting in `tests/unit/test_prompt_weighting.py`
- [x] T022 [P] [US3] Integration test for EGCA gradient localization and CGI update step in `tests/integration/test_egca_training_step.py`

### Implementation for User Story 3

- [x] T023 [US3] Implement composite reward function ($R_{\text{comp}}, R_{\text{prefix}}, R_{\text{dist}}, R_{\text{exact}}$) and cosine schedule in `src/oeis_learn/rl/reward.py`
- [x] T024 [US3] Implement Conditional Ground-Truth Trajectory Injection (S-GRPO / CGI) and AVSPO anchor advantages in `src/oeis_learn/rl/prompt_weighting.py`
- [x] T025 [US3] Implement sequence-chunked EGCA-GRPO policy loss with trace-grounded advantage localization in `src/oeis_learn/rl/egca_grpo.py`
- [x] T026 [US3] Integrate S-GRPO, trajectory injection, and composite rewards into curriculum training loop in `src/oeis_learn/rl/trainer.py`

**Checkpoint**: User Stories 1, 2, and 3 are functional — RL exploration operates with non-zero gradient updates on hard tasks and bounded $\text{ACR} \le 0.15$.

---

## Phase 6: User Story 4 - Progressive Micro-Benchmarking & Diagnostic Telemetry Protocol (Priority: P2)

**Goal**: Establish a 5-tier progressive test harness (Tiers 0–4) with real-time diagnostic telemetry to catch bugs, gradient collapses, and hyperparameter misconfigurations in $<45\text{ minutes}$ before multi-hour runs.

**Independent Test**: Execute the progressive testing harness across Tiers 0, 1, 2, and 3 in $<45\text{ minutes}$; verify that all diagnostic telemetry metrics fall within healthy operational bounds before authorizing Tier 4 execution.

### Tests for User Story 4

- [x] T027 [P] [US4] Contract test for progressive test harness reporting schema in `tests/contract/test_progressive_harness_contract.py`
- [x] T028 [P] [US4] Integration test for Tier 1 oracle reference solution fitting in `tests/integration/test_tier1_oracle_fitting.py`
- [x] T029 [P] [US4] Integration test for Tier 2 single-prompt RL convergence in `tests/integration/test_tier2_single_prompt_rl.py`
- [x] T030 [P] [US4] Integration test for Tier 3 synthetic micro-cohort curriculum progression in `tests/integration/test_tier3_micro_cohort.py`

### Implementation for User Story 4

- [x] T031 [US4] Implement 5-tier progressive test orchestrator and gate evaluator in `scripts/run_progressive_validation.py`
- [x] T032 [US4] Implement real-time telemetry metrics tracker and divergence halt callback in `src/oeis_learn/rl/telemetry.py`
- [x] T033 [US4] Add CLI command `test-progressive` and connect pre-flight checks to `train` command in `src/oeis_learn/cli/main.py`
- [x] T034 [US4] Update long-running autonomous training script with pre-flight gating in `scripts/run_long_e2e_benchmark.py`

**Checkpoint**: User Stories 1 through 4 are functional — pre-flight verification runs in $<45\text{ minutes}$ and halts on metric divergence.

---

## Phase 7: User Story 5 - Self-Supervised Latent Manifold Structuring & Algebraic Homomorphism (Priority: P3)

**Goal**: Regularize sequence representations via non-contrastive Kernel VICReg and explicit additive homomorphism loss ($\mathcal{L}_{\text{add}}$) to prevent dimensional collapse and discover algebraic identities for PSLQ theorem proving.

**Independent Test**: Extract embeddings for a 100-sequence mathematical benchmark; verify that latent space exhibits high rank dispersion ($\text{RDR} \ge 0.80$), forms distinct topological clusters under UMAP/HDBSCAN ($\ge 5$ families), and yields candidate triples verified by PSLQ ($<10^{-50}$ drop) and SymPy proofs.

### Tests for User Story 5

- [x] T035 [P] [US5] Unit test for Kernel VICReg and additive homomorphism loss in `tests/unit/test_vicreg_loss.py`
- [x] T036 [P] [US5] Integration test for latent vector arithmetic, PSLQ relation search, and SymPy proving in `tests/integration/test_discovery_pipeline.py`

### Implementation for User Story 5

- [x] T037 [US5] Implement additive homomorphism loss ($\mathcal{L}_{\text{add}}$) and shift equivariance ($\mathcal{L}_{\text{shift}}$) in `src/oeis_learn/discovery/vicreg_loss.py`
- [x] T038 [US5] Implement algebraic pair dataset generator for SSL pretraining in `src/oeis_learn/data/transforms.py`
- [x] T039 [US5] Update manifold clustering and HNSW vector relation search with rank dispersion tracking in `src/oeis_learn/discovery/manifold.py` and `src/oeis_learn/discovery/vector_search.py`
- [x] T040 [US5] Connect homomorphism-regularized discovery pipeline to CLI command `discover` in `src/oeis_learn/cli/main.py`

**Checkpoint**: All user stories (1 through 5) are functional and independently testable.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, contract validation, performance benchmarking, documentation, and end-to-end verification.

- [x] T041 [P] Update CLI contract and argument validation tests in `tests/contract/test_cli_contract.py`
- [x] T042 [P] Execute batch throughput benchmark across 8 CPU threads in `tests/integration/test_batch_throughput.py`
- [x] T043 Code cleanup, typing verification, and formatting across `src/oeis_learn/`
- [x] T044 Run end-to-end quickstart validation scenarios from `specs/002-synthesis-bootstrapping-and-soundness/quickstart.md`
- [x] T045 Update project documentation and architecture diagrams in `README.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion — **BLOCKS all user stories**.
- **User Stories (Phase 3+)**: All depend on Foundational phase completion.
  - US1 (P1) and US2 (P1) can proceed in parallel once Phase 2 is complete.
  - US3 (P2) depends on US1 and US2 components.
  - US4 (P2) depends on US1, US2, and US3 components.
  - US5 (P3) can proceed independently after Phase 2.
- **Polish (Final Phase)**: Depends on all user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Depends on Phase 2 — No dependencies on other stories.
- **User Story 2 (P1)**: Depends on Phase 2 — No dependencies on other stories.
- **User Story 3 (P2)**: Depends on US1 (grammar decoder) and US2 (elite buffer).
- **User Story 4 (P2)**: Depends on US1, US2, and US3.
- **User Story 5 (P3)**: Depends on Phase 2 — Independent of synthesis pipeline.

---

## Parallel Opportunities

- **Setup Phase**: T002 and T003 can run in parallel.
- **Foundational Phase**: T005 and T006 can run in parallel.
- **User Story 1**: T007, T008, and T009 tests can run in parallel.
- **User Story 2**: T013, T014, T015 tests and T016, T017 implementation tasks can run in parallel.
- **User Story 3**: T020, T021, and T022 tests can run in parallel.
- **User Story 4**: T027, T028, T029, and T030 tests can run in parallel.
- **User Story 5**: T035 and T036 tests can run in parallel.
- **Polish Phase**: T041 and T042 can run in parallel.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T006)
3. Complete Phase 3: User Story 1 (T007–T012)
4. **STOP and VALIDATE**: Run `pytest tests/unit/test_environment_tracker.py tests/unit/test_grammar_masker.py -v`. Confirm 100% of generated programs compile in-memory without syntax or scoping errors.

### Incremental Delivery

1. Setup + Foundational $\to$ Primitives established.
2. User Story 1 $\to$ 100% compilation soundness achieved (**MVP!**).
3. User Story 2 $\to$ Synthetic generator + SFT pretraining warmup ($\ge 80\%$ Stage 1 accuracy).
4. User Story 3 $\to$ S-GRPO + CGI trajectory injection + composite reward shaping ($\text{ACR} \le 0.15$).
5. User Story 4 $\to$ 5-tier progressive test harness ($<45\text{ minutes}$ pre-flight checks).
6. User Story 5 $\to$ Kernel VICReg + additive homomorphism loss + PSLQ theorem discovery.
7. Polish $\to$ End-to-end quickstart validation and documentation.
