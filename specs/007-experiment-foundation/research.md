# Research and resolved planning decisions

**Date**: 2026-09-16. Code evidence is pinned to `aa2c8c4579dd74fdd8114ec9456960a98fcf500e`; findings were rechecked while planning. Earlier Run011 measurements inform priorities, not Ryzen performance predictions. No new training or hardware benchmark was run for this document.

## R1 — Repair exact execution before throughput tuning (P0)

**Decision**: Checked logical signed-256 macros, native bit-vector semantics, full-range literal encoding and a bounded independent interpreter. Classify overflow and resource limits; do not widen WASM arithmetic as a prerequisite.

**Rationale**: Lowering emits an arbitrary literal as one i64, and the scalar helper's negative correction is defective. Available-prefix range coverage does not bound intermediate growth or incorrect candidate behavior. Overflow checks live in helpers, not extra model-predicted instructions; their overhead is measured by the bounded 007 numerical diagnostic. This design does not claim every native limb program has mathematical checked-integer semantics.

**Alternatives considered**: Wrapping wide macros would answer a different numerical question. Arbitrary-precision WASM increases implementation scope. Native LODA can later use its wider existing runtime but still needs a declared range/resource profile.

**Evidence**: [Lowering](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/sandbox/lowering.py#L63-L71), [scalar helper](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/sandbox/preamble.wat#L149-L198), [WebAssembly numerical semantics](https://webassembly.github.io/spec/core/exec/numerics.html). Current [census](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/reports/oeis_precision_census.json) mixes available-prefix and complete-horizon coverage; recalculate the latter.

## R2 — One final-source acceptance path (P0)

**Decision**: Parse/type-check, canonicalize, lower, compile and independently verify every admitted final candidate through shared services. Disable unqualified paths in the initial smoke; repair exact solver dispatch and typed optimization in US5, and reject bad proposed assignments even if a solver says success.

**Rationale**: The Dixon path returns an unchecked assignment; the optimizer ignores local.tee when deleting declarations; synthesis compiles macro source before lowering; Python multi-result and batch handling differ. Fixing only one caller leaves contradictory acceptance behavior.

**Alternatives considered**: The earlier containment-only choice is superseded by the user's request to repair known defects. Configuration still rejects unqualified paths; repair and parity evidence are mandatory before enabling their variants.

**Evidence**: [Solver return](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/decoder/constant_solver.py#L616-L628), [optimizer](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/sandbox/optimizer.py#L67-L88), [synthesis ordering](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/evaluation/synthesis.py#L198-L234), [fallback](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/sandbox/fallback_runner.py#L75-L80).

## R3 — Freeze exact indexed truth and measure real checkpoints (P0)

**Decision**: 20 visible + 80 hidden is an explicit new profile; group exact duplicates, ambiguous prefixes and bounded shifts; score one frozen representative per group. Use the existing actual-checkpoint synthesis approach with a visible-only process boundary. Final proposals/selection are sealed before truth is opened.

**Rationale**: Canary CLI executes canonical programs while echoing a checkpoint path. The real synthesis path exists but injects family-based recurrence scaffolds, hardcodes 120 terms and invents a memory value. Those facts prevent directly reusing historical scores. The user approved twenty input terms; choosing 100 total is a planning default, not a quotation of user approval.

**Alternatives considered**: Random ID splits leak detectable copies; selecting top-one after hidden verification biases the score; an ID-stratified benchmark based on known families adds metadata assumptions. Grouping does not establish full mathematical independence, so its limitations are explicit.

**Evidence**: [Canonical canaries](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/cli/evaluate_canaries.py#L115-L164), [metadata scaffolds](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/evaluation/synthesis.py#L390-L413), [old horizon contract](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/evaluation/protocol.py#L65-L68).

## R4 — Generic, verified, lossless SFT first (P0)

**Decision**: Separate generic typed-program sampling from existing named-family generators; validate independently and freeze a pool before training. Use a complete decimal-immediate codec and mean per-program next-token loss. Start with the existing small FP32 model and new random weights.

**Rationale**: Current SFT automatically accepts existing JSON or generates polynomial/Fibonacci/factorial demonstrations. Its codec silently maps unknown constants to UNK. A strict flag on the run name cannot fix these admission paths. Frozen offline pools provide the minimal deterministic control. RL repairs and complete online-state checkpoints are separate qualified slices in this same feature. The sampler's register/control skeleton is a declared generic prior, not a claim of zero mathematical assumptions.

**Alternatives considered**: Importing LODA solutions violates the selected research track; named teachers belong to a separate assisted decision. Only rejecting unknown values under the old tiny vocabulary would severely constrain generated programs. Byte tokenization is possible, but dedicated operators plus decimal digits give a simple lossless initial codec; token length/learnability comparison belongs to 008.

**Evidence**: [SFT defaults](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/rl/sft_trainer.py#L58-L103), [named generation](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/data/synthetic_generator.py#L58-L80), [lossy vocabulary](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/decoder/wat_grammar.py#L44-L47).

## R5 — Resume complete state; hash final bytes externally (P0)

**Decision**: Atomic checkpoint blob plus external manifest, complete active optimizer/RNG/data/counter state, immutable pool order, durable budget ledger and two-generation recovery. CPU deterministic fixtures use an injected clock; GPU limitations are separate.

**Rationale**: SFT saves weights/epoch/loss only. Existing checkpoint code hashes an initial file, inserts that hash and overwrites it. Even correct weight restoration cannot reconstruct update order or spent budget. A crash may roll back model state but must not refund computation time.

**Alternatives considered**: Adding only optimizer state misses RNG/permutation and budget; reseeding on every restart changes continuation. Universal bitwise CPU/GPU/version reproducibility is not promised by PyTorch.

**Evidence**: [Incomplete SFT checkpoint](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/rl/sft_trainer.py#L308-L319), [self-referential checkpoint hash](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/evaluation/checkpoint.py#L74-L93), [PyTorch reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html).

## R6 — One Docker baseline and an actual GPU gate (P0)

**Decision**: Start with the versioned AMD Ryzen PyTorch image named in the plan, resolve its digest, lock the complete environment and perform three actual model updates. Inspect existing host GPU access; do not add NixOS packages/drivers or silently switch devices. Native dependencies/build tools stay in Docker.

**Rationale**: AMD documents a Ryzen-compatible container configuration, but NixOS is not thereby certified. Containers share the host kernel. The current general ROCm matrix has moved on; the chosen 7.2.1 image is a conservative baseline rather than a claim of latest support. HIP uses PyTorch's cuda device namespace, so check HIP runtime identity as well as tensor residency.

**Alternatives considered**: Host ROCm installation violates the requested dependency boundary. A large wheel/kernel/backend sweep is unnecessary initially. If the pinned baseline fails, preserve evidence before selecting one targeted alternative.

**Evidence**: [Versioned Ryzen installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installryz/native_linux/install-pytorch.html), [current AMD matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html), [ROCm Docker host requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.1/how-to/docker.html), [PyTorch HIP behavior](https://docs.pytorch.org/docs/2.9/notes/hip.html), [Docker resource enforcement](https://docs.docker.com/engine/containers/resource_constraints/). No image digest or host measurement is fabricated in this planning artifact; T001 produces them.

## R7 — Repair the prover and preserve separate evidence scopes (P0)

**Decision**: Foundation result schemas permit only `proof_status=not_claimed`; the strict path does not call the legacy prover. Repair its symbol/domain/witness defects in US5, and add restricted certificate checkers plus program-based relation analysis in US7. A proof record never mutates a finite CandidateResult into a theorem.

**Rationale**: The prover can substitute a different assumption-bearing symbol from the one in a parsed expression and incorrectly promote a shifted relation. A finite match cannot repair an unsound proof procedure. A narrowly scoped repair is required immediately by the decision record; broader 007 proof support is deliberately limited to specified certificate classes.

**Alternatives considered**: Merely adding a warning leaves false proof promotion active. Quarantine alone did not satisfy the earlier decision. The user has now explicitly included concrete repairs and bounded discovery support; an unrestricted theorem prover remains outside the meaning of this requirement.

**Evidence**: [Symbol construction/substitution](https://github.com/grugnog/oeis-learn/blob/aa2c8c4579dd74fdd8114ec9456960a98fcf500e/src/oeis_learn/discovery/symbolic_prover.py#L60-L117).

## R8 — Use official Spec Kit workflows, with explicit verification limits

**Decision**: Pin CLI and generated Codex skills to official v1.0.7 (`fe1d00e3ccaf495880aaf90fb0e17679e82f065b`), use template resolution/setup/prerequisite scripts, then perform requirement/task/schema/link and semantic consistency checks. This branch switches the managed integration from Copilot to Codex because the installed combination fails upstream multi-install safety validation.

**Rationale**: Spec Kit's scripts locate artifacts and inspect the environment; they do not prove an experiment design sound or execute proposed acceptance tests. The separate constitution proposal makes changed governance reviewable. A proposed constitution and a planned hardware gate must not be labeled ratified/passed.

**Alternatives considered**: Hand-copying commands omits managed manifests; leaving the integration safety error unresolved gives an incomplete installation; editing managed metadata to make status green would conceal the error.

**Evidence**: [Pinned Spec Kit source](https://github.com/github/spec-kit/tree/fe1d00e3ccaf495880aaf90fb0e17679e82f065b), [official integration reference](https://github.com/github/spec-kit/blob/fe1d00e3ccaf495880aaf90fb0e17679e82f065b/docs/reference/integrations.md), [upgrade workflow](https://github.com/github/spec-kit/blob/fe1d00e3ccaf495880aaf90fb0e17679e82f065b/docs/upgrade.md).

## Deferred decisions

Only native LODA operations/numeric limits, cross-language token/primitive trade-offs and the paired language-comparison protocol belong to 008. KV caching, precision/capacity/position/encoding probes, learning/search repairs, abstraction and scoped proofs are included in 007. The latest user instruction supersedes the earlier deferrals; bounded conditional experiments do not require adopting an unproven variant.


## R9 — Include the measured bottleneck and repair learning mathematics

**Decision**: Include self/cross-attention KV caching with logit and mask equivalence, correct behavior-policy accounting and complete active-state resume, exact byte conditioning, and atomic multi-program replay. Keep SFT as the control, not as a reason to leave RL bugs unfixed.

**Rationale**: The earlier review measured generation as the dominant laptop phase. Cache engineering can be tested without another training arm. Stored old probabilities, masked support, a frozen complete reference and module-mode restoration are correctness conditions, independent of whether GRPO ultimately outperforms SFT.

**Alternatives considered**: A permanently disabled RL path does not satisfy the expanded repair request. An unconditional RL/encoder/position/model-size sweep obscures attribution. Use the single contrast selected by [the bounded protocol](contracts/experiments.md).

**Evidence**: Existing `decoder/sampler.py`, `decoder/wat_decoder.py`, `rl/trainer.py`, `rl/egca_grpo.py`, `rl/elite_buffer.py` and `curriculum/symple_bandit.py`; detailed source-specific design in [learning contract](contracts/learning.md). [GRPO research](https://arxiv.org/abs/2402.03300) motivates the objective, not its superiority for a random small model. [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) requires careful causal masks/dropout; implementation must use the pinned version's behavior.

## R10 — Broaden WAT and derive discovery from programs

**Decision**: Implement bounded checked-wide, arrays and streaming profiles, ranked program-derived proposals, exact multi-index validation and replayable restricted certificate classes. Retain finite synthesis evidence as a separate type. Mine only typed subprograms from the permitted archive.

**Rationale**: Fresh compute(n) can repeat work and the current discovery pipeline does not establish that its relationships arise from generated programs. Checked machine overflow, formula assumptions and finite evidence are distinct; a universal PROVEN label hides those distinctions.

**Alternatives considered**: Unlimited automatic theorem proving is neither promised nor needed. Returning UNKNOWN outside enumerated supported classes is correct behavior, not a deferred repair. Imported LODA solutions remain excluded. Typed rewrite rules with checked side conditions suffice without requiring an equality-saturation dependency.

**Evidence**: Existing `discovery/{pipeline,symbolic_prover,vector_search,pslq_solver,numerical_validator}.py` and the independently reproduced review counterexamples; concrete algorithms and class limits in [repairs contract](contracts/repairs.md).

## R11 — Repair packaging, source policy and decision traceability

**Decision**: Add actual clean-wheel CPU/native CI, migration parity for old launchers, exact multi-horizon/source eligibility and complete external-review-to-task mapping. Cap non-LODA measurements at two engineering hours and a six-hour single-contrast pilot.

**Rationale**: Internal spec requirement coverage did not establish coverage of all input reviews. Source-tree tests miss missing packaged WAT, and disabling native tests hides parity defects. Distinct source horizons need distinct denominators. Fixed bounds keep the expanded implementation scope from becoming an automatic research sweep.

**Alternatives considered**: Keeping these as vague later work conflicts with the latest user direction. Requiring every proposed architecture to win is unsound; a tested interface and evidence-backed not-triggered/inconclusive decision satisfies a conditional experiment, but never a known repair.

**Decision provenance**: Generic bootstrap, fresh weights, native LODA first, conditional transcoding, twenty observed terms, NixOS/Docker and roughly 1–3 days per language are user directions. The latest instruction includes every non-LODA gap and lets us choose KV inclusion; KV is included. Checked logical signed-256, 20+80, initial SFT, bounded supported proof classes and the exact numerical experiment thresholds are explicit planning defaults. None is falsely quoted as an earlier approval. The original immediate proof-repair sequencing is restored.
