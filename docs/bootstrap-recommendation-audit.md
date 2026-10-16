# Bootstrap recommendation audit and next experiments

This audit maps the two complete recommendations supplied on 2026-10-16 to
PR #41. Implementation and experimental evidence are different: CPU/container CI
can verify mechanics, but no encoder has yet passed the new GPU learnability
experiment or demonstrated better structural transfer.

## Recommendation coverage

| Recommendation | Implementation and evidence | Remaining condition or limitation |
|---|---|---|
| Diagnose the existing selected checkpoint without changing its run | `fresh/checkpoint_diagnostic.py`; portable `diagnose`; original checkpoint contract/profile/runtime verified, weights-only load and both code/runtime identities recorded | Requires retained weights and original examples on the GPU host; pre-007 weights are not accepted by this loader |
| Fixed training sample and all synthetic-development examples, stratified | `fresh/diagnostics.py`: fixed hash-ordered training sample; standalone diagnosis selects every structural and parameter-development row, reports target-length bands and expression/assignment/branch/loop categories | One-hour host cap, 50-minute generation budget; unattempted/timed-out cases stay in the denominator, with explicit coverage; a partial report is not a completed diagnostic |
| Correct versus shuffled conditioning | Same greedy policy, cyclic selection of a different prompt; separate shuffled denominator | A stratum with only one distinct prefix cannot supply a meaningful shuffle control |
| Cached versus uncached decoding | Teacher-forced logit parity at training checks; actual greedy token/termination/execution comparison on up to three length-spread examples per split during standalone diagnosis | Numerical logit agreement alone is not labeled generation agreement; unfinished comparisons report no pass |
| Termination, legality, prefix fit, continuation, diversity, EOS and incomplete streams | Separate 20-term prefix and 80-term continuation scores; native/reference agreement; parse check; output behavior hashes; raw and grammar-conditioned EOS probability traces; partial tokens survive budget exhaustion | Continuation scoring uses independent execution of the admitted synthetic oracle, not OEIS hidden labels; exact equality over 100 indices is not an all-domain proof |
| Tiny scratch memorization gate | `learnability`: same 32 shortest admitted training prompts for all arms, production optimizer/decoder, 1,000-update cap, checks every 100; at least 31/32 exact greedy executions, conditioning drop and cache-logit parity | Run this on short connected inputs. A successful process exit does not mean `learnability.json` passed. No GPU pass claimed yet |
| Preserve old generator and introduce connected targets | `data/bootstrap_programs.py`; original `generic_programs.py` unchanged; initialized connected trees/state, 16 parameter variants, no named OEIS teachers | Semantic cancellation can still occur; no unsafe dead-code deletion or algebraic rewrite is applied |
| Control constants, identity, branches and loops; retain short verified representatives | Frozen `connected-training-mixture/v1`: approximately one quarter per stage, at most 5% prefix-constant examples, at most one identity prefix; shortest admitted behavior representative; full original admission packets retained | Behavior labels use the visible twenty terms and are explicitly called prefix properties. Preparation may report a shortfall if a stratum cannot be filled within its lifetime budget |
| A/B/C/D progression and retain earlier practice | Expressions, compositions, bounded carried-state loops, conditions; both train and structural-development synthesis must reach configured 80% before promotion; prior stages remain eligible | Nested loops and unrestricted long programs are intentionally outside this initial bounded generator |
| Separate parameter and structural generalization | New non-tiny bootstrap runs deterministically reserve one variant from structures with at least three distinct behaviors, capped at 1/8 of input rows and 128; same split for all encoder arms; original structural split untouched | A dataset with no repeated structures reports an empty parameter-development split. It cannot establish parameter transfer. Tiny memorization does not reserve these examples |
| Synthesis-first selection, frequent validation, explicit controls | Synthetic execution primary, full structural loss supporting, 250-update default, explicit LR/patience/update ceilings; sampler/curriculum/replay state checkpointed | Periodic execution is sampled and bounded; standalone diagnosis supplies full development coverage. A wall-time cap need not be exhausted |
| Faithful tri-stream control | Explicit historical d256, 4 encoder/4 decoder, 4 heads, FFN1024, dropout0.1, FiLM, 16 odd primes, valuation cap32, no summary tokens | Restores the numeric architecture, not old warmstart weights, vocabulary, scaffolds or family teachers |
| Exact bytes supplement rather than replace features | `encoder/hybrid.py`: gated exact magnitude/difference bytes added before the same Transformer; byte-only arm also available | Exact serialization does not prove the learned projection is injective or superior |
| Matched three-arm comparison, including large-number tests | Common decoder seeded independently of encoder parameter count; parameter counts and update/benchmark timing recorded; explicit large adjacent, congruence and difference tests; public `--encoder` forwarding regression | Actual HIP performance and learning must be measured. New explicit encoder overrides also work on original sealed v1 inputs; unmodified v1 defaults stay unchanged |
| Attribute generator and encoder effects separately | Shared original-data screen with all three matched encoders, then fixed-encoder old-versus-connected comparison; commands in `bootstrap-training.md` | Do not interpret a simultaneous encoder+data change as an encoder ablation |
| Bounded visible-prefix search and constant grounding | Shared attempt/token/time budget; prefix verification before selection; greedy/sampling, width-two beam when attempts >=8, bounded literal/operator repair using visible constants | This is bounded literal grounding, not the full general symbolic fitter. The initial default is four attempts; an eight-attempt portfolio is a separate new-run comparison after bootstrap competence |
| Independently admitted training-only replay | Opt-in shorter-target replay, independent 100-term verification, structural/parameter/prefix holdouts, durable checkpoint state, at most32 targets | Only optimization-training prompt IDs are eligible; reserved parameter-development rows and all development/final continuations never enter updates. Keep off until bootstrap competence is established |
| Historical checkpoint positive control; new final cohort | Strict run never initializes from pre-007 checkpoints; final scoring explicit and deferred; documentation checks new cohort memberships against previously inspected ones | Historical migration/positive-control qualification remains separate and requires a compatible checkpoint. No claim that the old incompatible vocabulary was repaired |

## What is deliberately not claimed

This is not a restoration of the prototype's entire learning system. It keeps
spec 007's checked arithmetic, independent execution, complete checkpoints,
probability accounting and strict completion measurement. No family-specific
scaffolds, historical demonstrations, automatic completion repairs or warmstart
weights enter the primary track. GRPO and unrestricted symbolic constant solving
are not enabled by this PR. Those are later experiments, not prerequisites for
proving short-program SFT learnability. Likewise, no GPU utilization, synthesis
threshold, encoder superiority or release-qualification gate is certified by CPU CI.

## Decision sequence

1. Run `diagnose` on the previous selected checkpoint; inspect coverage, correct
   versus shuffled execution, EOS traces and cache-generation parity. If coverage
   is incomplete, do not infer failure on rows the budget could not evaluate.
2. Run a matched three-arm pilot on the **same original sealed data**. Treat it as
   an attribution screen, not permission for another long training run.
3. Prepare connected inputs once. Run all three tiny controls against that shared
   input set. Any failed memorization/conditioning/cache gate blocks scaling;
   inspect token alignment, masks, conditioning, optimization and decoding first.
4. Compare the chosen encoder on old and connected data with equal seed, hardware,
   LR, objective and update ceilings. Run standalone diagnosis on selected pilots.
   Require meaningful parameter and structural synthesis, not just lower loss.
   If a split is absent, collect more verified variants/structures before claiming
   transfer. Inspect stage/length denominators and constant-output diversity.
5. After competence, compare four-attempt and eight-attempt search at explicitly
   fixed token/time budgets, then enable verified replay in a separate new run.
   Keep the no-replay control. Broader constant fitting or GRPO needs its own
   design and measured justification; do not silently fold it into this screen.
6. Benchmark the selected model and size the batch from measured throughput and
   memory headroom. Expand data diversity before increasing epochs. Only then
   create a longer-run configuration. Use a fresh, membership-checked final cohort
   once after selection, and package the selected checkpoint's diagnostic bundle.

The companion command guide names inputs, run outputs, resume, evaluation and
packaging commands. Every experiment gets a new run directory and immutable
identity; existing failed/completed runs are never reset to reissue budgets.
