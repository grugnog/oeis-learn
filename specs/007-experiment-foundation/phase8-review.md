# Phase 8 review and remediation

This phase improves the existing encoder, decoder, sampler, original RL/SFT trainers,
curriculum, archive admission and checkpoint publication. New modules implement
missing services alongside those owners; they do not introduce another codebase.
The frozen foundation/v1 SFT control remains explicitly selectable. Phase 9 work
and LODA are outside this PR.

## Concrete corrections

- Real bounded self/cross-attention K/V caches reuse the existing decoder weights.
  CPU fixtures compare every logit through 1,024 tokens, mixed lengths, finished
  rows, beam reordering, invalidation and eager/SDPA attention. Prefix-ancestor
  positions use only tokens already observed; RoPE has explicit offsets.
- Sampling retains true action support, temperature, behavior log probabilities
  and per-candidate seeds. Every nested module mode is restored. The original
  trainer uses dropout-off scoring, actual population group variance, fresh
  REINFORCE or symmetric clipped GRPO, a full frozen encoder/decoder reference,
  exact masked categorical KL, safe entropy and per-trajectory normalization.
  Full-horizon rewards reject short successful executions, and telemetry labels
  only actual zero variance. Virtual anchors and asymmetric advantage wrappers
  cannot silently reintroduce the old objective.
  Forced/PAD actions are excluded; sampled EOS is included. Old shaping,
  synthetic-positive injection and unaudited teacher blending fail explicitly.
- Exact conditioning uses sign plus 32 magnitude bytes, exact rebased index and
  length/padding, power-of-two residues, explicit zero valuation and 33-byte
  differences. Dormant heads and inactive fusion parameters are removed from the
  original encoder. The lossy tri-stream control remains declared as such.
- Qualified generic sampling assigns a dependency-preserving structural group
  before an independent parameter draw. Structural 80/10/10 splits and expanded
  parameter/composition/OEIS-prefix guards apply to every archive origin.
  The archive retains distinct algorithms, limits each training output group to
  four implementations and shares its sampling weight across those algorithms.
- Admission validates provenance, codec round trip, independent execution and
  continuation before publishing an index. Long valid source is retained with
  explicit context ineligibility. Legacy inspection neither seeds teachers nor
  fabricates fuel/continuation. Actual 25% replay/75% generic SFT updates use a
  frozen archive view and advance counters only after optimizer commit.
- Curriculum records aggregate measured-cost visits and correct conditional
  probabilities for sampling without replacement. Default allocation is four
  prompts, 2..16 candidates each and total 32; caps, deterministic redistribution
  and unallocated budget are explicit. Statistics and named RNG state resume. Mixture weights are validated and used
  at stage two; resumed sampling binds the prompt pool as well as RNG state.
- Trace evidence comes from actual reference interpreter events with logical
  instruction IDs and WAT lowering sites. Native causal tracing reports
  unsupported. Whole-trajectory credit is explicit; guessed textual spans and
  guessed coverage are removed.
- Search has one attempt/expansion/token/wall ledger, seeded stochastic/greedy
  generation, beam search and replayable typed repairs. Source dedup does not
  collapse distinct algorithms sharing a prefix. Portfolio 32 is 16 stochastic,
  8 beam and 8 repair; baseline 16 has a different comparison identity. Retrieval
  must use a verified frozen training archive, including after resume.
- Self-training keeps continuation truth in a pinned training-only controller
  capability. Rounds freeze replay/retrieval, independently check discoveries,
  retain excluded evidence without admitting it to training, and reverify the
  runtime before the next round. Search hypotheses enter SFT with separate
  provenance; they never become on-policy samples.
- Versioned learning checkpoints bind a durable blob to model/profile/runtime,
  active reference, replay/round/curriculum/search/sampler/prefetch/scaler/LR
  scheduler state and RNG. Inactive fields are null. Interrupted fixtures match
  the next policy/replay update and search selection. The durable budget ledger
  remains independent of checkpoint rollback. Existing weights-only legacy files
  are inspection artifacts, not complete continuation state.

## Model and qualification

`configs/foundation/qualified.yaml` declares d384, four encoder layers, six decoder
layers, six attention heads and FF1536. Its sinusoidal model has **21,631,494 active
parameters** (encoder 7,328,128; decoder 14,303,366). All three supported position
variants have CPU forward/backward coverage. This is the requested approximately
25-million-parameter class; unused parameters are not added to reach a round total.

G8 executes owning regressions and pins their JUnit evidence, implementation and
runtime. CPU software PASS is separate from accelerator qualification. CPU evidence
cannot enable a hardware-qualified learning session. `foundation learning-train`
and `learning-search` require actual G7/G8 evidence and explicit service profiles.
Their CPU diagnostic mode reports `qualified=false`. The root qualification image
runs fresh G7, G8 and real CLI train/resume/search acceptance in CI.

Local broader foundation regressions encounter unavailable
`/proc/sys/kernel/random/boot_id`; protected budget accounting remains unchanged.
Ubuntu CI supplies the required OS surface. No training-speed or GPU-readiness
claim follows from the local CPU tests.

## Pending accelerator acceptance

Run within the project's Docker GPU environment:

```bash
lab-gpu exec -- python -m oeis_learn.evaluation.learning_gate \
  --output runs/007_phase8_gpu --hardware
```

The gate requires actual GPU FP32 cached/full equivalence for all declared
position variants at 1/2/513/1024 tokens and a finite FP16+GradScaler encoder/decoder update of the declared model for
all three position variants. The gate rejects incomplete architecture coverage.
It rejects skips and failures. Thermal throughput, worker-count selection,
peak memory and the final hardware profile remain measurements for the hardware
qualification phase; this PR does not invent those results. T062/T066/T067/T078/
T079 stay partially checked until their accelerator-dependent acceptance passes.

Versioned continuation formats differ intentionally from older weight files.
Native instruction tracing remains explicitly unsupported; the supported real
reference event path is tested. Macros and program proof remain gated for phase 9.
