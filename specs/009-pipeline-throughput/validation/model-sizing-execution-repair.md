**Latest accepted sizing result**: [model-sizing-operator-20261025T180134Z.md](model-sizing-operator-20261025T180134Z.md). Corrected sizing is accepted at56f6156/source735ede…:170 host,692 CPU tests plus31 subtests,20 actual-HIP tests and all15 trials pass. Actual8.12M/31.90M/104.50M profiles achieve median training5,934/4,219/2,092 tokens/s with2.45/3.36/5.89GiB observed role upper memory. All84 parity records have zero mismatches and640 strict native jobs complete; zero verified generated programs remains explicit.009:M004/T110 close at that source; original counts78 complete/18 runtime pending/15 research held.104.5M is the proposed next engineering candidate, not a production/learning choice. Persistent state/checkpoint/resume, longer-sequence and training-batch efficiency work remains before a longer run. No repeat of this size ladder or multi-day training is requested. Earlier handoffs below are historical.

# Corrected sizing execution policy and one rerun

[First target collection](model-sizing-operator-20261025T170755Z.md) passes687 CPU
tests plus30 subtests and all three CPU references/counts, but fails the first m8
encoder HIP parity assertion. All15 performance trials remain never started.
No model-size throughput/memory selection is available. The observed fused encoder
and memory-efficient attention dispatch was inconsistent with the sizing scope's
declared eager policy; its causal contribution to the numerical discrepancy remains
to be confirmed by actual HIP. The failed archive stays immutable and source-bound.

## Completed repair

`pipeline-sizing-execution/v1` explicitly requires FP32/highest matmul, TF32 off,
deterministic algorithms without warn-only mode, encoder/MHA fastpath off and only
SDPA MATH enabled. One scoped context covers the CPU reference, complete GPU trial
(including updates/validation/generation) and isolated runtime fixture. It audits
the actual flags and restores the previous settings on success, exception or nested
exit. Existing production and other-scope defaults stay unchanged.

Preparation, references and trial results bind this exact policy. New input/reference
records use v2; strict v1 schema/seal readers remain for historical evidence, while
current execution rejects missing/old policy instead of promoting it. Every profile
still uses the same architecture, initialization/seed, mathematical input groups,
FP32 model, data,32 updates,64 actual generation attempts, original clocks/phase/
resource caps and complete-only aggregate rules. No tolerance is relaxed. Changing
execution policy to implement the declared scope happens before any sizing timing;
there is no resampling of a favorable timing block.

The runtime fixture traps fused encoder/MHA entry points so an unexpected bypass
fails immediately. Public per-component `parity.jsonl` records actual shape/dtype,
finite status, mismatch count, maximum absolute error, retained atol2e-5/rtol2e-4
and execution-policy identity **before** the original assertions. Fixture output
now routes through `OEIS_TEST_OUTPUT/model-sizing-cpu|cuda` when available, preserving
diagnostics in the operator archive even on failure. No tensor/oracle values or
hidden truth are exposed. CPU/HIP numerical success remains pending, not inferred
from the dispatcher repair.

The selected PyTorch2.9 APIs are its documented
[SDPA context](https://docs.pytorch.org/docs/2.9/generated/torch.nn.attention.sdpa_kernel.html)
and [MHA fastpath control](https://docs.pytorch.org/docs/2.9/backends.html#torch.backends.mha).
They require no dependency/host change.

## Available validation

**170 host stdlib tests pass**. Five new cases cover scoped policy enforcement,
exception restoration, nesting, wrong backend rejection and historical v1/v2
schema/admission separation. Existing negative policy/resource/thread/reference/
partial cases now also reject missing/mismatched execution evidence. Complete host
adapter graph, source/canonical binding, all15 unstarted slots and archive tests pass.
Compilation/whitespace checks pass. Pinned CPU/HIP fixtures are updated and authored,
but were not executed locally; no host runtime dependencies were installed.

The actual public CLI exits3 for unavailable local Podman and retains all15 blocked
trials in a verified63-listed-member archive plus manifest.
[Local evidence](model-sizing-execution-local-evidence.json) binds exact source
`735ede18987edb3d48eeab8c6d9001ec10a5f63c6f0d6581304702acc44d5f15`,720 entries,
and archive/member hashes. The007 implementation binding is refreshed with its actual
foundation serializer;68 dispositions/full007=false and historical negatives remain.
All checklist items remain read-only and16/16 complete; no extension hooks exist.

## One corrected target command

After the completed code work, from the existing checkout:

```bash
git pull --ff-only origin 009-pipeline-throughput &&
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_model_sizing.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-model-sizing-$(date -u +%Y%m%dT%H%M%SZ)"
```

Return the single sibling ZIP, even on failure. The command repeats only applicable
current-source prerequisites, references and the never-started sizing roster; it
does not repeat the accepted isolated GPU/native/co-scheduling comparisons. No
separate build/profile/export or manual branches are needed. Run without competing
heavy compute on the shared workstation.

Planning1–3 hours with warm caches remains **uncalibrated**, including the newly
explicit math path. The immutable bounds remain1080s internal/1110s external per
GPU role, fifteen external roles4h37m30s, three600s CPU references, phase4 total16680s
below18000s, shared14400s build/engine and86400s global with original archive/
cleanup reserves. Do not enlarge a timeout or drop a profile to obtain a favorable
measurement. A repeat parity failure keeps sizing blocked and now exports better
diagnostics.

009:M004/current-sourceT110 remain pending. Original counts77 complete/19 runtime
pending/15 research held or blocked. No learned-quality comparison, default/learning
model choice, persistent-state soak or multi-day run is launched. New PR CI is not
awaited after publishing this repair; prior producer CI was inspected once with15
completed successes and one in-progress build, zero reported failures.
