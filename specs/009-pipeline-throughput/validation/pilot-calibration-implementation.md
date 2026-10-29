**Calibration is complete:** [141405Z audited PASS](pilot-calibration-operator-20261027T141405Z.md).
All three full-clock cost observations and current-source qualification pass.
No further calibration or duplicate soak is requested. The next implementation is
[versioned long-run admission/profile wiring](long-run-admission-next.md); no long
run is admitted yet. Earlier failed collections remain linked in the review.
The command below is the completed handoff, retained for reproducibility.

# Saved-snapshot calibration: one target handoff

This follows the [successful primary preflight and cost NO-GO](learning-pilot-operator-20261026T213151Z.md)
and implements [the scoped contract](../contracts/pilot-calibration.md). Code/local and actual target checks now pass, including private-file import,
pinned CPU/HIP and three full original-clock observations with a passing candidate
forecast. The following text records the historical delivered preparation scope.
No learned-success threshold, learning-strategy change or long run is registered.

## Run once from the existing checkout

**Completed on141405Z; no rerun requested.**

Keep the complete213151Z output directory; its public ZIP excludes the required
private files. This command reads its prepared corpus, original evaluation
settings and three saved diagnostic weight snapshots. It verifies all public and
private bindings before engine access. It never starts training, generates
reference rows, selects another corpus or repeats the hardware soak/size ladder.

```bash
git pull --ff-only origin 009-pipeline-throughput &&
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_pilot_calibration.json \
  --inputs tmp-validation/009-learning-pilot-preflight-20261026T213151Z \
  --max-seconds 86400 \
  --output "tmp-validation/009-pilot-calibration-$(date -u +%Y%m%dT%H%M%SZ)"
```

Current affected images/CPU/HIP/primary operations run once in the same DAG.
Only the selected engine is registered; independent Docker portability stays
pending. Shared images can use the engine's normal build cache; successful old
qualification is retained at its producer, not claimed as current runtime evidence.

Preserve the new complete output directory and return its sibling ZIP. A failed
private import still produces a diagnostic archive before any engine call. Missing
or changed data/weights block without regeneration or a replacement snapshot.
No manual profiler or multiple separate commands are needed.

## Runtime and cost expectations

The previous run measured approximately6206s (1h43m) after excluding its6488s
corpus preparation and1813s learner work. That is a historical comparison, not a
measured duration of this new scheduling policy. Full60s portfolios can take
longer than the old proportional slices, especially at initialization.

Three generation roles each have8600s (2h23m20s) maximum; each CPU scoring role
has1500s (25m). Their combined maximum is30300s (8h25m), inside a31000s session.
The roster has at most131 groups per snapshot (57neural/74template);60s each
requires7860s before model load/encoding/drain/publication overhead. Each role
reserves740s beyond that arithmetic bound and still fails explicitly on a parent
cut. This is a planning bound, not an empirical performance guarantee.
Current-source qualification, image builds, private-file auditing and archive
probes add time. The combined invocation has a24h hard ceiling, with separately
charged phase/build/export limits; it is not a short learner test. Progress and
bounded logs use the existing container supervision paths.

The new candidate forecast doubles the largest of three full finite-portfolio
costs and the all-selected scoring envelope for all six snapshots. It assesses
36000s evaluation/41400s safety while retaining24 learner hours,48h inclusive
and the80% threshold. It can still return OVER_BUDGET or UNKNOWN. Cost completeness
never turns grammar failures, timed-out native work or unscored continuations
into complete scientific negatives/successes. The new forecast cannot pass the
legacy long-run parser; actual long-run authority/profile review is subsequent.

## What is preserved

The original generator/recipe/seed,33024 native-admitted generic records, frozen
296 prompts,8/1/1/8 attempt roster, six scientific arms, exact100-term independent
scoring, native4s clocks, original60s portfolio clocks,104.5M model/FP32 execution,
private-data boundaries and strict/random-weight training provenance remain.
The three weight snapshots are diagnostics only and cannot resume learning.
The parent213151Z attempts retain their original complete/incomplete/unstarted
states. The new policy has distinct job and public clock identities; no expired
attempt is resumed and no old allowance is renewed.
