**Current status:** [The142043Z target review](learning-pilot-operator-20261026T142043Z.md) passes primary CPU/HIP/policy/operations but fails corpus selection. Local corrected full-capacity analysis is in progress; finish it before using the historical target command below. Reference-cache recovery, exact-byte memory compaction and strict recovery validation are implemented. The accepted hardware soak is unchanged.

**Current target review and repaired handoff:** [First pilot preflight](learning-pilot-operator-20261026T124145Z.md) verifies23 HIP passes and750 CPU passes/one public cost-fixture failure. Corpus/session never start. The scoped v2 probe repair and full diagnostics are implemented with221 host tests; corrected native/corpus/cost evidence remains pending. Use its one corrected combined command; retain old successful evidence and do not repeat the hardware soak/size ladder.

# Learning-pilot implementation and one target handoff

Recorded2026-10-26 UTC on `009-pipeline-throughput`, PR #60. The
[plan](learning-pilot-plan.md) is implemented through all unblocked code/local
work under the separate [contract](../contracts/learning-pilot.md). The actual
32,768-example feasibility audit, current pinned CPU/HIP results,30-minute
restart/cost validation and subsequent admission remain target-dependent.
The long study has not started.

The existing trainer/CheckpointStore now supports the exact fresh104.5M/batch16
strict profile, finite immutable recovery ledger and fixed0/1.5/3/6/12/24-hour
learner-role reservations. Six full296-row evaluations independently score the
actual snapshots across all six arms. Reporting includes per-split token-weighted
metrics, train/dev gaps, unique/repeated exposure, complete negatives/unknowns,
paired regressions/control differences, strata/families, and cautious backtested
2x/4x-token/compute scenarios. No final cohort or automatic strategy change.

Local verification:220 stdlib host tests pass; syntax/JSON/contract/scope and
foundation implementation binding checks pass; requirements16/16. The actual
public command produces a verified PARTIAL ZIP with both missing engines explicit,
dependent work BLOCKED, no omitted mandatory records, readiness BLOCKED and cost
UNKNOWN. Three local synthetic archive probes each verify4,096 payload members;
their timings qualify only this filesystem/source fixture, not workstation cost.
New real CPU/HIP tests are authored/registered, including actual v3 construction,
token metrics, public native scoring-cost sentinels and optimizer/checkpoint resume;
they have not run here. See [machine evidence](learning-pilot-local-evidence.json).

Original T011 closes from the completed archive-fault catalog plus retained actual
UID/ENOSPC/interruption evidence. T110 reopens for new runtime source, preserving
its old accepted closure. Original counts remain87 complete/9 engineering pending/
15 held research. T086 remains pending measured cost and a reviewed long-run
handoff. Original native/performance comparisons and independent Docker are not
silently replaced by this pilot.

## Run once on the workstation

Use the existing checkout/branch and approved inputs from the earlier run. This
imports only the eight exposed development cases, not trained engineering weights
or final cases. No `--run`, profiler command, manual image build or new host
dependency is required.

```bash
git pull --ff-only origin 009-pipeline-throughput &&
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines both \
  --manifest configs/foundation/pipeline_learning_pilot.json \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-learning-pilot-preflight-$(date -u +%Y%m%dT%H%M%SZ)"
```

Return the one sibling `OUTPUT.zip` printed at completion, including a failure or
partial result. Keep the original output directory: its prepared corpus/full-state
checkpoint references are needed for later admission/recovery and are intentionally
excluded from the public ZIP. The invocation prints30-second heartbeats and
preserves bounded role logs/progress. Missing Docker can leave the overall result
PARTIAL while primary Podman plumbing is separately reported.

This is **30 learner minutes, not a30-minute total test**. It includes fresh
initialization,15+15-minute training/restart, three full296-row control/generation/
CPU-scoring observations, all33,024 production corpus admissions, current package/
CPU/HIP/operations qualification and archive calibration. It does not repeat the
accepted size/batch ladder or two-hour hardware soak. No minimum learned success,
loss decrease or transfer threshold applies.

Cost basis and bounds:

| Component | Current measurement or hard limit |
| --- | --- |
| Accepted old full hardware invocation |9,452.064s; different workload, not a forecast for this new scope |
| New larger-corpus and full-control expected duration |UNKNOWN until this first exact-scope calibration; may take hours |
| Actual learner work |2×900s; initialization600s and two role caps1,020s |
| Corpus preparation |9,000s,262,144 counters,33,024 exact production admissions |
| Each of three evaluations |Generation6,000s plus CPU scoring1,500s |
| Combined session |28,800s |
| Full preparation |86,400s inclusive; maximum declared reservation84,000s |
| Engine builds/inspection |One shared14,400s allowance per engine |
| Storage/export |16GiB transient;5GiB free floor;512MiB ZIP/4,097 total members |

No trustworthy measured expected total exists yet for the changed corpus/full
controls. This is why L004/L012's measured-cost portions and T086 remain open;
ceilings and old hardware rates are not relabeled measurements. The consolidated
test obtains the missing data without a separate operator calibration loop.

## Review before a long run

Inspect `stages/pilot.report/{readiness,cost,qualification,learning}.json` and
`long-run-draft.json`. After an earlier failure, the independent finalizer puts
the surviving report in `stages/study.report/`. Every failed/unstarted milestone
and attempted native job remains visible. A zero-success result can have valid
plumbing, but unstarted attempts/missing cost/components cannot admit a study.
The forecast must cover all1,776 rows/10,656 arm outcomes and fit138,240s,
with evaluation at most28,800s and archive at most900s. Early-model cost
observations remain uncertain for later model-generated sources.

The draft is deliberately **not runnable admission**. After archive review,
L013 will freeze an exact-source/image/data/profile/seed costed decision and
explicit reviewed authority, or a concrete no-go/prospective revision. Only that
separately admitted phase6 study can run within the original48h inclusive/24h
learner allowance. Never edit a running decision or rerun to seek a favorable
learning result. No long-run command is issued before that review.

If the admitted study is interrupted, retain its decision and
`run-state/pipeline-pilots/DECISION_HASH/ledger.json` plus all referenced original
outputs. A later explicit invocation uses the same decision and a fresh output,
on the same boot/source/images/resources/CPU quota. Completed work is referenced
at its original producer; interrupted slots stay charged and are not retried.
Acknowledged termination and full-state CPU audit precede later unstarted roles.
Lost milestones remain missing; fatal breach, expired lifetime, changed boot or
unacknowledged cleanup prevents continuation. Recovery never renews the allowance.

After the first archive, avoid another HIP loop unless a concrete affected-scope
defect requires it. Actual long-run learning/capacity interpretation and the
same-regime-versus-curriculum decision remain L014/L015, not conclusions of this
short functional validation. No new PR CI is awaited.
