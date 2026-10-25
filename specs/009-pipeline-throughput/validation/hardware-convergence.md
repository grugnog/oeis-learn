**Current first-HIP handoff:** [Full-scale104.5M engineering soak](soak-implementation.md). Code/local delivery is complete:190 host tests and the public CLI produce a verified archive. One combined command performs qualification, batch selection, persistent two-window work, controlled restart, both-engine operations and reporting. Current original counts77/19/15 retain runtime and research gates. No new CI wait or learning/default admission. The accepted sizing and earlier component evidence remain scoped to their producers.

The records below are historical handoffs and source-specific decisions; use the current handoff above for this implementation.

**Sizing refinement accepted**: [model-sizing-operator-20261025T180134Z.md](model-sizing-operator-20261025T180134Z.md). Corrected sizing is accepted at56f6156/source735ede…:170 host,692 CPU tests plus31 subtests,20 actual-HIP tests and all15 trials pass. Actual8.12M/31.90M/104.50M profiles achieve median training5,934/4,219/2,092 tokens/s with2.45/3.36/5.89GiB observed role upper memory. All84 parity records have zero mismatches and640 strict native jobs complete; zero verified generated programs remains explicit.009:M004/T110 close at that source; original counts78 complete/18 runtime pending/15 research held.104.5M is the proposed next engineering candidate, not a production/learning choice. Persistent state/checkpoint/resume, longer-sequence and training-batch efficiency work remains before a longer run. No repeat of this size ladder or multi-day training is requested. This updates the earlier architecture direction below; accepted component evidence remains scoped to its producer.

**User-directed refinement**: After this report, the user requested a larger model before the long-run choice and authorized the proposed compact sizing screen. [Implementation/handoff](model-sizing-implementation.md) registers8.12M/31.90M/104.50M fresh profiles without changing accepted component results or production defaults. The no-model-change direction below describes the prior scope; current hardware sizing is explicit and separate. The persistent-state soak still waits for model choice and its own qualification.

# Converge hardware settings before a longer useful test

Decision after [accepted co-scheduling evidence](coschedule-operator-20261025T151306Z.md):
stop isolated batch-size, native-worker and component-overlap searches. The finite
rosters have answered their questions. Further experiments should address actual
generated work and persistent operation, with one combined target handoff after
its unblocked integration code is complete. Learning quality, curriculum/replay
changes and multi-day training remain held for the user's later decision.

## Working experimental configuration

| Setting | Converged direction | Evidence / remaining boundary |
| --- | --- | --- |
| GPU ownership | One owner, unchanged FP32 eager model/backend | Actual HIP parity and fresh-process containment pass |
| Decoder bookkeeping | `hoisted/v1` | Accepted same-batch component comparison; no precision change |
| Inference capacity | 128 physical rows is the measured ceiling of this finite search | About90–91% sampled GPU busy; capacity replay uses32 unique inputs, not128 generated programs |
| Native service | Four bounded workers, numerical-thread policy1 | Accepted full18-row screen and144-job coexistence; immutable4s clocks |
| Scheduling | Deliberate GPU/native overlap; retain model-update drain barriers |21.72/24.99% paired component elapsed reductions; no asynchronous updates |
| CPU threads/quota | Torch2 intraop/1 interop, at most16 CPU equivalents total | Whole-cgroup repeat average about2 equivalents; raising CPU usage is not a throughput goal |
| Shared memory | Existing measured90%-physical/ancestor/engine ceiling with headroom |112.45GiB enforced ceiling on this workstation,4.42GiB sampled role upper peak; no reason to fill spare RAM |
| Training/data policy | Keep current architecture, batch, precision, corpus, optimizer and learning strategy | No learning-quality experiment is needed to settle this hardware question |

This table is an experimental implementation direction, not a production default
or an already executable batch128 training configuration. Original five-pair
adoption and scientific gates retain their requirements.

## Concrete integration gaps

`decoder/bigint_generation.py:sample_batch` and
`fresh/pipeline_jobs.py:validate_policy` currently accept only1/8/16/32, with explicit
capacity admission for32. The accepted128 replay bypasses real grammar sampling
and proves numerical/capacity behavior only. Do not replace those guards with an
unchecked boolean or change defaults on the strength of this replay.

`fresh/bigint_diagnostics.py:generate` groups by conditioning mode and prompt,
with one active portfolio per prompt. Its32-prompt diagnostic therefore cannot
fill a128-row batch, even if the numeric limit is raised. Replicating prompts as
capacity work would not close useful-throughput evidence. Reaching128 real rows
requires a prospective workload with enough distinct eligible visible prompts or
a separately specified scheduling policy; RNG mapping, per-prompt budgets, mode
barriers, source/selection identity and fairness must remain explicit. Batch128 is
a supported capacity target, not a requirement to manufacture extra attempts.
If the real workload naturally offers only32 concurrent rows, use its qualified
real batch and measure useful throughput instead of padding to a utilization goal.

The current co-scheduling experiment loads fresh GPU processes, workers and caches
for each trial. It does not retain a model/optimizer across diagnostic/update/
checkpoint boundaries. Its native work is an independent corpus, not the programs
just produced by the GPU. Existing tiny-model checkpoint/update fixtures are useful
prerequisites, not a substitute for that integrated path under sustained load.

Aggregate cgroup residency rises with retained fixture inputs/evidence while the
controller RSS remains small. `memory.stat` was not captured, so file-cache versus
anonymous residency is unresolved. A persistent soak needs those categories plus
device accounting and disk/checkpoint retention; unchanged per-trial timing alone
cannot establish leak freedom. Parent and child reports share one cgroup and must
not be summed. The4.42GiB figure is a sampled conservative upper observation, not
a guarantee for a persistent trainer or a larger generation workload.

## One remaining combined engineering investigation

Implement this bridge in existing generator/scheduler/trainer/checkpoint adapters;
do not add a second trainer or run another isolated capacity ladder. Before any
new target timing, freeze the versioned actual-row workload, source/weights/data,
capacity proof, policies, complete denominator, finite slot roster, time/disk
allowances and interpretation. Keep all accepted and negative prior scopes.

1. Qualify the real sampling/native path with ragged/finished rows, unchanged
   per-prompt seeds/tokens/sources/admission outcomes, positive pool quiescence and
   one GPU owner. Retain four-second native clocks and original60s production
   versus180s fixed-work diagnostic policies; no renewed budgets or silent larger
   lookahead. Explicitly test whichever real batch is actually admitted.
2. Measure actual generated/decoded/independently verified work under frozen
   full-denominator budgets, including incomplete and unstarted attempts. Report
   programs per total minute, queue/owner wait, GPU duty, whole-cgroup CPU and
   memory categories, rather than infer performance from supplied replay tokens.
   Preserve the original zero-complete32-prompt baseline. Do not invent an
   equal-work speedup when its common complete subset is empty. Original adoption
   still needs its own five fresh cold/warm pairs and stratum/coverage checks when
   those prerequisites can be met; these two component pairs cannot replace them.
3. In the same finite invocation, run a persistent engineering soak with actual
   disposable updates, generation, native verification, checkpoint/export and a
   stop/resume boundary using existing state. A proposed engineering duration is
   about two hours, subject to prospective budget/cost admission, not a new
   authority or a claimed calibrated runtime. Report throughput/queue/memory over
   time; verify no active/reserved work at updates/checkpoints, no duplicate
   debits on resume, bounded logs/artifacts/checkpoints and clean shutdown.
   Retain compact interval summaries and mandatory fault/job evidence within
   existing caps rather than copying this fixture's per-trial payload indefinitely.

Combine applicable current-image CPU/HIP fixtures, real-path measurement, soak,
resource/operations checks and one result ZIP. Serialize timing scopes; overlap
only the explicitly named GPU/native work within a shared envelope. Do not compare
arms running independently side by side on this shared-memory workstation.
Do not request a new target command until the integration and local fault tests
are complete. The accepted component archive needs no rerun.

If this integration test exposes a specific bottleneck or correctness failure,
fix that cause and preregister only the necessary follow-up. Otherwise freeze the
configuration and move to the longer-run decision; no automatic256/512-row search,
8/16-worker search, cache expansion, model/precision change or curriculum sweep.

## Longer training decision

Successful integrated useful throughput and persistent-state qualification can
support a separately proposed multi-day run of the chosen learning strategy.
Its exact initialization/data/seed/checkpoint/validation cadence, complete cost
projection (at most80% of its allowance), resource/disk retention, stopping and
resume rules still need a frozen prospective manifest and the existing applicable
gates. This does not require reopening curriculum design as a hardware experiment.
The user's intended organic/stochastic-versus-curriculum decision belongs after
that later long run. Neither this report nor `all-ready` launches it automatically.

Original counts after this review:78 complete/18 runtime pending/15 research held
or blocked.009:O004 and current-sourceT110 close at5014881. Original useful full-
pipeline confirmation, relevant operations/portability, complete costs and scientific
tasks remain independently visible in `task-evidence.json`.
