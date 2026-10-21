# Diverse32 failure diagnosis and evidence reliability

The reviewed T096 attempt remains complete with a negative outcome. Its
31/32 rescue successes meet the success threshold, but 31/32 complete rescue
observations do not meet the frozen coverage requirement. The incomplete
prompt has a successful raw selection and equal-budget selection. The bounded
bundle lacks its detailed fitting trace, so the cause is not established.

This follow-up implements the user-approved diagnostic plan within Spec 008.
Evidence semantics return to Phase 2 ownership (T043/T047/T058); persistence
and export changes cover T069/T072/T077. No solver subset, candidate ranking,
raw/rescue fallback, model, budget, threshold or experiment is changed. The
original completed run is never resumed or automatically retried. Changed
source requires new qualification identities before future experiments.

## Work and acceptance

- [X] D001 Reproduce deadline and cross-arm completeness defects with focused
  tests; define separate generation, fitting and scoring evidence per arm.
- [X] D002 Correct accounting without changing visible-only selection; retain
  compact, reason-bearing evidence for every prompt, including failed cases
  outside the detailed-example allowance.
- [X] D003 Add a standard-library, read-only collector for the original run's
  sealed selections, permitted logs and retained fit evidence. Report missing
  records explicitly and produce a bounded archive without weights or private
  scoring truth.
- [X] D004 Provide a bounded candidate replay path with a separate analysis
  identity, original resource limits, and immutable parent records. Supply
  portable container commands; decoding replay on HIP is conditional on the
  collected evidence and is not an automatic qualification rerun.
- [X] D005 Run focused CPU regressions, document remaining hardware/data
  requirements, and leave CI unmonitored after pushing.

Completion means failures can be attributed to a stage and outcome, expired
scoring cannot count as complete, and unrelated controls cannot contaminate
another arm's coverage. It does not mean T096 passes. If diagnosis establishes
a wider solver/search limitation, propose a separate design and preregistered
experiment rather than tuning against this failed prompt's hidden outcomes.

## Collect and replay the original failed prompt

Run from the repository root after updating the branch. These commands require
no image rebuild and no GPU. Collection reads the original run and its
`run-state` siblings, verifies sealed selection hashes, records available
generation/scoring logs and fit-cache matches, and reports missing detailed
traces explicitly. It never opens private truth or checkpoint weights. Inputs
remain on the machine where the original run executed.

```bash
diagnostic_out="tmp-analysis/008-diverse32-diagnosis-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/diagnose_bigint_run.py collect \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --prompt sha256:02aa8138af680017900e08e7c01148a4128583e1d0778195d05f4119c337b6b9 \
  --output "$diagnostic_out/collection"
python3 scripts/diagnose_bigint_run.py replay --engine podman \
  --evidence "$diagnostic_out/collection" \
  --output "$diagnostic_out/replay"
```

Share `collection.zip` and `replay.zip` under that directory. Substitute Docker
consistently if appropriate. Replay defaults to the original immutable image;
it mounts only the collected visible case, this utility and a fresh output
directory. The utility itself is mounted from the current checkout, while
fitting, production execution and the reference load from the pinned image.
The original greedy source, seed, abstraction policy, resource/service limits
and four-second raw/fitting ceilings are retained. There is one fitting call,
with a fresh cache, within a30-second work allowance. The separate host ledger
reserves that work plus15seconds startup and30seconds cleanup before launch;
the nested work ledger records its30seconds without refund. Timeouts and
cleanup errors still produce a report archive with the full reservation charged.
`.runtime` caches never enter the archive. Every output directory must be fresh.

This is an isolated visible-candidate analysis, not an exact replay of original
queue/cache timing, a new decoding attempt, a continuation score, a corrected
T096 decision or a promotion. Image/runtime differences are explicit. Successful
fits alone were cached by the original client; missing failed-fit traces are
not evidence that fitting completed or that a particular failure occurred.

## Diagnosis decision

The first review should distinguish a missing rescue selection from a failed
100-index scoring call, then identify generation, native fit, independent
admission or deadline failure. If no original failed-fit receipt survives,
use the bounded visible replay as new diagnostic evidence and retain that
limitation. A changed implementation cannot manufacture historical completeness.
Only if this still requires decoding should a separately scoped fixed-checkpoint
HIP analysis be designed, with source/checkpoint identities, fixed prompts,
attempts and a finite child ledger. No retraining is part of this follow-up.

A larger solver subset, different raw/fitted selection policy or scheduler
redesign requires a separate design and preregistered experiment after this
diagnosis. The existing source-bound cost/HIP receipts remain historical;
future qualification must bind the changed source and image.

## Local implementation validation — 2026-10-21

The final CPU regression pass completed182 tests with zero failures/errors,
three explicit packaged-native skips and one GPU deselection. The skipped
cases require actual native scoring/preparation/fit-execute artifacts. No
Docker/Podman engine, packaged native image or HIP accelerator is available in
this implementation environment; mocked replay/container tests are not native
or hardware qualification. The original run's private role files are not
available here, so its exact failed fitting/scoring operation remains pending
the collection/replay commands above.

The suite covers deadline-expired scoring, full-horizon partial results,
legacy evidence, isolated greedy-versus-stochastic completion, independent
admission exhaustion reasons, all32 compact receipts under bundle pressure,
host generation/scoring evidence copying, core-summary bounds, parent
immutability, sealed-file tampering, symlink/traversal rejection, truth/weight
exclusion and replay timeout/missing-summary/cleanup failure accounting.
The regressions are registered in both qualification utilities; the standard
bigint CPU suite also discovers them automatically.

Ruff F/E9, formatting, standard-library-only utility help, documented Bash
syntax, Spec Kit prerequisites, the16/16 requirements checklist and whitespace
checks pass. See the [CPU validation receipt](phase-5-diagnostic-followup-cpu.json)
for the source and JUnit identities. The original T096 receipt is unchanged;
no new run, hidden scoring or final-cohort access occurred. CI is left running
without monitoring.
