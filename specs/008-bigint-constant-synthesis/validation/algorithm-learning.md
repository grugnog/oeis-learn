# Phase 2 data, model and policy evidence (T037–T057)

Status: CPU algorithm suites pass. This is implementation qualification, not a
training result or evidence of transfer improvement. Native fitter execution and
actual HIP are separate pending gates. Production remains disabled.

## Reproduction and recorded results

The local environment is Python3.12.14 / Torch2.9.1+cpu. The final Phase 2 suite
plus the affected model-construction and build-cache regressions passed
**147 tests with32 native/artifact skips** (179 total, zero failures). The image
qualification utility promotes missing native artifacts/skips to failure.
Traceability/handoff checks passed **36 tests** after rebinding the 007 rolling
source identity without promoting historical evidence.

An additional retained workflow/diagnostics/provenance regression run passed
**55 tests and failed9 existing pool-builder integration cases**. Eight fail at
`/proc/sys/kernel/random/boot_id`, which this workspace does not expose; the ninth
is the CLI exit-code consequence of that same unavailable clock. These are
recorded failures, not counted as passing qualification. The pool builder's
production clock and resume policy are unchanged; its container checks remain
required. Python F/E9, workflow YAML parsing, build-shell syntax, helper CLI help
and git whitespace checks passed. No extension hooks are registered.

[The machine-readable evidence](algorithm-learning-local.json) records test
counts, nonpassing cases, JUnit hashes, environment, frozen fixture hashes and
every model/data/decoder/policy source hash. The source snapshot identity is
`sha256:3828d8315c9aeeef336030613a61bc67510b4b1a817dc362e4e06f4e85f58469`.
The 007 Python implementation binding is
`sha256:b85bf3874797817a09f61c78b6e5143786f3b578fe68f3b8b589a73d6dcbf9ec`.
These bind source bytes, not an unexecuted native image.

```bash
# Local Python-only repetition: native cases skip if artifacts are absent.
PYTHONPATH=src OMP_NUM_THREADS=2 python - <<'PYTEST'
import runpy, subprocess, sys
suite = runpy.run_path("scripts/qualify_bigint_phase2.py")["CPU_TESTS"]
raise SystemExit(subprocess.call([
    sys.executable, "-m", "pytest", "-q", *suite,
    "tests/unit/test_foundation_ci_images.py", "tests/unit/test_bootstrap_models.py",
]))
PYTEST
```

Use the [portable helper](../quickstart.md#phase-2-qualification-available-now)
on the built image to require real native artifacts and separately actual HIP.
The temporary helper does not expose Phase 3 public CLI suites or launch recipes.

## Fixed semantics covered

- Four input groups have explicit fusion boundaries: magnitude before FiLM,
  modular gamma/beta including bias, difference/valuation gamma/beta including
  bias, and the exact-byte branch. Removed paths are zero, with no implicit
  rescaling. Retrained arms freeze the same excluded modules, including byte
  gate; checkpoint masks/modes and active/total/trainable counts are explicit.
- The decoder recipe is d256,4 heads,4 encoder+4 decoder layers, eager attention,
  token scale1, position scale16 and dropout0. Common decoder initialization is
  independent of encoder size. This is a **joint gain/dropout recipe**, not an
  isolated causal finding about either setting. CPU tests cover all
  correct/shuffled/absent teacher-forced logits and actual generated-token
  cached/full parity. HIP tests now exercise both as well as finite large-input
  backward/update and device/position-buffer movement; those tests are unrun here.
- Counter-based A–D families retain typed connected expressions, explicit shared
  parameter provenance,16 variants and disclosed bounded priors. There is no
  simplifier; trapping dead expressions remain exclusions. Canonical family
  identities preserve operators/control and parameter-sharing structure.
- Paired target views admit both concrete/parameterized bodies at the same
  1024-token bound or exclude both. Complete100-index reference values and all
  observed20-term windows with relative index gaps group behavior/shift overlap.
  This is observed-horizon grouping, not universal equivalence proof. Values up
  to the full intermediate resource width hash as canonical decimal strings,
  avoiding Python's global decimal digit guard.
- Diverse32 requires at least4 families, at least16 non-affine examples and at
  most8 offsets. Transfer selection produces512/128/128 or explicit shortfall;
  parameter-dev families occur in training, structural-dev families do not, and
  behavior groups never cross splits. Constant/offset caps are10%. The positive
  2,816-row fixture isolates split policy with synthetic cubic rows and a codec
  stub; real codec admission is tested separately. It does not claim a trained
  model or a generated real-data pool met these gates.
- Eight fixed non-neural templates accept no prompt metadata/output input and
  round-trip the real grammar. Search allocates one greedy/seven stochastic
  attempts,1024 tokens and60 seconds aggregate, with explicit beam replacement
  accounting and independent-check allowance. Incomplete output is retained as
  bounded tokens, never repaired. Same-candidate rescue binds original identity.
- Visible-only selection seals the chosen source before full100-index scoring;
  production and independent reference must both bind that source and all
  actual indices. A20-term prefix fit with wrong continuation scores zero.
  Checkpoint/curriculum selection uses only full-denominator visible fit, then
  complete teacher-forced loss/update ties; changed panels are rejected.
- Curriculum advances at80% twice with95% coverage and retains at least25%
  earlier examples. Stopping patience, replay replacement/cursor and curriculum
  survive restore. Replay requires independently verified100-term records from
  a frozen training roster, retains source/assignment/profile/track, caps4096
  records and25% of later batches, and is disabled in representation comparisons.
- Transfer gates require the exact equal-budget fitted portfolio, fixed128
  denominators, the three declared paired seeds, at least8/128 against every
  control in two seeds and mean6.25pp against each,95% coverage and at most4/128
  mean parameter-dev regression.2000 seeded family bootstrap resamples are
  descriptive; seed spread is reported separately. Screen projection includes
  every scheduled panel, full development losses, end controls, fitting,
  reference, hidden scoring and checkpoint costs; exceeding80% rejects the recipe.
- Contract/batch/ledger/final-plan transitions preserve debits and all arms on
  interruptions, reject stale generations and partial/provisional admission,
  and never restore final-cohort freshness. Ancestry binds actual shared
  foundation checkpoint/recipe hashes and compute, requires reviewed assistance,
  rejects strict descendants of assisted parents and preserves exposed groups.
- Alternative archives retain up to8 verified structural/cost Pareto alternatives
  with deterministic ties and omission records. Selecting a training
  representative does not delete alternatives. Finite match, conjecture, bounded
  verification, proof evidence and novelty review remain distinct; historical
  unreplayed evidence never executes retired solvers.

## Pending acceptance

T025/T036/T058 require the new native driver/core tests and independent admission
results. The prior merged image's success is recorded separately. Actual HIP
checks remain pending; return the helper's summary, XML, logs and weight-update
record after running it on ROCm. Any semantic failure reopens the corresponding
Phase 2 implementation task. No new PR checks were monitored or awaited.
