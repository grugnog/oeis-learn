# Phase 10 review and remediation

Scope: T093–T102 on top of merged phase 9 (`8187772`). Phase 11 is unchanged.
This review is **partial**, not full 007 qualification. CPU fixture timings and
record schemas do not establish GPU feasibility, performance or learning benefit.

## Findings and repairs

- Census coverage included short series in complete-horizon denominators, took
  absolute values before signed-width testing, and labeled growth from the last
  value. The shared read-only source policy now uses exact complete 20/50/100/120
  populations, separates available prefixes, handles signed minima correctly,
  retains offsets/keywords, and describes the full observed series without an
  asymptotic claim. The database path is explicit; malformed rows are reported.
- Extraction silently deleted malformed terms and inferred authoritative-looking
  `nonn`/`tabl` tags. Malformed terms now fail instead of compressing indices;
  inferred tags remain explicitly separate from unavailable source keywords and
  offsets. No source download or real release cohort was performed in this review.
- Historical baseline cohort IDs remain reproducible. The explicit release profile
  `prefix20_total100_affine_v2` adds exact nonzero rational affine/shift grouping,
  verified over the full overlap, with replayable rational witnesses and all-member
  reserved-prefix membership. Admission accepts this version explicitly.
- Warmstart repeated a handwritten family in its transition gate. It now samples
  current model weights with explicit seeds and restores module modes. A missing
  requested parent is an error. Identical successes have zero reward variance,
  just as identical failures do. Prefix-only legacy diagnostics cannot authorize
  Run 011; historical files are inventoried and remain `legacy_unverified`.
  Gate-only warmstart skips both training stages and preserves nested module
  modes. Run 011 lifecycle completion and Run 010 summaries no longer grant
  qualification. Run 008/009/010 no longer submit fixed passing readiness scores;
  missing/nonfinite observations stay null and fail even upper-bound gates.
  Existing numeric readiness reports remain valid; null requires a failed gate
  and explanatory diagnostics under the extended legacy report schema.
- Gate identity calculation depended on source checkout paths. The wheel embeds a
  build snapshot of code, owning tests and locks; installed code/resources are
  checked against it before identity reuse. This snapshot creates no PASS evidence.
- Added frozen experiment defaults, exact priority and paired useful-effect rules,
  a durable single-contrast issuance schedule, immutable measure inputs/results,
  and a bounded actual CPU cache fixture. Missing trigger diagnostics stay
  inconclusive. A crash cannot regenerate/refund issued work or launch another
  contrast. The live GPU/pilot adapter remains pending and fails explicitly.
- Added actual fuel distribution summaries, synchronized phase timers and
  exhaustion-only 100k/250k/1M fresh-call escalation sharing the 50M aggregate cap.
  The production pipeline separates prepare/compiled/reference/overhead time;
  policy updates synchronize encoding/generation/grounding/runtime/forward/backward
  boundaries and record actual entropy, reward variance, solver durations/outcomes,
  per-term fuel and resources. Run 011 aggregates actual variance instead of a
  pass-rate proxy and reports absent filter/sensor observations explicitly.
  Complete search/stream/SFT integration and sustained hardware observations are
  still pending; helpers and CPU fixtures alone do not close T096.
- Added externally bounded logical wider-reference tracing (4096 bits/50M steps),
  explicit unavailable strata, rescued-output counts and independent in-range-only
  checked/unchecked alternating benchmarks. Unchecked compiled bytes never leave
  the benchmark worker. This diagnostic does not verify origin labels as archive
  evidence; baseline-body/native-word tracing and sealed development scoring are
  pending and unavailable counts remain null.

## Validation and reproduction

```sh
SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation \
  bash .specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks
python specs/007-experiment-foundation/validate_artifacts.py
python -m pytest -q tests/unit/test_foundation_source_policy.py \
  tests/unit/test_foundation_decisions.py tests/unit/test_foundation_measurement.py \
  tests/unit/test_foundation_legacy_wrappers.py \
  tests/integration/test_foundation_overflow_diagnostics.py \
  tests/contract/test_foundation_qualified_cli.py \
  tests/integration/test_egca_training_step.py \
  tests/unit/test_readiness_policy.py tests/contract/test_readiness_report_contract.py \
  tests/integration/test_readiness_override.py tests/integration/test_transition_gate.py
bash scripts/foundation/test_wheel.sh
```

The installed-wheel script builds a wheel, installs into a fresh environment,
changes directory outside the checkout, unsets PYTHONPATH, uses Python isolated
mode, imports packaged WAT, verifies installed source bytes and executes actual
runtime, census and measurement CLI fixtures. Given actual G7/G8/G9 artifacts,
standalone installed tests also perform two diagnostic training updates across
resume, bounded search, discovery from the admitted generated archive, a fresh
isolated certificate check and rejection of stale evidence/GPU activation. The
scripts for prior CLI acceptance delegate to this same packaged fixture service.
Native absence and unavailable gate artifacts are explicit local skips. The new
`foundation.yml` job requires the actual built native wheel and executable gates;
all six installed-wheel cases must execute without skips. Live GPU acceptance
stays separate from ordinary CI.

Local limitations: no `lab-gpu`, Docker daemon or native extension; Linux boot ID
is not exposed, blocking the existing protected resume/accounting tests. Do not
remove the boot identity check to make this environment pass. Ubuntu CI owns
those checks. Final measured results are recorded after validation below.

## Remaining acceptance work

- T093: real indexed release source, metadata and structural split audit, followed
  by a release cohort freeze; fixture grouping is not a real corpus census.
- T094 is complete: final revision Docker/native and all six installed-wheel cases passed.
- T095: remaining historical command parity and end-to-end migration evidence.
- T096: integrate all phase/entropy/reward/solver/thermal telemetry and shared
  retry charging through real neural/search/stream callers.
- T097: genuine archive provenance linkage, baseline-body/native-word wider traces,
  sealed development prefix/continuation scoring and unique-group rescue counts.
- T098: coordinated qualified device/pilot execution under external hard deadlines;
  the durable decision controller and CPU fixture are not that live adapter.
- T099/T100: run the declared 128-prompt engineering session and one diagnostic/
  conditional paired pilot on qualified hardware. CPU speedups are not substituted.
- T101 disposition reporting is complete: every original input remains traced
  with tested, partial, unavailable or earlier-task evidence and unchanged design
  rationale. This reporting does not close its referenced incomplete tasks.
- T102: full G0–G10 release authorization still needs the earlier coordinated
  run/containment integration, mandatory remaining repairs and hardware/source
  evidence. Installed CPU CLI acceptance is complete, not hardware promotion.

## Requirement and input evidence map

| Tasks / requirements | Executed software evidence | Remaining acceptance |
| --- | --- | --- |
| T093; FR-024/007/009 | Exact horizon, signed bounds, indexed gaps, authoritative keyword arrays and affine membership fixtures in `test_foundation_source_policy.py` | Real release source and structural split audit |
| T094; FR-034/023/017 | Pinned Docker build, mandatory native parity, G7/G8/G9 and clean installed-wheel workflow | Complete software acceptance; live GPU separate |
| T095; FR-034/011/020 | Model-sampled warmstart, gate-only execution, nested modes, legacy lifecycle, missing explicit resume and null readiness regressions | End-to-end parity of remaining historical launch configurations |
| T096; FR-035/016/018 | Actual pipeline and policy timings; entropy/variance/solver evidence; fuel distributions, retry accounting and expired-deadline regressions | Complete search/SFT/stream integration and actual sustained device observations |
| T097; FR-037/002/003/008 | Independent wider pure-IR trace, signed overflow rescues, cap/unknown behavior, separately timed in-range helpers | Genuine archive linkage, baseline-body/native-word traces and sealed development scoring |
| T098–100; FR-036/038/014/016 | Frozen defaults, priority/effect boundaries, interruption charging, no second contrast and immutable measure CLI regressions | Coordinated live adapter and qualified engineering/conditional pilot |
| T101–102; FR-038/034/019 | All 68 original inputs mapped in [phase-10-dispositions.json](phase-10-dispositions.json); installed CLI and fail-closed readiness tests | Full G0–G10 evidence and remaining mandatory integration |

The machine-readable disposition map binds the original review inventory hash,
preserves evidence-based design rationale and lists actual task states. A checked
earlier task is reported only as checklist completion, never inferred release
qualification. All 20 recommendations, A01–A19, P01–P17 and D01–D12 remain traced;
only the three original LODA dispositions remain deferred. Unmeasured branches are
inconclusive and no cache, precision, architecture or learning adoption is claimed.

## Measured validation results

Final code revision: `f50f144bf78a38605b16bdc945c305526f0eecd1`.
The later commit records reports/task states and does not change qualified code,
owning tests or locks.

- [Ubuntu CPU and native run](https://github.com/grugnog/oeis-learn/actions/runs/38074159609):
  **680 passed, 54 documented capability skips** in the full CPU foundation
  suite. Its separate native job passed mandatory Rust/native parity, G7/G8/G9
  and the public diagnostic learning/discovery acceptance scripts.
- [Docker/native clean-wheel run](https://github.com/grugnog/oeis-learn/actions/runs/38074159601):
  **all 6 installed cases passed with zero skips**, including two train updates
  across resume, search, discovery, fresh certificate checking and measure. Actual
  native evidence was mandatory; source-tree imports/editable installs were absent.
  G8 software is PASS; G8 hardware remains PENDING. G10 is not promoted.
- Final local focused suite: **56 passed, 3 explicit native/gate-evidence skips**.
- Final local isolated installed wheel: **3 passed, 3 explicit native/gate-evidence
  skips**.
- Full local foundation run before the later fixes: **635 passed, 52 capability
  skips, 22 failures and 12 setup errors** caused by the absent Linux boot ID.
  This is not a passing local full-suite claim. The guard is preserved; actual
  Ubuntu CI above passes those checks.
- Compile checks, focused Ruff F/E9 and Spec Kit structural validation passed.
  Structural validation reports 52 requirements and 68 input items; it explicitly
  executes no runtime gates and ratifies no constitution.

T094 packaging/native CI and T101 disposition reporting are complete. T093,
T095–T098 and T102 retain partial markers. T099/T100 remain unavailable pending
qualified hardware and coordinated execution. No known pending repair is counted
as fixed by disabling it. Phase 11 is unchanged.
