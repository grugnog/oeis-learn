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
  Integration of every neural/search timing and production stream escalation is
  still pending; these helpers alone do not close T096.
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
  tests/contract/test_foundation_qualified_cli.py
bash scripts/foundation/test_wheel.sh
```

The installed-wheel script builds a wheel, installs into a fresh environment,
changes directory outside the checkout, unsets PYTHONPATH, uses Python isolated
mode, imports packaged WAT, verifies installed source bytes and executes actual
runtime, census and measurement CLI fixtures. Native absence is an explicit local
skip. The new `foundation.yml` job requires the actual built native wheel and
runs G7/G8/G9 parity separately without native fallback. Live GPU acceptance stays
separate from ordinary CI.

Local limitations: no `lab-gpu`, Docker daemon or native extension; Linux boot ID
is not exposed, blocking the existing protected resume/accounting tests. Do not
remove the boot identity check to make this environment pass. Ubuntu CI owns
those checks. Final measured results are recorded after validation below.

## Remaining acceptance work

- T093: real indexed release source, metadata and structural split audit, followed
  by a release cohort freeze; fixture grouping is not a real corpus census.
- T094: mandatory Docker/native installed-wheel results must be inspected.
- T095: remaining historical command parity and end-to-end migration evidence.
- T096: integrate all phase/entropy/reward/solver/thermal telemetry and shared
  retry charging through real neural/search/stream callers.
- T097: genuine archive provenance linkage, baseline-body/native-word wider traces,
  sealed development prefix/continuation scoring and unique-group rescue counts.
- T098: coordinated qualified device/pilot execution under external hard deadlines;
  the durable decision controller and CPU fixture are not that live adapter.
- T099/T100: run the declared 128-prompt engineering session and one diagnostic/
  conditional paired pilot on qualified hardware. CPU speedups are not substituted.
- T101/T102: complete review disposition evidence and G0–G10 release authorization
  only after all mandatory repairs and hardware/source measurements are verified.
