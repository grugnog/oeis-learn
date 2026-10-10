# Implementation results

Phase 10 implementation is in [PR #11](https://github.com/grugnog/oeis-learn/pull/11). [The review report](phase-10-review.md) records the completed integration, tests and explicit acceptance limits. [The disposition map](phase-10-dispositions.json) preserves all 68 original findings and binds the phase 10 package implementation hash.

The remaining historical-wrapper, telemetry, wider-trace and live-measurement software paths have been implemented. They include source-file replay, shared measured updates, charged stateful retries, provenance-linked tracing, isolated Docker roles, strict GPU smoke, frozen engineering comparisons, a single conditional pilot and replayed G0–G10 readiness. Diagnostic hypotheses now pass the same archive admission and original holdouts as self-training. Encoder triggers probe the actual frozen baseline's input features; legacy feature aliases cannot spuriously trigger a comparison for the repaired baseline. Repeated installed-wheel verification allocates fresh run directories while preserving earlier checkpoints. Historical resume selects epochs numerically. Timed-out workers are reclaimed before replacement without regenerating requests; container and client reclamation share one two-second allowance.

Phase 10 executable code is `ba89b9694128775cd136b890f77a249b650a70ae`. Phase 10 code validation:

- [CPU/native review CI](https://github.com/grugnog/oeis-learn/actions/runs/38088372626): **712 passed, 56 explicit capability skips**; compile/lint, Spec Kit structural checks and native qualification passed.
- [Installed-wheel/native/Docker CI](https://github.com/grugnog/oeis-learn/actions/runs/38088372595): **6 installed-wheel cases passed with no skips**, **69 phase 10 regressions passed**, and **8 actual Docker role tests passed**. Repeated train/resume/search verification preserves independent checkpoint directories.
- G7 and G9 passed; G8 software passed. G8 hardware and G10 live/source acceptance remain pending. These results do not qualify GPU learning or performance.

Local focused validation recorded **50 passed, 1 Docker capability skip**, then **16 passed, 1 skip** for final repair regressions and **22 passed** for source/readiness tests. Local source-distribution and wheel builds preserved all **388** qualification source/resource bindings. The earlier restricted-host broad run and its environmental limitations are recorded in the review report. CPU fixtures cannot establish GPU performance or learning benefit.

T093 and T095–T100/T102 remain half-checked for actual-source, trained-model or GPU acceptance. T094 packaging and T101 reporting remain checked. The actual release corpus is unavailable here and is a data dependency; its audit runs on CPU once supplied. The independently named bounded capacity pilot uses FF2048 to reach the 25M class; the existing qualified FF1536 constructor remains supported. Forward/backward interface fixtures execute for both encoder representations and positional variants even when their comparison is not triggered.

## Phase 11 cross-cutting review

[The phase 11 review](phase-11-review.md) records repaired operational documentation,
original-input/task reconciliation and atomic bounded handoff/replay. See the
[handoff](handoff.md) and [current 68-input dispositions](phase-11-dispositions.json).
Earlier validation above remains historical; full 007 qualification stays pending.

Local focused regression command:

```sh
OMP_NUM_THREADS=2 .review-venv/bin/python -m pytest -q -ra tests/unit/test_foundation_handoff.py tests/contract/test_foundation_qualified_cli.py --junitxml=phase11-evidence/focused-final.xml
.review-venv/bin/ruff check --select F,E9 src/oeis_learn/evaluation/handoff.py src/oeis_learn/evaluation/review_dispositions.py scripts/verify_phase11.py scripts/archive_phase11.py tests/unit/test_foundation_handoff.py
```

Result: **24 passed, 3 explicit native/gate capability skips**; lint passes.
The locked CPU environment uses Python 3.12.14 and torch 2.9.1+cpu; dependencies
are exported from `uv.lock` with torch pruned and installed with hashes/no-deps.
Docker, Cargo and `lab-gpu` are unavailable locally. No host compiler/ROCm
installation or production boot/deadline relaxation was performed.

Actual CPU quickstart command: `OMP_NUM_THREADS=2 .review-venv/bin/python
scripts/verify_phase11.py --output phase11-evidence/quickstart`. The workspace lacks
`/proc/sys/kernel/random/boot_id`: cohort/census/measurement/handoff execute,
pool rejects and dependent training/evaluation remain explicit failures. The
initial bounded archive had 27 files / 77,160 bytes and pending G0–G10. Its hash
is historical after subsequent code remediation; Linux CI executes the final
full quickstart and fresh replay. Missing GPU/source/trained-model acceptance
keeps T104/T105 and earlier acceptance tasks half-checked.

Additional local validation:

- Broad command: `OMP_NUM_THREADS=2 .review-venv/bin/python -m pytest -q -ra tests/unit/test_foundation_*.py tests/contract/test_foundation_*.py tests/integration/test_foundation_*.py --junitxml=phase11-evidence/broad.xml`: **699 passed, 56 capability skips, 22 failures, 12 setup errors**. Nineteen failures and all twelve setup errors directly name the unavailable Linux boot ID; three CLI assertions reflect the corresponding invalid-input exits. These are not marked passing. Current Linux CI must supply acceptance.
- `python -m build --sdist --wheel --no-isolation --outdir phase11-evidence/dist` and an extracted-sdist wheel rebuild succeed. The wheel verifies **395** source/resource bindings, including exact original-input and task-checklist governance bytes. Native compilation stays in Docker CI.
- Official upstream Spec Kit at `fe1d00e3ccaf495880aaf90fb0e17679e82f065b`, CLI 1.0.7: `specify version`, `specify check`, and `SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation specify integration status` succeed. Integration is Pi, with zero modified/missing/invalid/unchecked managed files; local coding-agent CLIs are unavailable and not installed for this review.
- `SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation bash .specify/scripts/bash/check-prerequisites.sh --json --require-spec --require-tasks --include-tasks` passes. `python specs/007-experiment-foundation/validate_artifacts.py` passes: 52 requirements, 108 tasks, 68 original findings, 100% structural requirement/task coverage, 82 checked links and 14 negative schema cases. Neither command executes runtime gates.
- Read-only semantic review compares constitution/spec/plan/contracts/tasks/source findings; no new constitution conflict or unmapped requirement/task. Findings and remaining runtime acceptance are summarized in the review. Requirements/design checklists remain untouched.

Current Linux/native/wheel CI results will be recorded below after execution;
no future result is treated as passing evidence.
