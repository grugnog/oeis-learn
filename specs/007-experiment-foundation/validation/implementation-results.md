# Implementation results

Phase 10 implementation is in [PR #11](https://github.com/grugnog/oeis-learn/pull/11). [The review report](phase-10-review.md) records the completed integration, tests and explicit acceptance limits. [The disposition map](phase-10-dispositions.json) preserves all 68 original findings and binds the current package implementation hash.

The remaining historical-wrapper, telemetry, wider-trace and live-measurement software paths have been implemented. They include source-file replay, shared measured updates, charged stateful retries, provenance-linked tracing, isolated Docker roles, strict GPU smoke, frozen engineering comparisons, a single conditional pilot and replayed G0–G10 readiness. Diagnostic hypotheses now pass the same archive admission and original holdouts as self-training. Encoder triggers probe the actual frozen baseline's input features; legacy feature aliases cannot spuriously trigger a comparison for the repaired baseline. Repeated installed-wheel verification allocates fresh run directories while preserving earlier checkpoints. Historical resume selects epochs numerically. Timed-out workers are reclaimed before replacement without regenerating requests; container and client reclamation share one two-second allowance.

Current executable code is `ba89b9694128775cd136b890f77a249b650a70ae`. Current code validation:

- [CPU/native review CI](https://github.com/grugnog/oeis-learn/actions/runs/38088372626): **712 passed, 56 explicit capability skips**; compile/lint, Spec Kit structural checks and native qualification passed.
- [Installed-wheel/native/Docker CI](https://github.com/grugnog/oeis-learn/actions/runs/38088372595): **6 installed-wheel cases passed with no skips**, **69 phase 10 regressions passed**, and **8 actual Docker role tests passed**. Repeated train/resume/search verification preserves independent checkpoint directories.
- G7 and G9 passed; G8 software passed. G8 hardware and G10 live/source acceptance remain pending. These results do not qualify GPU learning or performance.

Local focused validation recorded **50 passed, 1 Docker capability skip**, then **16 passed, 1 skip** for final repair regressions and **22 passed** for source/readiness tests. Local source-distribution and wheel builds preserved all **388** qualification source/resource bindings. The earlier restricted-host broad run and its environmental limitations are recorded in the review report. CPU fixtures cannot establish GPU performance or learning benefit.

T093 and T095–T100/T102 remain half-checked for actual-source, trained-model or GPU acceptance. T094 packaging and T101 reporting remain checked. The actual release corpus is unavailable here and is a data dependency; its audit runs on CPU once supplied. The independently named bounded capacity pilot uses FF2048 to reach the 25M class; the existing qualified FF1536 constructor remains supported. Forward/backward interface fixtures execute for both encoder representations and positional variants even when their comparison is not triggered.

Phase 11 has not started. Full 007 release qualification remains pending.
