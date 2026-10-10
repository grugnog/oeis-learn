# Phase 10 review and remediation

Scope: T093–T102 on merged phase 9 (`8187772`), in [PR #11](https://github.com/grugnog/oeis-learn/pull/11). Phase 11 is unchanged. Software implementation and hardware acceptance are separate; this report does not claim full 007 qualification.

## Implemented integration

- Indexed source import binds a release manifest, exact indexed term files, authoritative metadata and file hashes. Release audit reopens those files and replays exclusions, affine/shift grouping, all-member reserved prefixes and the frozen training archive's structural split. Stripped samples cannot supply missing offsets or keywords. Empty training archives cannot pass.
- Existing Run 008/009/010 launchers use shared configuration, explicit-parent, actual readiness and lifecycle services. Resume selects the numeric latest epoch and checks its resolved parent once. Shared SFT updates now cover ordinary SFT, embedding warmup, bridge and verified replay. Historical configurations and `legacy_unverified` status remain explicit. Reporting validates authoritative schemas and includes those schemas in the installed wheel.
- Neural/search/runtime operations record actual synchronized phases, predictive entropy, population reward variance, solver observations, fuel distributions and resources. Existing AMD hwmon is read when uniquely available; unsupported sensors stay unavailable. Search/policy use deterministic exhaustion-only retries. Stateful streaming restores the pre-call state while retaining every failed attempt's fuel and reference steps.
- Wider tracing supports logical IR and baseline WAT bodies. Mathematical wide violations and native word wraps are separate. Raw generic/model samples can be provenance-replayed before strict admission, so overflow does not exclude the population being studied. Archive entries replay shared admission checks. Sources are sealed before development truth is read; correlated implementations are deduplicated for rescued-program and unique added-group counts. Unchecked bytes stay inside the independent benchmark worker.
- A host Docker coordinator owns one bounded main role and at most eight one-GiB CPU workers, with no network, Docker socket, shared writable repository or worker GPU capability. Enumerated views separate learner, generator and development evaluator. Persistent native workers retain compiled caches; timed-out workers are reclaimed before replacement, and failed requests are never regenerated. Worker reclamation shares one two-second allowance. Host deadlines, reserve/quota monitors, exclusive-controller checks, durable request/result binding and cgroup reclamation remain outside the learner.
- `coordinated-smoke` performs the strict three-update GPU smoke inside those boundaries; resume retains the original run/checkpoint/budget journals. `qualify-baseline` binds actual G0–G6 artifacts to the current checkpoint, including native conformance, executed containment, measured GPU updates and owning suites. Lifecycle labels cannot substitute for these records.
- `prepare-measurement` freezes exactly 128 generic prompts, training/structural-validation views, development-only evaluator data, models, code/runtime/image identities and prerequisite evidence. It requires replayed G0–G8 evidence; preparation runs in the matching image. The host repeats hashes and admission replay, then the GPU probe independently replays the gates and checks actual code/device identity. No final-test truth enters the bundle.
- Live engineering executes cached/uncached and batch/attention comparisons for seeds 7001–7003, fixed 10 warmups/30 alternating repeats, actual 4/8-worker and fuel horizons, grammar comparison, 32-update precision fixtures, ten-minute steady-state segments and bounded 128/256/512 relation corpora. Each corpus's 60-second allowance includes construction and features. Unsupported context or ABI horizons are explicit shortfalls; a grammar microbenchmark alone cannot authorize end-to-end adoption. No speedup is claimed here.
- The conditional pilot charges the diagnostic and every setup/update/checkpoint/generation/development operation within the frozen caps. Its encoder trigger probes the actual frozen baseline before learned projection. It chooses one contrast, uses matched fresh seeds and common 15/30/45/55 evaluations, counts diagnostic hypotheses and admits self-training through the real archive and stores optimizer, reference, curriculum, replay and all active RNG state. Effects use actual charged arm duration with solved-count nonregression; first-pair and confirmation rules are durably enforced. Missing matched structural coverage remains inconclusive. Failures are charged immediately and cannot reissue an arm.
- Release readiness replays supplied G0–G10 evidence and distinguishes baseline-ready from full-release-qualified. Current dispositions, installed-wheel/native acceptance, Docker role evidence, source audit and measured live results are mandatory. Measurement replay verifies the role-owned results, immutable scheduled evaluations, checkpoint bytes and durable pilot events.

## Current validation

Executable code revision: `ba89b9694128775cd136b890f77a249b650a70ae`. The final evidence update changes documentation only; package/runtime sources remain those tested below.

- [CPU/native review CI](https://github.com/grugnog/oeis-learn/actions/runs/38088372626): **712 passed, 56 explicit capability skips**; compile/lint, Spec Kit structural checks and native qualification passed.
- [Installed-wheel/native/Docker CI](https://github.com/grugnog/oeis-learn/actions/runs/38088372595): **6 installed-wheel cases passed with no skips**, **69 phase 10 regressions passed**, and **8 actual Docker role tests passed**. Repeated train/resume/search verification preserves independent checkpoint directories.
- G7 and G9 passed; G8 software passed. G8 hardware and G10 live/source acceptance remain pending. These results do not qualify GPU learning or performance.

Local focused checks recorded 50 passed/1 Docker skip, then 16 passed/1 skip for final repair regressions and 22 source/readiness tests passed. At `fad5822`, a source-distribution round trip rebuilt a wheel and verified all 388 qualification source/resource bindings, including governance, schemas and owning tests. Current installed-wheel acceptance is independently recorded above.

The earlier restricted-host broad run recorded 654 passed, 55 skips, 25 failures and 12 setup errors. Most failures involved its unavailable Linux boot ID or worker-startup deadlines. A real telemetry/replay-test regression was repaired. Production boot guards/deadlines were retained; the final Ubuntu CPU/native/container checks above supply executable acceptance.

The closed pilot interfaces have finite CPU forward/backward fixtures for both encoders, RoPE, prefix-ancestor positions and the bounded FF2048 capacity variant. The existing qualified FF1536 constructor is retained. Diagnostic hypotheses must pass archive admission and the original holdouts before contributing new-structure counts. Encoder diagnostics probe the frozen baseline's actual input features.

## Remaining acceptance

| Task | Outstanding acceptance | GPU needed? |
| --- | --- | --- |
| T093 | Actual indexed release corpus, authoritative metadata and real structural-archive audit | No; source files are unavailable here |
| T095 | Sustained historical launch parity with actual trained weights | Yes for the historical GPU runs |
| T096 | Actual accelerator synchronization, thermal/power and sustained telemetry | Yes |
| T097 | Representative source/model/verified strata on real data; interface, trace, provenance and sealed-score fixtures are implemented | Real data/model artifacts needed; CPU tracing is supported |
| T098 | Live device orchestration acceptance | Yes; CPU transport and actual Docker CI remain independently executable |
| T099 | Frozen engineering measurements, precision and workstation steady-state | Yes |
| T100 | Real diagnostic and conditional paired pilot | Yes |
| T102 | Actual run-bound G0–G10 release authorization | Includes GPU and real-source evidence |

These tasks retain `[/]`; T099/T100 now have executable adapters rather than unavailable implementations. No hardware result is inferred from a fixture or a checklist. Only the existing LODA dispositions remain deferred.

## Reproduction

```sh
OMP_NUM_THREADS=2 python -m pytest -q \
  tests/unit/test_foundation_phase10_integration.py \
  tests/unit/test_foundation_roles.py tests/unit/test_foundation_measurement.py \
  tests/unit/test_foundation_decisions.py tests/unit/test_foundation_legacy_wrappers.py \
  tests/integration/test_foundation_overflow_diagnostics.py \
  tests/integration/test_foundation_qualified_resume.py \
  tests/integration/test_egca_training_step.py
python specs/007-experiment-foundation/validate_artifacts.py
bash scripts/foundation/test_wheel.sh
```

`OEIS_ROLE_TEST_IMAGE` selects the resolved CPU Docker image for actual role/cache/reclamation acceptance. `OEIS_GPU_ROLE_IMAGE` selects the resolved ROCm image for the separately marked coordinated HIP test. Missing capabilities remain skipped and cannot qualify release readiness.
