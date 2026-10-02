# Phase 7 review and recovery status

This draft reviews phase 7 of spec 007 only. Phase 8 has not started.

## Publication and validation boundary

An initial implementation and local commit were lost when the execution workspace reverted during publication. An earlier reported 620-passed/4-skipped result belonged to that lost tree and MUST NOT be used as evidence for this PR. No JUnit file or passing report has been reconstructed from those numbers.

The recovered grounding regression suite ran locally: **13 passed, 1 failed**. The failure was the recognized recurrence exhausting the legacy 10,000-fuel per-index default after 13 outputs. Direct execution confirmed OUT_OF_FUEL with the correct Fibonacci prefix. The recovered service now requests the declared 1,000,000-fuel ceiling for that recurrence path; its process wall-time and aggregate limits remain bounded. That change was not rerun locally because the execution service disconnected again.

The recovered code was subsequently validated in [GitHub Actions run 37075710546](https://github.com/grugnog/oeis-learn/actions/runs/37075710546) at commit `463771013ba6613da79c9c62d646064c7ad9dee8`: **472 passed, 4 skipped**. Compilation, targeted Ruff F/E9 checks and spec-kit prerequisite/artifact validation also passed. The first CI run exposed two registry-loading failures; fixing the path-versus-text call to the JSON helper resolved both. The recurrence fuel correction passed. See [phase-7-software-report.json](phase-7-software-report.json) for exact scope, skips and the JUnit/environment artifact.

This covers the foundation suites and new phase 7 regressions, not all original legacy owning suites. All phase 7 task markers remain partial until the remaining validation and integration work is complete.

## Findings and recovered remediation

1. **Unsafe affine classification and floating-point fitting.** Parameter products could enter empirical-basis least squares, including C*C fitted as C=4. The shared service uses typed parameter degree, exact rational elimination with explicit pivot columns, and bounded integer constraints for deficient systems. Scalar modular expressions have a separately named bit-vector solver.
2. **False solver outcomes.** Rank deficiency, unlucky modular pivots, failed reconstruction and unknown were conflated with UNSAT/timeouts. Modular rank is diagnostic, exact elimination provides the supported complete linear constraints, and solver status/reason determines the outcome. Corrected augmented rank and removed the deficiency reward penalty.
3. **Unverified final candidates.** No-placeholder candidates previously bypassed execution; training and evaluation diverged. Both public solver adapters now route to one killable service over the first at most20 visible terms. Final source is executed by production and independent reference semantics for the supported subset. Hidden tails are not fitting data. Training no longer auto-abstracts scalar helper literals or inserts prefix-only results into the elite buffer.
4. **Unfounded recurrence recognition.** Loop substring and placeholder count were treated as recurrence evidence. The recovered recognizer matches every operation/declaration/guard/state transfer of one order-two template, including initial states. Its lag matrix is cached by the immutable visible prefix within this fixed recognizer/index convention. Other recurrence shapes remain unsupported.
5. **Unqualified mathematical proof claims.** Both prover APIs now parse an allowlisted bounded arithmetic AST using one canonical integer n. Domain preimages handle negative/zero scales; original poles survive cancellation. Formula identities require an independent rational-evaluation certificate. A nonzero CAS residual alone is not a counterexample; an exact defined in-domain witness is required. Formula/OEIS correspondence remains an explicit external assumption.
6. **Unvalidated formula registry/provenance.** The registry checks content hashes, strict definition fields, expression syntax and index/domain convention, and returns copies. Bundled formula references are accurately labeled repository assumptions rather than verified OEIS source evidence. The discovery protocol is pinned to the repaired registry.
7. **Unsafe regex optimizer.** Full legacy modules are returned byte-identically. The supported strict-body rewrite catalog removes only typed nops and total effect-free push/drop pairs, retains writes/tee/traps/control ordering and reparses the output.
8. **Automatic native selection and misleading qualification.** The legacy runner selects Python explicitly and rejects Rust pending repair/conformance. The Python adapter checks the requested result profile, fresh state, memory and cumulative fuel limits. Native source remediation/build/parity remain incomplete; disabling it is not described as a repair. Native-required synthesis fails closed. Finite legacy matches are diagnostic, not qualified successes.
9. **Missing G7 authorization boundary.** Added a fail-closed gate and architecture-measurement entry point. Self-declared PASS reports cannot enable services. Positive activation, qualified record validators and portable evidence replay are unfinished software work.

## Task accounting

| Tasks | Current status / remaining work |
|---|---|
| T050, T054, T055 | Grounding repairs and regressions recovered; rerun actual final tree and reconcile original owning suites before completion. |
| T051, T057 | Parser, proof service, registry and scoped reporting recovered; CPU validation and all original consumer/schema regressions still required. |
| T052, T058 | Conservative typed optimizer plus Python regressions present. Native fixtures/parity and complete rewrite-evidence validation pending. |
| T056 | Shared public dispatch, final-source checks, narrow recurrence/data cache and elite boundary changed; broader owning integration validation pending. |
| T059 | Explicit runtime selection/limits repaired on Python side; Rust implementation, Docker build and mandatory no-fallback parity still pending. |
| T053 | qualified/v1 record validation and replay remain incomplete; not implemented by the baseline schema or this gate. |
| T060, T061 | Negative gating scaffold present. Qualified configuration/CLI activation, positive gate evidence, complete regression evidence and native parity remain incomplete. |

These are not all GPU-blocked tasks. Several are unfinished software/integration tasks after the workspace failure. GPU validation from earlier phases also remains separate.

## Validation commands and next steps

Run the included CPU workflow or, in the project Python environment:

```sh
OMP_NUM_THREADS=2 PYTHONPATH=src:. python -m pytest -q -ra tests/unit/test_foundation_*.py tests/contract/test_foundation_*.py tests/integration/test_foundation_*.py
SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation bash .specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks
python specs/007-experiment-foundation/validate_artifacts.py
```

Then run the original grounding, symbolic, discovery, synthesis, optimizer and sandbox owning suites; adjust stale contract expectations to scoped statuses without weakening their semantic assertions. Native dependencies/builds belong in the root Dockerfile environment. Do not proceed to architecture/learning measurements, claim G7 PASS, or mark this PR ready until the remaining software and native checks are resolved.
