# Phase 7 review and remediation

Phase 7 repairs the existing synthesis, execution and discovery code. It adds qualified records and an authorization boundary around the repaired services. The fixed foundation SFT smoke remains unchanged; phase 8 has not started.

## Repairs and acceptance coverage

| Tasks | Implemented behavior and evidence |
|---|---|
| T050, T054, T055 | One contained grounding service uses typed parameter dependence, exact affine elimination with explicit pivots, bounded integer solving and a separately named faithful bit-vector tier. Rank deficiency, unlucky primes and timeout do not imply UNSAT. Original solver entry points and their owning contract/integration suites are included in G7. |
| T051, T057 | Both prover APIs use bounded allowlisted parsing, canonical symbols, exact domain preimages, original denominator exclusions and checked rational certificates or concrete witnesses. Formula/OEIS correspondence remains an explicit assumption. Registry hashes and report consumers are checked; legacy proof labels are rendered unverified. |
| T052, T058 | Typed liveness removes only supported dead writes and total pure operations, retaining trapping producers, local.tee and control ordering. Every transformation is reparsed and checked against a separate rewrite certificate. Python and Rust fixtures cover traps, state and result profiles. |
| T053 | All ten qualified/v1 record kinds have positive and adversarial fixtures. Closed fields, versions, integer representations, identities, origins, modes, parent chains, budgets and status-specific evidence are checked. Solution and scoped UNSAT records, formula certificates and witnesses have separate contained replay. Future record formats do not enable future algorithms. |
| T056 | Training and evaluation share grounding dispatch and final-source verification, including no-placeholder candidates. The supported order-two recurrence recognizer checks state, initials, guards, updates and output, accepts alpha-renaming and independent limb-copy order, and caches by visible values, fit indices, order and transform identity. Hidden continuations are excluded from fitting. |
| T059 | The rebuilt Rust runtime uses explicit scalar/wide/strict profiles, exact four-limb decoding, fresh stores/instances, typed trap classification and bounded per-call/aggregate fuel. Python owns worker concurrency; native calls release the GIL. Native selection is explicit, and missing/stale native builds fail closed. Cargo dependencies, Rust sources and the compiled extension are identity-bound. |
| T060, T061 | G7 requires current implementation/runtime identities, owning-suite JUnit files, executed Rust-test evidence, the entire native parity transcript and portable replay. Activation resolves content-addressed base/parent/service profiles and preserves effective settings immutably. The harness exercises grounding, optimization, proving, native dispatch and the architecture-measurement gate with actual G7 evidence. CLI activation is tested in Docker. |

## Qualification procedure

The root Dockerfile has a CPU `qualification` target using the same installation and native-build scripts as development/GPU targets. No host compiler or additional system dependency is required. The build checks Rust formatting and uses the committed Cargo lockfile.

```bash
docker build --target qualification -t oeis-learn:g7 .
mkdir -p runs/007-g7/reports
docker run --rm -v "$PWD/runs/007-g7/reports:/evidence" oeis-learn:g7 bash -c '
  set -euo pipefail
  trap "chmod -R a+rX /evidence" EXIT
  python -m oeis_learn.evaluation.repair_gate --output /evidence
  oeis-learn foundation activate-qualified --config /evidence/qualified-contract.json \
    --gate-report /evidence/g7.json --profiles /evidence/profiles --json \
    > /evidence/activation-cli.json
'
```

Use a new output directory for each qualification. `g7.json` is the gate report; failed acceptance remains PARTIAL. The evidence bundle includes Rust logs, owning-suite JUnit files, `native/parity.json`, embedded portable replay records, `activation.json`, a qualified contract and content-addressed profiles. The command exercises CLI activation in the same image and preserves `activation-cli.json`. CI also retains exact Python/Rust build information.

Activation must run in the same qualified runtime/image. Changing implementation, dependencies or the native binary requires new qualification. A summary JSON, test count or self-declared PASS does not authorize a service. The full parity corpus comprises 12 regressions, 10,000 fixed arithmetic cases and 256 structured programs evaluated at indices 0–99, with Python, Rust and independent reference agreement.

The foundation CPU suite checks durable crash recovery using the same recorded execution outcomes. Independent fresh executions may differ at a wall-time limit; those resource limits are tested separately. The real-data discovery regression no longer manufactures a Fibonacci proof from unrelated polynomial formulas: it retains that finite relation as a conjecture and independently checks the supported triangular identity.

The contained formula worker proposes its rational certificate with exact fractions, avoiding a cold SymPy import within the two-second deadline. Its independent checker still reconstructs the degree bound and evaluates the original expressions at enough distinct defined points. An identically zero denominator cannot produce an identity certificate; denominator exclusions and formula/OEIS assumptions remain part of the checked statement.

## Scope and task accounting

See [phase-7-software-report.json](phase-7-software-report.json) for the exact tested commit, CI run, results and artifact hashes. Only evidence for that tree is counted; the historical lost-workspace 620-pass claim is excluded.

The report copy under `runs/007_phase7_review/reports/` is an audit record. Activation additionally needs all referenced JUnit files, Rust logs and the parity transcript from the CI artifact, in the matching qualified runtime. Copying a PASS report alone cannot activate services.

Grounding proves bounded visible-fit statements, not all-index sequence correctness. Recurrence recognition deliberately supports the checked order-two transition; other shapes remain unsupported. Formula identities are conditional mathematical statements and do not prove generated programs. Rewrite certificates establish value/trap equivalence, not identical fuel consumption. Legacy synthesis stays diagnostic and uses explicit Python execution; qualified native admission uses the G7-activated FoundationRunner.

G8/G9 learning, replay, search, richer program proofs and extended runtime modes remain disabled pending their own phases. GPU tests and the pinned production-container resource probe from earlier phases retain their pending markers. G7 is software qualification and does not certify AMD HIP training or complete spec 007.
