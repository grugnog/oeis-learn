# HIP failure-class audit

Implementation update, 2026-11-01 UTC: T103–T109 are now implemented and locally
validated; see [operational repairs](operational-repairs.md). Acceptance-011 was
subsequently made available and inspected; see [HIP evidence](hip-acceptance.md).
The original audit below records the evidence and open tasks at audit time. Its
statements that 011 was unavailable and the repairs were open are historical.
Fresh hardware qualification remains pending for the changed source.

Audit date: 2026-11-01 UTC. Audited source: remote commit
`f2fd4f454d8f8f3d6d12ed9692b851c0ad42b4d0` (T102; same source tree as local
`e77b5a6`). This is an audit and remediation backlog, **not a repair or hardware
qualification claim**. No HIP command was executed. T103–T109 are open.

## Conclusion

Most repeated runs have exposed orchestration, resource-budget, container-boundary
and evidence-packaging defects. The recorded CPU/native and HIP qualification
suites passed in runs 007–010 while the complete feature workflow still failed.
Those passes demonstrate their tested scope; they do not establish successful
generalization replay, full-state resume or final export.

The repair process was too narrow: it followed the first visible exception, then
sent the operator back through the expensive workflow. Missing diagnostic layers
required additional exports, and small/fake fixtures hid differences in real
directory ownership, image identities, checkpoint size and runtime environment.
One diagnostic addition also preceded a new crash. A longer timeout can help a
measured slow operation, but cannot repair these contract and recovery defects.

Seven remaining gaps were identified. Four have local behavioral counterexamples;
one runtime-environment mismatch was reproduced through actual command construction;
the storage and test-coverage gaps follow from inspected code. These findings do
not prove that each gap caused an earlier operator failure.

## Evidence and limits

Re-read available original archives and verified their bundle inventories/hashes:

| Artifact | SHA-256 |
|---|---|
| `acceptance-001.zip` | `ea6b03bb9bbc5bb0257081d88e73c0b7f77699bf5de65cce48aa9890618faa52` |
| `acceptance-010.zip` | `bc28e3d8190ce92c1d8e3aa8397fea9c71df6be2118018604fa3da4467d79709` |
| `acceptance-010-diagnostics.zip` | `dd11c83fe80c2b3829863c27109fef0674933dd49aa188f5fdae48e18b9615f3` |

Runs 002–009 below rely on the previously recorded evidence in
[hip-acceptance.md](hip-acceptance.md) and [local-checks.md](local-checks.md), checked
against current source and remediation tasks. Their original archives are not
available for independent reinspection in this audit. `acceptance-011.zip` was not
available to read; its result is unknown. Historical introductory status paragraphs
in the validation/task documents are stale; the later dated run records preserve
the more recent evidence. This report does not change any original run status.

## Failure history by type

| Runs | Observed failure class | What the evidence supports |
|---|---|---|
| 001 | Engine identity/configuration | A command selected as Docker returned Podman metadata. The deliberate portability guard rejected it. This is not a GPU computation failure. |
| 002 | Filesystem ownership contract | The runner created a directory that image inspection required to be fresh. T085 reproduced this deterministic integration defect locally. The original exception was missing from the archive, so the diagnosis rests on matching behavior rather than a captured stack trace. |
| 003, 005, 006; GPU inspection in 004 | Cold-start deadline | Inspection expired before Python startup; 006 ended in overlay setup. Later startup succeeded with larger allowances. The underlying cold-start mechanism was not established. |
| 004 | Test/storage environment mismatch | CPU fixture setup assumed more free space than the roughly 1.63 GB tmpfs provided. Missing nested test results also obscured diagnosis. T087/T089 addressed reporting and the fixture assumption. |
| 007 | Cleanup, packaging and host/container boundary | CPU/native and 27 HIP tests passed. Consumer inspection exhausted its cleanup allowance. Packaging rejected legitimate pinned-build filenames. A related review found host-side schema validation requiring a container dependency. T092–T095 addressed these distinct paths. |
| 008 | Nested deadlines and full-model cost | CPU/native and 27 HIP tests passed. The retained model exceeded the 30-second learner allowance; process termination could consume the container-removal envelope. Nested role evidence was missing. T096–T098 addressed these paths. |
| 009 | Diagnostic interference | The generator exited 139 at about 32.39 seconds during its first asynchronous traceback dump, before its 90-second deadline. No core established causality. T100 removed cross-thread stack walking while retaining external supervision and timestamp progress. |
| 010 | Cross-image identity and missing application outcome | CPU/native and 27 HIP tests passed; retained-eval, refresh and parameter-target cases passed. The first continued-SFT learner completed three updates in about 86 seconds. The replay learner then returned application FAIL with process exit 0. Its omitted result was recovered separately: `replay evidence changed on restore`. |

T102 aligned replay admission with the learner image while leaving GPU access off
for admission. A local regression demonstrated that independently built CPU/GPU
workers can yield different runtime identities. The original 010 evidence did not
retain both compared identities, so the exact differing field remains unproven.
The unavailable 011 artifact cannot be used to claim T102 passed on hardware.

## Remaining findings

Paths below are repository-relative. Counterexamples used temporary directories,
recorded launch boundaries and mocked clocks/signals; no container or GPU ran.

### F5 — Critical: watchdog can signal after its handler is gone — T103

**Source:** `src/oeis_learn/pipeline_validation/disk.py`,
`DiskWatchdog.__enter__/__exit__`. **Requirement:** Constitution III, NFR-006/007,
SC-007. **Gap:** contradicts recoverable-operation requirements.

Shutdown sets the stop event and joins the monitor for two seconds, then restores
the previous SIGUSR1 handler. An already-running disk scan can outlive that join.
If it subsequently detects a breach, the monitor unconditionally sends SIGUSR1.
The previous handler may be the default process-termination action or a different
watchdog's handler. Recovery can therefore be bypassed or attributed to the wrong
scope.

Reproduction: block only the monitor's `check()`, exit the context, then release
the blocked check with `ResourceBreach`; replace `os.kill` with a recording mock.
Observed `watchdog_alive_after_context=true`, then signal 10 sent after context
exit. No actual unsafe signal was delivered. Repair must address ownership/lifetime
and bounded shutdown, including nested guards, rather than disable disk monitoring.

### F1 — High: finalization can lose export or export contradictory success — T104

**Source:** `scripts/generalization_hip_serial.py`, `execute` verifier handling and
final archive copy. **Requirement:** FR-033/034, NFR-007, SC-007. **Gap:** contradicts
the terminal failure/export contract.

Verifier-summary JSON parsing occurs outside the stage exception handler. The
wrapper also treats `integrity == PASS` as sufficient verification even following
a nonzero verifier exit, or accepts exit zero without requiring a valid summary.
Its shared archive was generated before the final packaging/verification ledger.

Two recording-runner counterexamples exercised the real `execute` function:

| Injection | Observed outcome |
|---|---|
| Verifier exits 1 and leaves truncated `{` summary | Uncaught `JSONDecodeError`; workflow remains PARTIAL; no shared archive |
| A real packed report says PASS; verifier exits 1 but leaves integrity/scope PASS | Wrapper returns 1 and final workflow says FAIL; shared archive still says PASS |

These are synthetic failures at the external boundary, not fabricated hardware
receipts. Finalization must validate the canonical verifier contract, preserve
valid negative-scope results, and distinguish them from an infrastructure failure
after an earlier PASS summary. Preserve original receipts and add final outcome
evidence; do not silently rewrite an earlier qualification record.

### F2 — High: diagnostics still omit whole control surfaces — T105

**Source:** `pipeline_validation/generalization_diagnostics.py`, `candidates` and
`nested_controls`. **Requirement:** FR-033, NFR-007, SC-007. **Gap:** partial.

Created representative controls at actual workflow layouts and ran `collect`.
All five paths below were absent from the inventory, rather than recorded as
explicit omissions:

```text
private/logs/replay-resume/process.json
private/logs/replay-resume/disk/disk.json
qualify/current/image-1/logs/process.json
qualify/authority-validation/role/result.json
replay-resume/case/private/replay/seed/training/arm/execution/role/disk.json
```

The outer stage stdout/stderr tails are copied elsewhere, but they do not replace
the watchdog's termination reason or disk observation. Related inspection paths
include the current-device inspection and bundle-verifier inspection. A process
killed before publishing its role result can still require another manual export.
Use bounded projections for controls with private paths/commands; do not broadly
archive private directories or credentials.

### F3 — High: slow persistence can exhaust the clock before work starts — T106

**Source:** `pipeline_validation/generalization_training.py`, `train_updates` and
`run_torch`; `generalization_roles.py`, `project_and_invoke`. **Requirement:**
FR-024, NFR-006, plan update-boundary resume and containment. **Gap:** partial.

`train_updates` checks the deadline before saving the reserved pending batch, but
not again between that save and the update. With a one-second allowance and a
virtual save costing two seconds, an update started at second 2; the function
returned COMPLETE at second 4. The pending debit is already persisted, so stopping
before the update should remain resumable.

Related clock edges require the same repair: `run_torch` computes remaining time
before its initial save, then starts a fresh loop deadline afterward; role input
verification/copying consumes time before passing the original allowance to
`invoke`. Outer process watchdogs still bound execution—this is not a claim of
unbounded GPU work. The defect increases the chance of hard termination instead
of a clean budget stop and makes nested time accounting inconsistent.

### F4 — High: qualified and production scratch environments differ — T107

**Source:** `pipeline_validation/generalization_roles.py`, `invoke`;
`container_runtime.py`, `private_runtime_environment`/`Engine.command`;
`container_validation.py`, `_stages`. **Requirement:** NFR-004/006/008. **Gap:** partial.

Qualification provides writable private TMPDIR/HOME/cache locations, explicit
thread controls and a 2 GiB `/tmp` limit. Production generalization roles pass no
runtime environment and inherit the 64 MiB `/tmp` default under a read-only root.
The existing helper explicitly explains that ROCm/LLVM scratch can exceed that
small tmpfs on the first backward pass.

Calling real `invoke` and `Engine.command` with launch/reclaim recorded produced:

```json
{"tmpfs":["/tmp:rw,noexec,nosuid,nodev,size=67108864,mode=1777"],"environment":[],"runtime_directory":false}
```

No engine was executed. The environment mismatch is established; actual scratch
exhaustion is not. Run 010's first learner succeeded, so this must not be presented
as its failure cause. Fix the mapping through each existing `/output/<role>`
mount; the helper's `/work/output` constant cannot be reused unchanged. Keep
scratch private and covered by disk accounting.

### F6 — High: preflight does not model the complete acceptance storage graph — T108

**Source:** `pipeline_validation/generalization_training.py`, `storage_preflight`;
`generalization_workflow.py`, `replay`; `generalization_acceptance.py`,
`training_proof`; `generalization_commands.py`, `_resume_inputs`/`_copy`.
**Requirement:** FR-024, NFR-006/007. **Gap:** partial.

Reservation multiplies estimated snapshot bytes by milestone/pin count and the
registered training arms/seeds. Acceptance additionally creates paused and resumed
learner runs, full-state proof inputs and temporary projected channel copies.
These operations are absent from that reservation model. Persistent auxiliary
inputs and temporary copies can coexist with the original checkpoints.

Run 010's replay preflight reserved 13,937,811,680 bytes for ten snapshot units
using 417,986,648 parameter bytes. This audit has not demonstrated that the actual
peak exceeds the frozen cap. It establishes that the preflight calculation does
not explicitly cover the downstream graph. Model peak live data, including actual
retention/deletion order; a blind multiplier or cap increase is not evidence.
Likewise, the downstream proof role's fixed 60-second allowance has not yet been
validated for the complete retained-model state. Budget review must include copy,
hash, deserialize, proof and cleanup costs rather than extrapolate kernel time.

### F7 — Medium: passing smoke tests do not cover the new orchestration — T109

**Source:** `tests/integration/test_generalization_gpu.py`,
`test_generalization_device_load_and_finite_update`; CPU fixture workflows and
`container_validation.py` test selection. **Requirement:** NFR-003/005,
SC-006/007, plan C/D validation. **Gap:** partial coverage.

The HIP smoke test exercises real model save/load and a finite update through
`_batch_loss`. It does not call the new `run_torch`, full-state checkpoint store,
replay cursor restoration or uninterrupted-versus-resumed path. The full acceptance
workflow intends to cover those later, but earlier failures repeatedly prevented
reaching them. CPU workflows use small fixtures and mocked external boundaries;
the counterexamples above show gaps in their fault coverage.

The smoke test is useful and its passing result is valid. The gap is treating that
scope plus fixture success as sufficient readiness to request the next expensive
full run. Exercise canonical boundaries and all post-training branches locally;
extend or relocate a small installed-path HIP resume check inside the existing
serial invocation, without adding another operator command or duplicate expensive
qualification.

## Remediation and next-run gate

1. Complete T103–T109 with meaningful regressions. Group any nontrivial lifecycle
   or resource-accounting design in one reopened Phase-B algorithm block, then do
   routine integration/testing. Commit and push each logical group to the existing
   branch/PR.
2. Run a combined local fault matrix: slow scans/saves/copies; malformed verifier;
   report/verify failure after a successful payload; application FAIL with process
   exit 0; each missing control path; post-training evaluation/proof/resume/export.
   Verify every failure produces an honest complete or partial archive.
3. Check the whole retained-model artifact/time graph with size metadata and
   command-construction tests. Do not require multi-GB local fixtures or guess a
   new timeout. The user permits exceeding five hours when justified; that is not
   a reason to increase budgets without measuring the work being bounded.
4. Inspect the existing 011 artifact if made available before selecting new
   hardware work. Then use one canonical serial operator invocation that includes
   remaining representative acceptance, failure validation and automatic evidence
   packaging. Reuse only evidence whose full closure still validates.

No CI wait or new HIP run was requested by this audit. No claim is made that a
further run is ready now. The remaining findings are open in
[tasks.md](../tasks.md#phase-3-convergence).

## Validation of this audit

Executed local counterexamples for F1 (two cases), F2, F3 and F5, and actual
command-construction inspection for F4. Re-verified the three available archive
inventories/hashes. F6/F7 are source-backed coverage/accounting findings, not
simulated hardware successes. Checked task numbering, append-only preservation and
documentation formatting. No runtime code changed and no full regression suite
was rerun for this documentation-only change.
