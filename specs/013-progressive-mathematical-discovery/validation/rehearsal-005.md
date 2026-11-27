# Workstation rehearsal 005

The original support archive (`rehearse(5).zip`) has SHA-256
`ffcd0525d8d986736c78e8ea83ebe6fac65cd93f413a922e5ff1a6993632fa82`.
All 158 inventoried members matched their recorded size and hash; there were no
collection omissions. Session `run-state/discovery/013-hip-005` bound source
`sha256:dc8f0e780484fb511c8fec3b66be9cc892df00af88d29e7b6cb2f3b18bb43f2d`.

Both images built. All **2,319 CPU/native tests** passed (ten deselected and 82
subtests passed), as did all **27 actual-HIP preflight tests**. Base qualification
completed. All four selected tiny-model cases published original PASS producers,
with update, parameter change, full restore, deterministic continuation,
generation, native scoring and cleanup successful. Their original seals and
exact source/image/engine/hardware/case scope bindings were independently checked
against the archived matrix.

| Input length | Microbatch | Peak allocated bytes | Bytes before workspace release | Final allocated bytes |
| --- | --- | --- | --- | --- |
| 4 | 7 | 204450816 | 67108864 | 0 |
| 20 | 7 | 205117952 | 67108864 | 0 |
| 20 | 16 | 219075072 | 67108864 | 0 |
| 4 | 16 | 204450816 | 67108864 | 0 |

The original producer identities, in that order, are:

- `sha256:2e0c164ea8a30ccb4fdba2009aaef8d6d4181cfda4e4b9f43c2a069a9718dcc5`
- `sha256:32a15db9861df24d56a2192582308cba94a6790c52cf302b0169f6c37d3ea55e`
- `sha256:1ad4274648b46fd14ccf1ae854354123bc00ac628c655dea4b2609f036f58f93`
- `sha256:dc539d57a5efb236afd0b32f44795cee6a3ef6e293af6646bd75245a7452d5ba`

Qualification reconciliation rejected all four receipts because the previous
repair added `cleanup_diagnostics` to the closed v1 measurements object. This
was a producer/consumer contract mismatch, not another failed GPU cleanup.
Failure collection worked: all independent cases ran and all four failures
survived in the support archive. Preparation, freeze and campaign acceptance
remained blocked. These receipts are retained as original execution evidence,
not rewritten or promoted to qualification authority. T082 and T088–T111 retain
their pending requirements; this is not research-model, sustained-run or
end-to-end acceptance evidence.

## Repair and verification

The producer now keeps allocator diagnostics exclusively in `measurements.json`,
already collected on success and failure, and publishes exactly the existing v1
receipt fields. The strict validator is unchanged. The regression exercises real
CPU updates, full checkpoint continuation, generation and native scoring with
explicit device-boundary doubles, then feeds the resulting original producer
bytes through `hardware_plan`. It failed for both successful execution cases
before the fix and passes afterward. Deliberately retained allocations still
fail without publishing a producer. No CPU fixture is actual-HIP authority.

All **108** targeted cleanup, workstation, receipt/reuse, handoff, qualification,
packaging, CI-roster and implementation-digest checks passed with native/build
requirements enabled. The expensive installed all-stage test was explicitly
deselected. The package implementation digest was refreshed without changing
historical dispositions. `git diff --check` passed. No image was built locally;
CI was not awaited. A new source-bound rehearsal is required to finish current
qualification and reach campaign acceptance.
