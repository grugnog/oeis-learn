# Phase 5 arithmetic qualification — T089

Status: **COMPLETE** for reviewed implementation source
`sha256:942afe05c9cbe3ba95f839ed071ca7a44855b1222d2e0b03d82f0c39eeeef562`.
Workflow39686633400 at head9f86c4d passes184 selected cases with zero failures,
errors or skips. The artifact ZIP CRC/SHA256, JUnit hash and source binding
verify. The matching native build, runtime, worker, compiler and committed lock
identities are retained in the [execution review](phase-5-execution-review.json).
These are actual managed-image tests, not local mocked execution.

The independent measurement corpus has75 case/batch records. Its three repeated
`generated-16384` batch100 samples reach aggregate fuel after82 indices; this
resource exclusion is explicit and is not arithmetic disagreement or successful
full-coverage throughput. Expected overflow remains classified separately. No
arithmetic discrepancy is admitted. Full Phase5 qualification remains blocked
on its other gates; no training permission follows from T089.

`scripts/qualify_bigint_phase5.py --inside arithmetic` runs frozen arithmetic,
independent reference, language/codec, decoder and native signed-256 suites with
zero failed/error/skipped cases required. Its `summary.json` binds the complete
source snapshot and packaged runtime, worker, compiler and lock identities.
The measurement contract also checks successful outputs and classified failures
through the real packaged measurement executable. Resource exclusions are
reported separately from exact arithmetic disagreements; no failed result is
admitted. Existing Phase 2/3/4 evidence retains its original source binding.

Run the public host helper described in [phase-5.md](phase-5.md) to reproduce.
