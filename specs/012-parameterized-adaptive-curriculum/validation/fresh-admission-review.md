# Blocked fresh-cohort admission and support audit

The uploaded admission diagnostic verifies the original frozen plan and the
sealed BLOCKED producer result. The availability record's exact 2,307 bytes
match the producer's SHA-256
`5046fd1eda62aac24bcd0f9246e90598517a848e287119293a754f1ea3916aa9`.
The frozen plan's references also match the uploaded successful `probes-003`
plan/result bytes. No model generation or scoring was reached.

| Family prefix | Category | Required | Available |
|---|---|---:|---:|
| `3f7f434b8e62` | certified | 2 | 0 |
| `5c475cecf9da` | certified | 1 | 0 |
| `83a8423db5b0` | ambiguous | 2 | 0 |
| `ee89096d39c3` | certified | 2 | 0 |

All 2,288 draws pass reference admission. After exclusions and source
deduplication, 1,371 candidates remain across 19 families. There are 917
`finite_window` rejections and nine `training_probe_window` matches; no other
exclusion reasons are recorded. Since 2,288 minus 917 is exactly 1,371, the
finite-window set accounts for every removal, the nine probe matches add no
unique removal, and there is no additional deduplication loss among survivors.
Thus all 64 draws from each of the four deficient families were rejected for
an existing 20-term-window match. Their reference executions were independently
reproduced locally without failures (62/64 distinct sources for the first family,
64/64 for each other family).

The seven missing slots include five certified and two ambiguous tasks. Other
families' surplus candidates cannot fill those slots under the frozen quotas.
Dropping these families changes the population; it cannot silently repair the
original 128-task design. Increasing solver allowances or tuning the model
cannot change this pre-inference feasibility result.

## Meaning of the window rule

The existing policy excludes a candidate if *any* contiguous 20-term segment
of its 100 terms equals any such segment of an exclusion sequence. It does not
require an identical source, whole sequence or initial visible prompt.

For the `5c475...` family, output is `p0*n*n` for n<=12 and `2*p1*n*n` afterward.
Changing p0 while retaining p1 changes both source and visible prompt but leaves
the later branch shared. A public synthetic example with p0=1 versus p0=3 and
p1=2 is rejected solely for `finite_window`, even though its source, group,
variant and initial 20 terms differ. This reproduces the declared exclusion
mechanism; it is not evidence of an implementation error or permission to
weaken that mechanism. The aggregate report does not identify the original
cell responsible for each observed match.

The current 64-draw failure does not prove that every possible numeric assignment
is excluded. Conversely, more random draws are not justified without checking
remaining support. A compact sufficient test can distinguish these cases.

## Read-only support census

`scripts/pac_fresh_support_audit.py` imports only the Python standard library.
It pins the reviewed protocol identity, verifies the expected frozen-plan bytes
and every referenced exclusion file, and scans the original 16 cells plus the
observed PAC cohort and training probes. Assessment inputs remain exclusion-only.
It does not evaluate models, run a native worker, build an image, modify a plan
or grant qualification/eligibility. It exports counts and input/code hashes,
not original sequences, sources or assignments.

For each allowed assignment, the registered family necessarily contains one
of the following windows under the original integer law [-1024,1024]:

| Family prefix | Window | Coefficient classes |
|---|---|---:|
| `3f7f434b8e62` | p0-n, n=0..19 | 2,049 p0 values |
| `5c475cecf9da` | 2*p1*n*n, n=13..32 | 2,049 p1 values, independently of p0 |
| `83a8423db5b0` | n*n+p0*n, n=0..19 | 2,049 p0 values, independently of p1 |
| `ee89096d39c3` | n-(p0+p1), n=0..19 | 4,097 sums |

If every class's witness window appears in the original exclusion data, every
allowed assignment is blocked under the current rule: a larger reservoir with
the same law cannot help that family. This avoids enumerating millions of
two-parameter assignments. An uncovered witness class is **inconclusive**:
some other window or exclusion may still block its assignments. It does not
prove that any fresh task can be admitted. Counts by original cell identify
where support was covered without exporting those windows.

Inputs are capped at 256 MiB per file and 4 GiB total, processed one cell at a
time; there is a cooperative 30-minute deadline. The current originals total
about 2 GB, so this is a CPU/file-reading diagnostic. These are not model runtime
or performance measurements. A failed audit emits the actual exception and
does not publish a conclusion.

Run from the existing qualified checkout. Fetch the helper into ignored
`run-state` instead of switching branches, so the checkout's qualified source
closure remains unchanged:

```bash
cd /home/owen/workspace/oies-learn &&
git fetch origin 012-pac-support-audit &&
git show FETCH_HEAD:scripts/pac_fresh_support_audit.py > run-state/pac_fresh_support_audit.py &&
python3 -S run-state/pac_fresh_support_audit.py \
  --plan run-state/generalization/pac-selector-fresh-001/execution/plan.json \
  --expected-plan-sha256 sha256:6a572f57d16a10f0ebe18d7dbe0c55a4f0fe5be205f6da7cdd0f2837a5f48de5 \
  --output run-state/generalization/pac-selector-support-001
```

Share `run-state/generalization/pac-selector-support-001/report.zip`.
The result is always DIAGNOSTIC_ONLY with `freeze_eligible: false`; PASS means
the census completed, even if every family has no support.

Keep the frozen `fresh-001` execution BLOCKED. No reseeding, quota reduction,
numeric-law expansion or exclusion change is implemented here. A subsequent
scientific amendment must be explicit, prospective, separately frozen and
clear about any changed population/freshness claim. It must not use model
scores to choose a favorable cohort.

## Verification boundary

Seven focused tests pass: coefficient support sizes and extreme/zero witness
windows match the independent exact reference; synthetic full coverage proves
blocking; incomplete coverage remains inconclusive; input tampering, unsafe
paths, changed protocol and expired work cannot publish conclusions; a copied
helper runs with `python -S` and binds its actual script bytes. The related
selector unit/integration run passed 21 checks before the final deadline test.
No workstation support census result is available yet. No GPU or training work
was executed by this implementation.
