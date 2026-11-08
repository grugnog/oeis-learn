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

## Verified first support result and all-window follow-up

The subsequently uploaded v1 result verifies its canonical seal, original plan
identity/byte reference and exact published helper digest. Result identity:
`sha256:6068d06b65b8722a17007ce98951fab56c178efc3f8c7dee884e905f3fa9c5bd`;
helper digest:
`sha256:63aecae1829a7473a89bb906ef30bb6c8ed4f77726c02735566ee201dab3f5e3`.
The audit completes in 15.716 seconds. All 18 cell records are present, and
their incremental coverage counts sum to the reported totals.

| Family prefix | Covered witness classes | Uncovered witness classes | Full-law conclusion |
|---|---:|---:|---|
| `3f7f434b8e62` | 1,619/2,049 | 430 | inconclusive |
| `5c475cecf9da` | 1,996/2,049 | 53 | inconclusive |
| `83a8423db5b0` | 600/2,049 | 1,449 | inconclusive |
| `ee89096d39c3` | 2,900/4,097 | 1,197 | inconclusive |

These uncovered counts are not numbers of eligible tasks, nor proof of remaining
support. V1 tests one guaranteed window per coefficient class. The same target
assignment may match an old sequence at another window position. For instance,
the window 10,9,...,-9 occurs in the p0=90 member of the first family at offset
80, even if its initial p0=90 window is uncovered. Branch-dependent targets have
analogous extra windows and mixed windows at their branch boundary.

Coverage first encountered at each group of inputs is:

| Input group, recorded scan order | `3f7f...` | `5c47...` | `83a8...` | `ee89...` |
|---|---:|---:|---:|---:|
| Four original selection/assessment cells | 558 | 17 | 9 | 1,234 |
| Static training plus eleven refresh cells | 954 | 1,978 | 582 | 1,504 |
| Observed PAC structural cohort | 107 | 1 | 8 | 162 |
| Training probes | 0 | 0 | 1 | 0 |

This incremental attribution depends on scan order; the groups can overlap.
It is not the coverage that would disappear if one input group were removed.
The v1 report remains DIAGNOSTIC_ONLY and does not establish whether increasing
the draw budget can restore the original quotas.

### Exact all-window audit

The helper's optional `--all-windows` mode produces
`pac-fresh-support-audit/v2`. It checks every candidate position 0..80 under the
same family definitions and numeric law, using inverse arithmetic constraints
rather than generating millions of programs:

- A matching linear window blocks an interval of allowed p0 values or sums.
- A matching pure quadratic branch blocks a parameter row or column.
- A window crossing a branch boundary fixes a parameter pair. Third differences
  locate possible joins; the exact 20-value equations validate each pair.
- Globally quadratic crossing windows receive explicit treatment when a branch
  contributes only one term, or both branches have identical coefficients.

The union of rows, columns and individual pairs counts the complete 2,049 by
2,049 grids with sparse byte rows, bounded by about 8.4 MB of point flags total.
For n-(p0+p1), assignment counts use the triangular multiplicity of each sum;
the report separately retains the number of distinct output classes. No original
windows or parameter assignments are exported. The original byte/time caps,
input hashes and diagnostic-only authority remain unchanged.

A zero window-free assignment count proves no numeric assignment in that family
can survive the current window exclusion rule. A positive count establishes only
window support: candidates still require all frozen admission checks, including
native/reference agreement. It does not authorize reseeding, a larger reservoir,
quota changes, or model evaluation. A scientific amendment remains separate.

Run the extended diagnostic from the same qualified checkout, without switching
branches or rebuilding an image:

```bash
cd /home/owen/workspace/oies-learn &&
git fetch origin 012-pac-support-audit &&
git show FETCH_HEAD:scripts/pac_fresh_support_audit.py > run-state/pac_fresh_support_audit.py &&
python3 -S run-state/pac_fresh_support_audit.py --all-windows \
  --plan run-state/generalization/pac-selector-fresh-001/execution/plan.json \
  --expected-plan-sha256 sha256:6a572f57d16a10f0ebe18d7dbe0c55a4f0fe5be205f6da7cdd0f2837a5f48de5 \
  --output run-state/generalization/pac-selector-support-002
```

Share `run-state/generalization/pac-selector-support-002/report.zip`. The all-window
workstation result is pending; the observed v1 counts are not reclassified.

Validation of the extension: 29 affected checks pass, including 14 focused support
cases. The independent exact reference supplies every window of extreme/zero
and one-term-boundary assignments; inverse projection recognizes each one.
Projected membership is also compared with brute-force window equality on an
independent coefficient grid, including extra quadratic/nonquadratic negatives.
Tests cover row/column/point overlap, shifted linear matches, triangular sum
multiplicity, the optimized row reader versus individual-window projection, both
complete/incomplete synthetic censuses and copied `python -S --all-windows`
execution. No workstation v2 support result or speedup is claimed.
