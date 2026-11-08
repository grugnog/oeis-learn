# Adopted 7,664-draw structural reservoir amendment

The workstation execution is now reviewed in
[fresh-002-review.md](fresh-002-review.md). Admission succeeds at 1.298 GiB peak
preparation RSS; the scientific result remains INCOMPLETE because D completes
64/68 against the unchanged 65/68 floor. Preserve the execution and inspect its
existing score-producer evidence before considering another run.

The operator authorized implementation on 2026-11-07 after reviewing the
[all-window support result and concrete proposal](fresh-reservoir-proposal.md).
The earlier `pac-selector-fresh-001` plan and BLOCKED execution remain immutable.
This is a separately frozen experiment with unchanged scientific comparison.

## Registered change

Policy `reservoir-7664/v1`:
`sha256:7c0be8a12df99cd7c9a9cc3bbefba0b1b55d381a5e1917da63b3e72c9c6580c1`.
Protocol `pac-selector-structural/v2`:
`sha256:6603333f19a0dbf13926cb5e3c5d481cd1af01b7d814099cd36ba342e14937e3`.
The exact policy and reviewed parent-module sources are packaged in
`src/oeis_learn/generalization/policies/selector-reservoir-amendment.json`.

| Family prefix | Old draws | New draws | Required tasks |
|---|---:|---:|---:|
| `3f7f434b8e62` | 64 | 4,096 | 2 |
| `5c475cecf9da` | 64 | 512 | 1 |
| `83a8423db5b0` | 64 | 512 | 2 |
| `ee89096d39c3` | 64 | 512 | 2 |

Other cells keep their original draw counts, including zero-quota cells. Total
draws are 7,664, within the existing 8,192-identity reservoir. The descriptor
catalog, seed (20261108 with the existing structural +1 offset), independent
uniform integer law [-1024,1024] and per-family stream capacity 8,192 are
unchanged. Every old candidate prefix reproduces exactly; new candidates extend
those prefixes. No reseeding or repeated draw until a desired outcome is allowed.

Retain all original exclusion inputs/rules, the 23 positive-quota families and
128 tasks (68 certified / 26 ambiguous / 34 unresolved), B/D checkpoints,
selector, all seven controls, 20 visible/100 total terms, eight attempts and
60-second portfolio, category reporting, and certified 65/68 completion floor.
No research training, HPO, family removal or quota transfer is introduced.
Preparation and evaluation keep their existing work envelopes and qualified
CPU/memory/thread/runtime/precision/security facts.

## Probe reuse and current qualification

The new `amend` command independently validates:

1. The exact unamended frozen parent and its executed copy, original BLOCKED
   custodian result/availability, four recomputed deficits, and absence of any
   B/D model execution beneath that failed parent.
2. The bound successful original probe result and score producers, unchanged
   inventory/recipe/checkpoint provenance, primary completion >=244 for both
   selectors, zero incomplete traces and complete generation.
3. The original v2 all-window census, its fixed helper digest, original plan
   bytes, all 18 original exclusion inputs, positive support and reconciled
   scan increments. The census grants no execution authority on its own.
4. The historical qualification's original producer/artifacts and the exact
   source closure underlying those old probe observations.

Probe reuse has a separate, source-bound compatibility receipt. Every production
and build byte must match that original closure except the registered new
reservoir modules/data and two shared modules. Test Python changes and the
standalone diagnostic helper do not affect this inference comparison. Non-Python
test/build resources remain protected.

For the two shared modules, the registry contains the original full source text,
whose digest must equal its original qualification entry. AST comparison permits
only the named custody/orchestration function bodies, with identical signatures
and no decorators/default side effects. All module-level code and all other
functions, including generation, selection, scoring and metric calculations,
remain protected. The sole reporting exception is an exact optional
`admission.json` archive-member statement; modifying metrics or that filename
fails compatibility. This is not blanket source-hash replacement.

The compatibility check was also exercised against the actual published
`probes-003` source closure
`sha256:d29306dbcc384b15ab054757c4da3d28ada9ca1883c74085f6dc7e477d661040`,
reconstructed from its matching Git tree. It passes. Old probe observations keep
their original source/images and references; they are not relabeled as new runs.

The command builds current qualification/GPU images and requires new CPU/native
and actual-HIP qualification before reporting FROZEN. Runtime, hardware,
precision, resource, security and build-dependency facts must still match the
historical scope; only the explicitly compatible source closure and image IDs
change. Actual structural execution then independently consumes that new
qualification. No probe panels or research learners are rerun by this command;
qualification still includes its normal real finite-update/device checks.

## Workstation commands

From the repository root, fetch and check out the separate implementation stack:

```bash
cd /home/owen/workspace/oies-learn &&
git fetch origin &&
git switch 012-pac-reservoir-amendment &&
python3 -S scripts/foundation_run.py --engine podman pac-selector-validation amend \
  --previous run-state/generalization/pac-selector-fresh-001 \
  --support run-state/generalization/pac-selector-support-002 \
  --output run-state/generalization/pac-selector-fresh-002
```

After it reports FROZEN, run that exact new plan:

```bash
python3 -S scripts/foundation_run.py --engine podman pac-selector-validation structural \
  --plan run-state/generalization/pac-selector-fresh-002/plan.json
```

The freeze requires a fresh output directory and pins its one execution path.
Source/image edits, altered proof inputs, changed scoring, a moved execution
path, or an already-started execution fail closed. The v1 freeze/draw interface
continues to accept only its original protocol; the amendment is explicit.

## Preparation gates and reporting

The custodian reference-admits the fixed new draw and applies the unchanged
exclusions before model evaluation. It records reference/exclusion elapsed time,
index rows and process peak RSS. A 14-GiB peak ceiling reserves room within the
unchanged 16-GiB qualified role envelope; crossing the ceiling stops before native
admission or model evaluation. The existing role/cgroup work and memory limits
still apply. A further quota deficit remains BLOCKED. There is no automatic
fallback cohort, extra seed, replacement family or additional reservoir retry.

After quotas, every selected target still requires native/reference agreement.
Successful reports additionally include a sealed `admission.json` containing
public availability/resource counts and the original producer reference. It
does not replace qualification or model scores. Failure ZIPs retain bounded
availability and preparation-resource records under `admission-diagnostics/`,
with source hashes and explicit omissions, excluding private payload fields.

Share `run-state/generalization/pac-selector-fresh-002/execution/report.zip` on
success, or the printed `failure.zip` if a gate stops. The report's extra
admission record avoids another manual packaging step for resource/quota review.

## Validation and rollback boundary

132 affected local checks pass; one installed-native-image check skips in this
workspace. New tests cover exact prefix retention, rejection of seed/law/metric/
inference/import/signature mutations, original evidence tampering, new
qualification failure, changed resources/images, one fixed execution path,
custodian-only role mounts, memory breach before native work, private-field
omission, and success/failure admission exports. The real wheel packaging check
includes the new policy data and modules. Only the current implementation hash
changes among the historical 68 dispositions. Stdlib help, whitespace,
prerequisites and read-only 16/16 checklist pass; no hooks and no CI wait.

This workspace has no qualified native image or HIP device. Current image
qualification, new cohort availability/RSS, and B/D model scores remain
workstation measurements; no such result is claimed here. Restoring the old
branch does not make `fresh-001` retryable. Preserve both plans and their original
statuses; a failed new scope requires its own explicit diagnosis.
