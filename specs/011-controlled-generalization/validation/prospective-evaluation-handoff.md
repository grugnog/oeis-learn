# Prospective retained-pilot evaluation: operator handoff

**Current status:** the first invocation completed qualification but failed fresh
cohort admission. Do not repeat the full command below yet. Follow the
[quota audit and proposed scope amendment](prospective-quota-audit-20261104.md) first.
The recovery evidence has already been supplied; do not export it again.
T174 remains incomplete.

The following is the original pre-run handoff, retained for provenance.
Implementation was complete through T173; actual hardware execution T174 remained
pending. The existing PR branch contains the changes. This run performs **no new
research training**. The canonical qualification session still performs its small
finite-update/resume acceptance fixtures.

## What this run measures

Compare the exact completed hourly-pilot latest static and refreshed weights on
256 fresh selection instances. Preserve59 original families and their exact
family-by-holdout counts,128 parameter/128 structural rows, uniform integer
coefficients[-1024,1024],20 visible/100 scoring terms, eight attempts and60-second
matched portfolios, seven controls, original bank seeds and float32 model recipe.
The fresh numeric seed is20261104. The4,688-candidate reservoir is finite; no
automatic extra draws, seed retry, quota relaxation or budget sweep occurs.

The unchanged family policy yields153 certified within-family continuation rows,
55 demonstrated-ambiguity rows and48 classification-unresolved rows, if the cohort
passes admission. This does not certify identification across different families.
The primary comparison is certified structural family-macro equal-budget confirmed
success; certified parameter macro is secondary. All strata/controls, raw counts,
paired discordances and censored outcomes are exported. Each certified
stratum/arm needs95% primary completion before interpretation. There is one
training seed, no automatic winner/adoption decision and no assessment scoring.

The new native cohort is checked against **all** original prepared training,
selection and assessment cells. Assessment payloads are copied only into a
read-only exclusion-custodian projection; the host hashes their bytes without
parsing their contents. They never reach generation, fitting, selection scoring,
interactive analysis or reports. Exact source/group/variant and every shifted
20-term window are checked, with structural-family separation in both directions.
Original files remain unchanged. The resulting exact roster, checkpoints, recipe,
helpers, source, image pair and decision rules are frozen before inference.

## One command block

Run from the existing workstation checkout. The first Python command is a
standard-library-only file check and cannot launch containers. It checks the
reviewed preparation inventory, both completed learner records and actual retained
weights, and requires128GiB free. The script deliberately stops if essential
original files have been removed; it will not substitute another run.

```bash
(
  set -euo pipefail
  cd /home/owen/workspace/oies-learn
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization

  python3 -S specs/011-controlled-generalization/validation/prospective_evaluation.py check-inputs

  podman build --target qualification -t oeis-learn:generalization-cpu .
  podman build --target gpu -t oeis-learn:generalization-gpu .

  EVAL_OUTPUT="run-state/generalization/observable-eval-$(date -u +%Y%m%dT%H%M%SZ)"
  python3 -S specs/011-controlled-generalization/validation/prospective_evaluation.py run \
    --engine podman \
    --pilot run-state/generalization/fresh-p1-hourly-20261104T093555Z \
    --prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
    --imported run-state/generalization/uniform-import \
    --cpu-image oeis-learn:generalization-cpu \
    --gpu-image oeis-learn:generalization-gpu \
    --output "$EVAL_OUTPUT"
)
```

This updates the existing branch; it creates no branch or PR. Current images and
one new qualification are necessary because the bank original-source fix changed
two entries in the previously qualified production source inventory. The temporary
runner changes add no further production snapshot changes. Do not reuse the old
`bfd04a2` qualification as current. An explicitly supplied `--qualified PATH` can
consume a compatible original producer after a failed follow-up; it cannot bypass
source/image/resource checks or infer success from an old summary.

Order: input verification → current-source acceptance once → fresh CPU/native
cohort/exclusion checks → new E0 preflight → static generation/scoring → refreshed
generation/scoring → checked metrics and verified archive. Every gate fails closed.
The generator gets only frozen weights and20-term prompts; scoring gets its own
selection truth/candidate projection. Research training and assessment operations
are absent. The E0 runtime also declares verified checkpoint copies, so its
checkpoint roster is not only descriptive metadata.

## Runtime, limits and returned evidence

Plan for roughly **3–5 hours after builds**, with uncertainty in qualification and
original-corpus file I/O/native preparation. The previous256-task generators took
54.8 and60.5 minutes; each scorer took about10 seconds. These observations support
about two hours for inference, not a guarantee for the new roster. The local
13.05-second reference check does **not** measure native execution, workstation I/O,
old-corpus exclusion scanning, or HIP throughput.

The30-hour campaign ceiling is deliberately generous and includes one hour for
report/recovery; image builds precede it. Acceptance keeps its existing395-minute
ceiling. Fresh preparation gets4 hours, preflight30 minutes, each generator6.5
outer hours with6-hour inner role, and each scorer30 outer/20 inner minutes.
Per-target budgets are unchanged. At most64GiB additional run state is allowed;
128GiB free is required initially. No cleanup/pruning is automatic.

On completion, the script prints the exact archive path, normally
`$EVAL_OUTPUT/report.zip`. On failure it prints a verified recovery or emergency
archive instead, with failure status preserved. Share that ZIP. Full checkpoints,
original exclusion contents, new hidden truth and private caches are omitted.
Frozen protocol, candidate programs, attempt/cost records, original qualification
and feature proofs, scores, aggregate metrics and bounded diagnostics are retained.
A completed evaluation with poor model accuracy is not a software failure;
insufficient primary completion remains scientifically inconclusive.

## Local verification and limitations

- All31 pre-existing requirements/design checklist items remain checked; the custom
  implementation checklist has no checkbox gate. No extension hooks are registered.
- **147 affected local tests passed.** Independent policy counterexamples were
  observed failing before implementation.
  Tests cover exact shifted windows/large integers, role/family separation,
  duplicates/shortfalls/deadlines, native mismatch, original input/weight tampering,
  real E0 schema/runtime/checkpoint projections, real container command construction,
  frozen per-arm/cohort role binding, stage ordering, failed-gate suppression,
  private mounts, bounded inner/outer deadlines and verified failure exports.
- The full reference-only reservoir has2,400 admitted parameter candidates and
  2,281 admitted structural candidates from2,288 draws; its quotas are feasible.
  Seven structural draws were rejected by reference admission. The bounded chooser
  selected256 rows. Original-corpus exclusions and native qualification are pending
  the operator's retained files and image; no original assessment values were read
  during this local check.
- Production snapshot remains
  `sha256:d44d83d2a8a88344893a49ccd1b80a3d8b3246a8c86953adf858105cf86dc57d`.
  The previous bank-fix source transition remains the only qualification change.
- No HIP, real native container execution or CI wait occurred here. Prior unrelated
  broader handoff checks requiring unavailable host wasmtime are not relabeled PASS.
