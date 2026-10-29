# Saved-pilot diagnostics handoff

`oeis-learn pipeline pilot-diagnose collect|models` is a read-only post-run
interface. It does not train, resume, fit generated programs, expose final data,
renew historical evaluation clocks or choose a learning strategy. The
[versioned contract](../contracts/pilot-diagnostics.md) specifies populations,
identity checks, numerical policy and evidence bounds.

## One target invocation

Run from the repository root with the full retained run, admission and producer
directories still present. Public ZIPs do not contain saved weights or prepared
visible examples; extracting a public ZIP cannot reconstruct those directories.
This command includes collection, current installed-package/CPU/native/HIP
checks and all six saved-model probes. There is no separate qualification or
calibration handoff. It never starts another learner.

```bash
git pull --ff-only origin 009-pipeline-throughput &&
PYTHONPATH=src python3 -m oeis_learn.cli.main pipeline --engine podman --cpus 16 \
  pilot-diagnose models \
  --run tmp-validation/009-learning-pilot-longrun-20261027T180510Z \
  --admission tmp-validation/009-longrun-admission-20261027T180510Z \
  --inputs tmp-validation/009-learning-pilot-preflight-20261026T213151Z \
  --max-seconds 28800 \
  --output "tmp-validation/009-pilot-diagnostics-$(date -u +%Y%m%dT%H%M%SZ)"
```

The portable entry point uses the same implementation:
`python3 scripts/foundation_run.py --engine podman --cpus 16 pilot-diagnose models`
with the same remaining arguments. Use `--scope-report` to inspect the plan
without starting engines or creating an output. `--device cpu` is an explicit
CPU-only diagnostic; HIP mode never falls back. Docker is also supported.

The eight-hour ceiling includes up to four shared hours for image builds,
qualification, six serial snapshot roles capped at thirty minutes each, and
archive/cleanup reserves. These are ceilings, not measured model-probe costs.
The invocation may finish sooner or produce partial results when bounded work
cannot finish. It checks disk and role memory through the existing workstation90
resource owner; it does not prune images or caches. Keep the original directories
unchanged until diagnostics finish.

Default ordinals are 0 through 5, with initialization always verified. Weight
comparisons use initialization and the previous selected ordinal. Optional
`--snapshots 0 2 5` selects fewer checkpoints and records that reference explicitly.
Optional `--training-rows 4096` uses a deterministic ID-hash subset of training
rows; all held-out development rows and the fixed matched/activation panels stay
planned. The result identifies the subset and does not call it a full-population
estimate. `--spectra` adds bounded strided matrix samples, not full weight SVDs.

## Outputs and interpretation

Return the single printed `ARCHIVE ...zip` after completion. The archive excludes
weights, optimizer state, full examples, program bodies, logits and activation
arrays. No additional data upload is needed for the scalar reports.

- `stages/diagnostics.collect/collected.json` summarizes retained validation and
  training histories, arm outcomes, incomplete/deadline bounds, program diversity,
  generation events and resource/timing audits.
- `stages/diagnostics.import/import.json` records verified source/data/snapshot
  identities and frozen population IDs without embedding private inputs.
- `stages/diagnostics.snapshot-N/snapshot-N/per-example.jsonl` contains canonical
  teacher-forcing losses, nonpadding token accuracy, entropy, exact-token program
  matches and family/length/loop/branch strata.
- The same snapshot directory contains `matched-controls.json`,
  `tensor-comparisons.json`, `activations.json` and the role `summary.json`.
  Matched correct/shuffled/absent inputs share canonical targets. Activation
  summaries include pooled memory variance/similarity and transformer-block
  scalar statistics. Weight summaries include finite values, norms, deltas,
  zero fractions and similarities against both reference checkpoints.
- `stages/study.report/diagnostics.json` combines split/family/stage/stratum
  summaries, matched interventions and paired checkpoint loss/exactness changes.
  Missing rows and snapshots remain explicit.

Small matched and activation panels run before the larger per-example census;
development rows precede training rows. Original role deadlines remain finite.
Canonical token accuracy is not sequence continuation success. Input interventions
and tensor patterns may distinguish conditioning insensitivity from ordinary
training overfit, but do not establish a unique causal mechanism. Original
unmeasured gradients/activations cannot be recovered. Any curriculum/replay or
fresh-data experiment remains a separate decision.

Missing or changed private inputs fail before engine access. Other failed
prerequisites block dependent probes while independent collection/reporting
survives. A role interrupted before publishing its row stream has no recoverable
per-example observations. Preserve its archive and rerun diagnostics into a fresh
output if desired; do not edit the original run, extend historical attempts or
substitute reconstructed weights. No automatic restart occurs.

## Collection alone

For engine-free triage, a verified public directory or ZIP is sufficient:

```bash
PYTHONPATH=src python3 -m oeis_learn.cli.main pipeline pilot-diagnose collect \
  --run tmp-validation/009-learning-pilot-longrun-20261027T180510Z.zip \
  --output "tmp-validation/009-pilot-collection-$(date -u +%Y%m%dT%H%M%SZ)"
```

The model invocation already includes this step. Do not run it separately unless
only historical artifact collection is wanted.

## Implementation validation

All 264 host stdlib tests pass. Host tests cover closed authority, deterministic populations, changed
computational dependencies, actual weight/pointer/audit binding, symlink and ZIP
traversal rejection, exact role mount isolation, main CLI routing, bounded DAGs,
token-weighted grouped reports, missing-input blocking and failure archives.
The new production diagnostic code is source-compatible with the recorded
producer inventory; model/data/codec/decoder dependencies remain unchanged.

Pinned CPU/HIP fixtures execute the actual full-size model with tensor-only
saved weights, assert unchanged parameter digests and test padded token losses,
matched target equality, missing denominators and absent-input representation
statistics. They are authored fixtures, not local CPU/HIP execution evidence.
Target execution/review remains task D007. No CI completion is required before
this handoff.
