# Bootstrap diagnostic repair

The diagnostic must distinguish a model error from incomplete evaluation before
the bigint runtime or encoder architecture changes. The original 32 shortest
admitted examples remain the memorization control, with the existing 31/32
execution gate. A separate short, structurally varied admitted view will test
whether that result extends beyond a single offset template.

The implementation provides:

- Make learnability update and lifetime limits explicit and reproducible.
- Compare cached and uncached free generation on persistent literal failures,
  retaining integer-token, termination and conditioning evidence.
- Report incorrect, incomplete and timed-out attempts separately, including
  coverage of the correct and substituted-prompt evaluations.
- Select a deterministic diverse short-program view from independently admitted
  training examples, without importing OEIS solutions or hidden evaluation truth.
- Add CPU regressions and marked actual-GPU acceptance tests. CPU evidence does
  not establish HIP learnability; the declared GPU gate remains mandatory before
  longer training.

The fixed-width language, model architecture and historical run contracts remain
unchanged. New run controls and diagnostic artifacts are recorded explicitly;
resuming a run cannot reissue its lifetime budget.

## GPU checks and reproduction

Run from the host checkout on the PR branch or after merging. Docker is also
supported by replacing `--engine podman` with `--engine docker`. No host Python
packages, privileged account, VM or SSH integration are required.

```bash
python3 scripts/foundation_run.py --engine podman image ensure
python3 scripts/foundation_run.py --engine podman validate \
  --suite gpu --output tmp-validation-bootstrap-step1
```

The GPU suite includes full-size tri-stream, byte and hybrid checks for repeated
and negative literals, cached/full teacher-forced logits, and actual cached versus
uncached greedy token streams. It does not claim that random full-size models
have learned the memorization task.

Analyze the existing hybrid checkpoint without changing its weights, optimizer,
run artifacts or lifetime ledger:

```bash
python3 scripts/foundation_run.py --engine podman diagnose \
  --run runs/bootstrap-tiny-hybrid-002 \
  --output tmp-analysis/bootstrap-hybrid-step1-diagnosis
```

This selects the source run's chosen checkpoint and compares all its selected
training prompts, including the persistent failures. `diagnostic.json` records
teacher-forced target probabilities, free-generation integer-delimiter and EOS
evidence, the first target-token divergence, and actual cached/uncached token
parity. Results include the observed runtime and original checkpoint identities.
The new diagnostic artifact is not a reinterpretation of the original run score.

The original shortest-example control uses the same sealed prepared inputs:

```bash
python3 scripts/foundation_run.py --engine podman learnability \
  --inputs tmp-bootstrap-inputs --encoder hybrid \
  --view shortest --max-updates 3000 --budget-seconds 7200 \
  --validation-every-updates 100 --diagnostic-seconds 180 \
  --run runs/bootstrap-step1-hybrid-control-001
```

Run the separate diversity diagnostic with a new run name:

```bash
python3 scripts/foundation_run.py --engine podman learnability \
  --inputs tmp-bootstrap-inputs --encoder hybrid \
  --view diverse --max-updates 6000 --budget-seconds 7200 \
  --validation-every-updates 100 --diagnostic-seconds 180 \
  --run runs/bootstrap-step1-hybrid-diverse-001
```

The diverse view chooses 32 existing admitted training rows, each at most 128
body tokens. It uses at least four syntax shapes, at least 16 non-affine visible
behaviors, at most eight offset examples, and distinct visible prefixes. It fails
with a preparation instruction if the sealed pool cannot supply that view. It
does not create hand-written teachers or inspect OEIS truth. Neither tiny run
consumes the final OEIS cohort or claims generalization.

All five controls (`view`, `max_updates`, `budget_seconds`,
`validation_every_updates`, `diagnostic_seconds`) are recorded in the immutable
run contract and initial issuance. Defaults are shortest/3000/3600/100/180.
Update counts are caps; patience and the lifetime reserve can stop a run sooner,
with the reason retained in `early-stop.json`. A changed control requires a new
run. `resume --run ...` retains the original limits and spent lifetime budget.

## Results and analysis packages

`learnability.json` contains the individual gate checks. Passing requires at least
31/32 correct executions, complete conditioning coverage with a drop of at least
four successes, teacher-forced cache equivalence, and complete cached/uncached
free-generation comparisons on all 32 training prompts. Timed-out or unavailable
controls do not pass. Incorrect, incomplete, timeout and execution-failure counts
are reported separately. Generation token limits are distinguished from deadlines.

Exit 0 means the gate passed; exit 2 means measurement completed but the gate did
not pass. Other errors remain execution failures. A completed failed-gate run is
still packaged and retains its checkpoints and evidence.

The wrapper automatically writes `tmp-analysis/<run-name>.zip`, including
`learnability-examples.json` with the selected synthetic training prompts and
targets. It excludes private evaluation truth. Send the original-checkpoint
diagnosis ZIP and both new run ZIPs for analysis. A `pack --checkpoint` invocation
can additionally include the selected weights when necessary; ordinary diagnostic
packages remain small.
